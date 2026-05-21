"""
Métricas cuantitativas para evaluar explicaciones del LLM frente a SHAP y la predicción.
"""
from __future__ import annotations

import re
from typing import Any

from .explainer import (
    SHAP_NEGATIVE_THRESHOLD,
    SHAP_POSITIVE_THRESHOLD,
    StartupExplainer,
    TOP_FEATURES,
)

SUCCESS_TERMS = (
    "éxito",
    "exito",
    "adquis",
    "acquired",
    "salida favorable",
    "apuesta segura",
)
CLOSURE_TERMS = (
    "cierre",
    "fracaso",
    "closed",
    "declive",
    "alto riesgo de cierre",
    "propensa a cerrar",
)
ACTION_TERMS = ("recomend", "estrateg", "accion", "debería", "deberia", "suger")


def select_representative_indices(y_proba: Any) -> dict[str, int]:
    """Misma lógica que notebook 04_shap_explainability."""
    import numpy as np

    proba = np.asarray(y_proba, dtype=float)
    return {
        "exito_alta_confianza": int(np.argmax(proba)),
        "cierre_alta_confianza": int(np.argmin(proba)),
        "borderline": int(np.argmin(np.abs(proba - 0.5))),
    }


def top_shap_features(
    shap_values_dict: dict[str, float],
    *,
    n: int = TOP_FEATURES,
) -> tuple[list[str], list[str]]:
    positive = [
        (f, v)
        for f, v in shap_values_dict.items()
        if v > SHAP_POSITIVE_THRESHOLD
    ]
    negative = [
        (f, v)
        for f, v in shap_values_dict.items()
        if v < SHAP_NEGATIVE_THRESHOLD
    ]
    positive = sorted(positive, key=lambda x: x[1], reverse=True)[:n]
    negative = sorted(negative, key=lambda x: x[1])[:n]
    pos_names = [StartupExplainer.readable_feature_name(f) for f, _ in positive]
    neg_names = [StartupExplainer.readable_feature_name(f) for f, _ in negative]
    return pos_names, neg_names


def _normalize(text: str) -> str:
    return text.lower().replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")


def score_verdict_alignment(explanation: str, success_probability: float) -> float:
    """1.0 si el tono del texto coincide con la predicción; 0.5 ambiguo; 0.0 contradictorio."""
    text = _normalize(explanation)
    predicts_success = success_probability >= 0.5
    success_hits = sum(1 for t in SUCCESS_TERMS if t in text)
    closure_hits = sum(1 for t in CLOSURE_TERMS if t in text)

    if predicts_success:
        if success_hits > closure_hits:
            return 1.0
        if success_hits == closure_hits == 0:
            return 0.5
        return 0.0 if closure_hits > success_hits else 0.5

    if closure_hits > success_hits:
        return 1.0
    if success_hits == closure_hits == 0:
        return 0.5
    return 0.0 if success_hits > closure_hits else 0.5


def score_shap_mention_rate(
    explanation: str,
    shap_values_dict: dict[str, float],
) -> float:
    """Fracción de features SHAP relevantes (top ±) mencionadas en el texto."""
    pos_names, neg_names = top_shap_features(shap_values_dict)
    relevant = [n for n in pos_names + neg_names if n]
    if not relevant:
        return 1.0
    text = _normalize(explanation)
    mentioned = sum(1 for name in relevant if _normalize(name)[:20] in text or any(
        word in text for word in _normalize(name).split()[:3] if len(word) > 4
    ))
    return mentioned / len(relevant)


def score_actionability(explanation: str) -> float:
    """1.0 si hay sección de recomendaciones con verbos de acción."""
    text = _normalize(explanation)
    has_section = bool(re.search(r"recomend|estrateg", text))
    has_actions = any(t in text for t in ACTION_TERMS)
    if has_section and has_actions:
        return 1.0
    if has_section or has_actions:
        return 0.5
    return 0.0


def score_structure(explanation: str) -> float:
    """1.0 si detecta las 4 secciones esperadas del prompt."""
    text = _normalize(explanation)
    markers = [
        "resumen ejecutivo",
        "fortalezas",
        "punto",
        "alerta",
        "recomend",
    ]
    hits = sum(1 for m in markers if m in text)
    return min(1.0, hits / 4.0)


def evaluate_explanation(
    explanation: str,
    features_dict: dict[str, Any],
    shap_values_dict: dict[str, float],
    success_probability: float,
) -> dict[str, float]:
    """Scores automáticos en [0, 1] para el notebook y el informe."""
    return {
        "verdict_alignment": score_verdict_alignment(explanation, success_probability),
        "shap_mention_rate": score_shap_mention_rate(explanation, shap_values_dict),
        "actionability": score_actionability(explanation),
        "structure": score_structure(explanation),
    }


def composite_score(metrics: dict[str, float]) -> float:
    weights = {
        "verdict_alignment": 0.35,
        "shap_mention_rate": 0.35,
        "actionability": 0.15,
        "structure": 0.15,
    }
    return sum(metrics[k] * weights[k] for k in weights)
