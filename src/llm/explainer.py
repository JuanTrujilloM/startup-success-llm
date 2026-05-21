# Infraestructura LLM — cliente Groq y construcción de prompts SHAP para StartupExplainer
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import Groq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "llama-3.1-8b-instant"
SHAP_POSITIVE_THRESHOLD = 0.05
SHAP_NEGATIVE_THRESHOLD = -0.05
TOP_FEATURES = 5

FEATURE_LABELS: dict[str, str] = {
    "age_first_funding_year": "Edad al recibir primer financiamiento (años)",
    "age_last_funding_year": "Edad al recibir último financiamiento (años)",
    "age_first_milestone_year": "Edad al alcanzar primer hito (años)",
    "age_last_milestone_year": "Edad al alcanzar último hito (años)",
    "relationships": "Número de relaciones/contactos clave de los fundadores",
    "funding_rounds": "Total de rondas de financiamiento recibidas",
    "funding_total_usd": "Monto total financiado en USD",
    "milestones": "Número total de hitos/milestones alcanzados",
    "is_CA": "Ubicación en California (CA)",
    "is_NY": "Ubicación en Nueva York (NY)",
    "is_MA": "Ubicación en Massachusetts (MA)",
    "is_TX": "Ubicación en Texas (TX)",
    "is_otherstate": "Ubicación en otros estados de EE.UU.",
    "is_software": "Sector: Software",
    "is_web": "Sector: Web",
    "is_mobile": "Sector: Mobile",
    "is_enterprise": "Sector: Enterprise",
    "is_advertising": "Sector: Publicidad/Advertising",
    "is_gamesvideo": "Sector: Videojuegos/Games & Video",
    "is_ecommerce": "Sector: E-commerce",
    "is_biotech": "Sector: Biotecnología/Biotech",
    "is_consulting": "Sector: Consultoría",
    "is_othercategory": "Sector: Otras Industrias",
    "has_VC": "Financiada por Venture Capital",
    "has_angel": "Financiada por inversionistas Ángeles",
    "has_roundA": "Completó Ronda A",
    "has_roundB": "Completó Ronda B",
    "has_roundC": "Completó Ronda C",
    "has_roundD": "Completó Ronda D",
    "avg_participants": "Promedio de participantes por ronda",
    "is_top500": "Startup listada en el Top 500",
}

SYSTEM_PROMPT = (
    "Eres un analista experto en Venture Capital (Capital de Riesgo) y científico de datos. "
    "Tu especialidad es evaluar startups tecnológicas e interpretar modelos predictivos complejos. "
    "Tu tono debe ser analítico, sumamente profesional, perspicaz y comprensible para inversionistas financieros."
)


class GroqConnectionError(RuntimeError):
    # Error al validar o usar la API de Groq
    pass


class StartupExplainer:
    # Cliente Groq + prompts System/User a partir de SHAP locales

    def __init__(self, model_name: str = DEFAULT_MODEL, api_key: str | None = None):
        load_dotenv(PROJECT_ROOT / ".env")
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self._client: Groq | None = None

    @property
    def client(self) -> Groq:
        if self._client is None:
            if not self.api_key or not str(self.api_key).strip():
                raise GroqConnectionError(
                    "GROQ_API_KEY no configurada. Copia .env.example a .env y define tu clave "
                    "(https://console.groq.com/keys)."
                )
            self._client = Groq(api_key=self.api_key.strip())
        return self._client

    def validate_connection(self) -> str:
        # Comprueba API Key y cliente Groq con una petición mínima; lanza GroqConnectionError si falla
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": "Responde solo: OK"}],
                model=self.model_name,
                max_tokens=8,
                temperature=0,
            )
        except GroqConnectionError:
            raise
        except Exception as exc:
            raise GroqConnectionError(
                f"No se pudo conectar con Groq (modelo={self.model_name}): {exc}"
            ) from exc

        content = (response.choices[0].message.content or "").strip()
        if not content:
            raise GroqConnectionError("Groq respondió vacío; revisa cuota o estado del servicio.")
        return f"Conexión OK — modelo={self.model_name}, respuesta={content!r}"

    @staticmethod
    def readable_feature_name(feature_name: str) -> str:
        return FEATURE_LABELS.get(feature_name, feature_name)

    @staticmethod
    def format_feature_value(feature_name: str, value: Any) -> str:
        if feature_name.startswith(("is_", "has_")):
            return "Sí" if float(value) > 0 else "No"
        if feature_name == "funding_total_usd":
            if float(value) < 10:
                return f"{float(value):.4f} (valor escalado)"
            return f"${float(value):,.2f} USD"
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    def _rank_shap_impacts(
        self,
        features_dict: dict[str, Any],
        shap_values_dict: dict[str, float],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        positive: list[dict[str, Any]] = []
        negative: list[dict[str, Any]] = []

        for feature, shap_val in shap_values_dict.items():
            item = {
                "name": self.readable_feature_name(feature),
                "val": self.format_feature_value(feature, features_dict.get(feature, 0)),
                "shap": float(shap_val),
            }
            if shap_val > SHAP_POSITIVE_THRESHOLD:
                positive.append(item)
            elif shap_val < SHAP_NEGATIVE_THRESHOLD:
                negative.append(item)

        positive = sorted(positive, key=lambda x: x["shap"], reverse=True)[:TOP_FEATURES]
        negative = sorted(negative, key=lambda x: x["shap"])[:TOP_FEATURES]
        return positive, negative

    @staticmethod
    def _impacts_to_markdown(
        impacts: list[dict[str, Any]],
        direction: str,
    ) -> str:
        if not impacts:
            if direction == "positive":
                return "- Ninguno con impacto significativamente positivo."
            return "- Ninguno con impacto significativamente negativo."

        lines = []
        for item in impacts:
            if direction == "positive":
                lines.append(
                    f"- **{item['name']}** (Valor: {item['val']}): "
                    f"Aumentó la probabilidad de éxito (SHAP: +{item['shap']:.3f})"
                )
            else:
                lines.append(
                    f"- **{item['name']}** (Valor: {item['val']}): "
                    f"Disminuyó la probabilidad de éxito (SHAP: {item['shap']:.3f})"
                )
        return "\n".join(lines)

    def build_prompts(
        self,
        features_dict: dict[str, Any],
        shap_values_dict: dict[str, float],
        success_probability: float,
    ) -> tuple[str, str]:
        # Construye los mensajes System y User sin llamar a la API; retorna (system_prompt, user_prompt)
        positive, negative = self._rank_shap_impacts(features_dict, shap_values_dict)
        pos_text = self._impacts_to_markdown(positive, "positive")
        neg_text = self._impacts_to_markdown(negative, "negative")

        prob_percentage = success_probability * 100
        verdict = "ÉXITO / ADQUISICIÓN" if prob_percentage >= 50 else "CIERRE / FRACASO"

        user_prompt = f"""
Has recibido una predicción de un modelo predictivo XGBoost entrenado sobre datos históricos de Crunchbase.
Tu tarea es explicar esta predicción cuantitativa usando la atribución de variables SHAP (explicabilidad local) y redactar un informe ejecutivo estructurado en español.

### Datos de la Startup y Predicción del Modelo:
- **Predicción Final:** La startup es clasificada como propensa a: **{verdict}**
- **Probabilidad de Éxito Estimada por el Modelo:** **{prob_percentage:.2f}%**

### Factores Positivos de Mayor Peso (Valores SHAP > 0):
{pos_text}

### Factores de Riesgo / Negativos de Mayor Peso (Valores SHAP < 0):
{neg_text}

---

Redacta un informe de análisis de 4 secciones en Markdown:

1. **Resumen Ejecutivo**
2. **Fortalezas del Ecosistema y Operación (SHAP positivos)**
3. **Puntos Críticos de Alerta (SHAP negativos)**
4. **Recomendaciones Estratégicas del VC** (2 a 3 acciones concretas)
""".strip()

        return SYSTEM_PROMPT, user_prompt

    def explain_startup(
        self,
        features_dict: dict[str, Any],
        shap_values_dict: dict[str, float],
        success_probability: float,
        *,
        temperature: float = 0.4,
    ) -> str:
        # Genera explicación narrativa vía Groq; lanza GroqConnectionError si la API falla
        system_prompt, user_prompt = self.build_prompts(
            features_dict, shap_values_dict, success_probability
        )
        prob_percentage = success_probability * 100
        verdict = "ÉXITO / ADQUISICIÓN" if prob_percentage >= 50 else "CIERRE / FRACASO"
        positive, negative = self._rank_shap_impacts(features_dict, shap_values_dict)
        pos_text = self._impacts_to_markdown(positive, "positive")
        neg_text = self._impacts_to_markdown(negative, "negative")

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=self.model_name,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except GroqConnectionError:
            raise
        except Exception as exc:
            raise GroqConnectionError(f"Error al generar explicación: {exc}") from exc
