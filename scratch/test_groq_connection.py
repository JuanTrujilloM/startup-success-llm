# Prueba aislada de conexión Groq — ejecutar desde raíz: python scratch/test_groq_connection.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm.explainer import GroqConnectionError, StartupExplainer


def main() -> None:
    print("=== Test conexión Groq (StartupExplainer) ===\n")
    explainer = StartupExplainer()

    try:
        msg = explainer.validate_connection()
        print(msg)
    except GroqConnectionError as exc:
        print(f"FALLO: {exc}")
        sys.exit(1)

    # Probar build_prompts sin API (datos ficticios mínimos)
    features = {"relationships": 10, "milestones": 3, "is_CA": 1}
    shap = {"relationships": 0.12, "milestones": 0.08, "is_CA": -0.02}
    system, user = explainer.build_prompts(features, shap, 0.87)
    print(f"\nPrompt system ({len(system)} chars): OK")
    print(f"Prompt user ({len(user)} chars): OK")
    print("\n=== Infraestructura LLM lista ===")


if __name__ == "__main__":
    main()
