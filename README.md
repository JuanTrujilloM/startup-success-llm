# Predicción de Éxito de Startups (Crunchbase) + Explicación con LLM

> Proyecto Final — Inteligencia Artificial · Universidad EAFIT 2026-1

Este proyecto implementa un modelo de Machine Learning (**XGBoost Classifier**) entrenado sobre datos históricos de startups de Crunchbase para predecir si una startup será exitosa (`acquired`) o propensa a cerrar (`closed`). Para cerrar la brecha de confianza de los modelos de caja negra, el sistema calcula valores **SHAP** locales y los integra como contexto enriquecido en un **LLM** (Llama-3.1 vía la API gratuita de Groq), el cual genera informes ejecutivos estructurados en español con el rol de un analista de Capital de Riesgo (VC).

El proyecto se despliega a través de **StartupLens**, una interfaz interactiva en Streamlit con tema oscuro y 5 pestañas de análisis.

**Pregunta de investigación:** ¿Puede un clasificador XGBoost entrenado en datos de Crunchbase predecir el éxito de startups con alta confiabilidad, y puede un LLM (Llama-3.1) generar narrativas explicativas claras y fieles usando valores SHAP locales como contexto?

---

## Integrantes

| Nombre | Correo |
|--------|--------|
| Juan Esteban Trujillo Montes | jetrujillm@eafit.edu.co |
| Andrés Pérez Qunchía | aperezq@eafit.edu.co |
| Sahian Salome Gutierrez Ossa | ssgutierro@eafit.edu.co |

---

## Demo

**Video demo:** [Link al video (YouTube/Drive/Loom)]  
**App interactiva:** [Link a Streamlit Cloud (opcional)]

---

## Estructura del proyecto

```
startup-success-llm/
├── README.md
├── requirements.txt
├── .env.example
│
├── docs/
│   ├── informe_final.pdf             <- PDF compilado desde LaTeX (Overleaf)
│   └── fig_*.png                     <- Gráficos exportados desde notebooks
│
├── notebooks/
│   ├── 01_eda.ipynb                  <- Análisis exploratorio de startups
│   ├── 02_preprocessing.ipynb        <- Limpieza, escalamiento, split, SMOTE
│   ├── 03_modeling.ipynb             <- XGBoost + Baseline (LogReg), métricas
│   ├── 04_shap_explainability.ipynb  <- SHAP global y local (TreeExplainer)
│   └── 05_llm_explanations.ipynb     <- Generación y evaluación de explicaciones LLM
│
├── src/
│   └── llm/
│       ├── explainer.py              <- Cliente Groq y construcción de prompts SHAP
│       └── evaluation.py             <- Métricas cuantitativas para evaluar el LLM
│
├── data/
│   ├── raw/                          <- startup_data.csv (descargar de Kaggle, ver abajo)
│   └── processed/                    <- X_train, X_test, y_train, y_test, shap_values_test
│
├── models/
│   ├── checkpoints/scaler.pkl        <- Escalador StandardScaler (versionado en repo)
│   └── xgboost_model.pkl             <- Modelo entrenado (regenerar con scratch/)
│
├── scratch/
│   ├── recreate_model.py             <- Regenera xgboost_model.pkl desde cero
│   ├── recreate_scaler.py            <- Regenera scaler.pkl desde cero
│   └── test_groq_connection.py       <- Verifica la conexión con la API de Groq
│
└── app/
    └── main.py                       <- StartupLens: demo interactiva en Streamlit
```

---

## Instalación y ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/JuanTrujilloM/startup-success-llm.git
cd startup-success-llm
```

### 2. Crear entorno e instalar dependencias

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\Activate.ps1    # Windows PowerShell

pip install -r requirements.txt
```

### 3. Descargar el dataset

El dataset proviene de Kaggle (datos históricos de startups de Crunchbase):

1. Ir a [Startup Success Prediction — Kaggle](https://www.kaggle.com/datasets/manishkc06/startup-success-prediction)
2. Descargar `startup_data.csv`
3. Colocarlo en `data/raw/startup_data.csv`

> Los datos procesados (`data/processed/`) ya están versionados en el repo, por lo que los notebooks 03–05 pueden ejecutarse sin el CSV original.

### 4. Configurar la API de Groq (gratuita)

1. Crear cuenta en [console.groq.com](https://console.groq.com/keys) y generar una API Key.
2. Copiar el archivo de ejemplo y definir la clave:
   ```bash
   cp .env.example .env
   ```
   ```env
   GROQ_API_KEY=tu_api_key_aqui
   ```
3. Verificar la conexión:
   ```bash
   python scratch/test_groq_connection.py
   ```

### 5. Regenerar artefactos del modelo

El escalador (`models/checkpoints/scaler.pkl`) ya está en el repo. El modelo XGBoost no se versiona por tamaño; regenéralo con:

```bash
python scratch/recreate_model.py
```

### 6. Ejecutar StartupLens

```bash
streamlit run app/main.py
```

La app abre en `http://localhost:8501` con dos modos de entrada (manual y aleatorio) y 5 pestañas de análisis: SHAP, Predicción, Perfil, Métricas del modelo e Informe IA.

---

## Resultados principales

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| Baseline (LogReg) | 60.5% | — | — | — | 0.612 |
| **XGBoost** | **82.2%** | **84.5%** | **86.7%** | **85.6%** | **0.889** |

Dataset: 923 startups · Split 80/20 estratificado · SMOTE en train (477 vs 261 → 477/477)

---

## Stack tecnológico

| Capa | Librerías |
|------|-----------|
| ML & Data | `scikit-learn`, `xgboost`, `imbalanced-learn`, `pandas`, `numpy` |
| Explicabilidad | `shap` (TreeExplainer) |
| LLM | `groq==1.0.0` · Llama-3.1-8b-instant (API gratuita) |
| Visualización | `matplotlib`, `seaborn`, `plotly` |
| App | `streamlit` |
| Python | 3.10+ |
