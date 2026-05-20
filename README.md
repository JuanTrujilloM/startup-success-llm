# Predicción de Éxito de Startups (Crunchbase) + Explicación con LLM

> Proyecto Final — Inteligencia Artificial · Universidad EAFIT 2026-1

Este proyecto implementa un modelo de Machine Learning (**XGBoost Classifier**) entrenado sobre datos de startups de Crunchbase para predecir si una startup será exitosa (adquirida - `acquired`) o propensa a cerrar (`closed`). Para cerrar la brecha de confianza de los modelos de caja negra, el sistema calcula valores **SHAP** locales y los integra como contexto enriquecido en un **LLM** (vía la API gratuita de Groq), el cual genera explicaciones estructuradas en lenguaje natural en español con el rol de un analista de Capital de Riesgo (VC).

El proyecto se despliega a través de una **interfaz interactiva en Streamlit**.

**Pregunta de investigación:** ¿Puede un clasificador XGBoost entrenado en datos de Crunchbase predecir el éxito de startups con alta confiabilidad y un LLM (Llama-3) generar narrativas explicativas claras y fieles usando valores SHAP locales como contexto?

---

## Integrantes

| Nombre | Correo |
|--------|--------|
| Nombre Apellido | correo@eafit.edu.co |
| Nombre Apellido | correo@eafit.edu.co |
| Nombre Apellido | correo@eafit.edu.co |

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
│   ├── informe_final.pdf          <- PDF compilado desde LaTeX (Overleaf)
│   └── fig_*.png                  <- Gráficos exploratorios y curvas ROC
│
├── notebooks/
│   ├── 01_eda.ipynb               <- Análisis exploratorio de startups
│   ├── 02_preprocessing.ipynb     <- Limpieza, escalamiento, split
│   ├── 03_modeling.ipynb          <- Entrenamiento de XGBoost y Baseline (LogReg)
│   ├── 04_shap_explainability.ipynb  <- Valores SHAP globales y locales
│   └── 05_llm_explanations.ipynb  <- Generación de explicaciones de Groq (Llama-3)
│
├── src/
│   └── llm/
│       └── explainer.py           <- Conectividad con Groq y prompt engineering
│
├── data/
│   ├── raw/                       <- startup_data.csv (descargar de Kaggle, ver abajo)
│   └── processed/                 <- Datos limpios (X_train.csv, X_test.csv, y_train.csv, y_test.csv, shap_values_test.csv)
│
├── models/
│   ├── checkpoints/               <- scaler.pkl (escalador de variables)
│   └── xgboost_model.pkl          <- Pesos entrenados del clasificador
│
└── app/
    └── main.py                    <- Demo interactiva en Streamlit (explicación + SHAP)
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
# Activar entorno:
# En Windows (PowerShell):
venv\Scripts\Activate.ps1
# En Linux/macOS:
source venv/bin/activate

# Instalar los paquetes principales:
pip install -r requirements.txt
```

*(Nota para Windows: si la instalación completa falla debido a límites de ruta larga de Windows con JupyterLab, puedes instalar únicamente el stack principal con: `pip install scikit-learn xgboost imbalanced-learn numpy pandas shap groq python-dotenv matplotlib seaborn plotly streamlit joblib scipy`)*

### 3. Descargar el dataset

El dataset original proviene de Kaggle y corresponde a datos históricos de startups de Crunchbase:

1. Ir a [Startup Success Prediction en Kaggle](https://www.kaggle.com/datasets/manishkc06/startup-success-prediction)
2. Descargar `startup_data.csv`
3. Colocar el archivo en `data/raw/startup_data.csv`

### 4. Configurar la API del LLM (Groq — gratuita)

Para generar las explicaciones en lenguaje natural, utilizamos la API de Groq (con el modelo `llama-3.1-8b-instant` o similar):

1. Regístrate de manera 100% gratuita en [Groq Console](https://console.groq.com/keys).
2. Crea una API Key y cópiala.
3. Duplica el archivo `.env.example` y cámbiale el nombre a `.env`:
   ```bash
   cp .env.example .env
   ```
4. Abre `.env` y define tu clave:
   ```env
   GROQ_API_KEY=tu_api_key_aqui
   ```

### 5. Artefactos locales (modelo y escalador)

El **escalador** (`models/checkpoints/scaler.pkl`) ya está versionado en el repo para que Streamlit pueda normalizar entradas nuevas. El **modelo XGBoost** (`models/xgboost_model.pkl`) no se sube a Git; regenéralo con:

```bash
python scratch/recreate_model.py
```

Si cambias el preprocesamiento y tienes `data/raw/startup_data.csv`, puedes regenerar también el escalador:

```bash
python scratch/recreate_scaler.py
```


### 6. Ejecutar la demo Streamlit

Para lanzar el panel interactivo localmente:

```bash
streamlit run app/main.py
```

---

## Resultados principales

| Modelo | AUC-ROC (Test) | Accuracy (Test) | F1-Score (Test) | Recall (Éxito) |
|--------|----------------|-----------------|-----------------|----------------|
| Baseline (LogReg) | 0.6120 | 0.6054 | — | — |
| XGBoost + Hyperparameters | **0.8260** | **0.7892** | **0.8408** | **0.8583** |

---

## Stack tecnológico

- **ML & Data:** `scikit-learn`, `xgboost`, `pandas`, `numpy`
- **Explicabilidad:** `shap` (TreeExplainer)
- **LLM API:** `groq` (Llama 3.1, gratuito y ultrarrápido)
- **Visualización:** `matplotlib`, `seaborn`, `plotly` (gráficos interactivos)
- **Frontend App:** `streamlit` (interfaz)
- **Python:** 3.10+
