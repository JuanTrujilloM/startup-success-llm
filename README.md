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

**Video demo:** [Ver en Google Drive](https://drive.google.com/file/d/1M2IBgvLNv2Kt8kTV9EgoKDL8IiRVX55Z/view?usp=drivesdk)  

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
│   ├── guia_usuario.md               <- Guía de uso de la aplicación
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

> Flujo probado esperado: usar Python 3.10 o 3.11, instalar dependencias, regenerar el modelo si no existe y ejecutar la app con Streamlit. La API de Groq solo es necesaria para la pestaña **Informe IA**; el resto de la aplicación funciona sin esa clave.

### 1. Clonar el repositorio

```bash
git clone https://github.com/JuanTrujilloM/startup-success-llm.git
cd startup-success-llm
```

### 2. Crear entorno e instalar dependencias

Linux/macOS:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Descargar el dataset

El dataset proviene de Kaggle (datos históricos de startups de Crunchbase):

1. Ir a [Startup Success Prediction — Kaggle](https://www.kaggle.com/datasets/manishkc06/startup-success-prediction)
2. Descargar `startup_data.csv`
3. Colocarlo en `data/raw/startup_data.csv`

> Los datos procesados (`data/processed/`) ya están versionados en el repo, por lo que la app, el script `scratch/recreate_model.py` y los notebooks 03–05 pueden ejecutarse sin el CSV original. El CSV crudo solo es necesario para reproducir desde cero el notebook 01, el notebook 02 o `scratch/recreate_scaler.py`.

### 4. Configurar la API de Groq (opcional, gratuita)

Este paso solo es obligatorio si se quiere usar la pestaña **Informe IA**.

1. Crear cuenta en [console.groq.com](https://console.groq.com/keys) y generar una API Key.
2. Copiar el archivo de ejemplo y definir la clave:

   Linux/macOS:

   ```bash
   cp .env.example .env
   ```

   Windows PowerShell:

   ```powershell
   Copy-Item .env.example .env
   ```

   ```env
   GROQ_API_KEY=tu_api_key_aqui
   ```
3. Verificar la conexión:
   ```bash
   python scratch/test_groq_connection.py
   ```

### 5. Regenerar artefactos del modelo

El escalador (`models/checkpoints/scaler.pkl`) ya está en el repo. Si `models/xgboost_model.pkl` no existe después de clonar el repositorio, regenéralo con:

```bash
python scratch/recreate_model.py
```

### 6. Ejecutar StartupLens

```bash
streamlit run app/main.py
```

La app abre en `http://localhost:8501` con dos modos de entrada (manual y aleatorio) y 5 pestañas de análisis: SHAP, Predicción, Perfil, Métricas del modelo e Informe IA.

### 7. Verificación rápida

Para comprobar que el código principal importa correctamente:

```bash
python -m py_compile app/main.py src/llm/explainer.py src/llm/evaluation.py scratch/recreate_model.py
```

Para instrucciones de uso de la interfaz, consultar [`docs/guia_usuario.md`](docs/guia_usuario.md).

---

## Resultados principales

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| Baseline (LogReg) | — | 79.0% | 81.0% | 80.0% | 0.7927 |
| **XGBoost** | **78.9%** | **82.4%** | **85.8%** | **84.1%** | **0.8241** |

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
| Python | 3.10 o 3.11 recomendado |

---

## Despliegue en Streamlit Community Cloud

La app se despliega gratis y sin tarjeta en [share.streamlit.io](https://share.streamlit.io).

1. Entra con tu cuenta de GitHub y elige **Create app** → este repositorio.
2. Configura:
   - **Main file path:** `app/main.py`
   - **Python version:** `3.11` (en *Advanced settings*). Es obligatorio:
     `numpy==1.26.4` no tiene wheels para 3.12+ y la instalación falla.
3. En *Advanced settings* → **Secrets**, pega tu clave de
   [console.groq.com](https://console.groq.com/keys):

   ```toml
   GROQ_API_KEY = "gsk_..."
   ```

4. **Deploy**. El primer arranque tarda unos minutos instalando dependencias.

Sin la clave la app funciona igual — predicción, métricas y gráficos SHAP — y
solo la pestaña de informe LLM muestra un aviso de configuración.

### Notas

`models/xgboost_model.pkl` (285 KB) está versionado a propósito, como excepción
al `.gitignore`: `app/main.py` lo carga al arrancar y sin él el despliegue no
levanta. Se regenera con `notebooks/03_modeling.ipynb`.

En local la clave se toma de `.env` vía `python-dotenv`; en la nube, del secreto
de Streamlit. `app/main.py` puentea el segundo caso hacia `os.environ`, que es
donde `src/llm/explainer.py` la busca.
