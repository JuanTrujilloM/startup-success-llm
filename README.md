# Detección de Fraude Financiero + Explicación con LLM

> Proyecto Final — Inteligencia Artificial · Universidad EAFIT 2026-1

Clasificación de transacciones fraudulentas con XGBoost + SMOTE sobre el dataset Credit Card Fraud (Kaggle). Un LLM genera explicaciones en lenguaje natural usando los valores SHAP como contexto.

**Pregunta de investigación:** ¿Puede un clasificador XGBoost detectar fraude con AUC > 0.95 y un LLM generar explicaciones comprensibles usando SHAP values como contexto?

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
fraud-detection/
├── README.md
├── requirements.txt
│
├── docs/
│   └── informe_final.pdf          <- PDF compilado desde LaTeX (Overleaf)
│
├── notebooks/
│   ├── 01_eda.ipynb               <- Análisis exploratorio
│   ├── 02_preprocessing.ipynb     <- Limpieza, SMOTE, split
│   ├── 03_modeling.ipynb          <- XGBoost, umbral, métricas
│   ├── 04_shap_explainability.ipynb  <- SHAP waterfall, feature importance
│   └── 05_llm_explanations.ipynb  <- Prompts + respuestas del LLM
│
├── src/
│   ├── data/
│   │   └── preprocessing.py       <- Pipeline de limpieza reproducible
│   ├── models/
│   │   └── train.py               <- Entrenamiento XGBoost
│   ├── evaluation/
│   │   └── metrics.py             <- AUC-ROC, F1, Precision-Recall
│   └── llm/
│       └── explainer.py           <- Construcción de prompts + llamada al LLM
│
├── data/
│   ├── raw/                       <- creditcard.csv (descargar de Kaggle, ver abajo)
│   └── processed/                 <- datos limpios generados por notebooks
│
├── models/
│   └── checkpoints/               <- xgboost_model.pkl (generado al entrenar)
│
└── app/
    └── main.py                    <- Demo Streamlit (opcional)
```

---

## Instalación y ejecución

### 1. Clonar el repositorio

```bash
git git clone https://github.com/JuanTrujilloM/startup-success-llm.git
cd startup-success-llm
```

### 2. Crear entorno e instalar dependencias

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 3. Descargar el dataset

El dataset no está en el repo por su tamaño (144 MB). Descargarlo manualmente:

1. Ir a https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Descargar `creditcard.csv`
3. Colocarlo en `data/raw/creditcard.csv`

O con la API de Kaggle:
```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw/ --unzip
```

### 4. Configurar la API del LLM (Groq — gratuita)

```bash
# Crear cuenta gratis en https://console.groq.com/keys
# Copiar la API key y crear el archivo .env:
echo "GROQ_API_KEY=tu_api_key_aqui" > .env
```

### 5. Ejecutar los notebooks en orden

```bash
jupyter notebook
```

Correr en secuencia:
1. `notebooks/01_eda.ipynb`
2. `notebooks/02_preprocessing.ipynb`
3. `notebooks/03_modeling.ipynb`
4. `notebooks/04_shap_explainability.ipynb`
5. `notebooks/05_llm_explanations.ipynb`

### 6. (Opcional) Correr la app Streamlit

```bash
streamlit run app/main.py
```

---

## Resultados principales

| Modelo | AUC-ROC | F1 | Recall (fraude) |
|--------|---------|-----|-----------------|
| Baseline (mayoría) | 0.50 | 0.00 | 0.00 |
| Logistic Regression | — | — | — |
| XGBoost + SMOTE | — | — | — |

*(completar con resultados reales)*

---

## Stack tecnológico

- **ML:** `scikit-learn`, `xgboost`, `imbalanced-learn` (SMOTE)
- **Explicabilidad:** `shap`
- **LLM:** `groq` (Llama-3.1-8b-instant, gratuito)
- **Visualización:** `matplotlib`, `seaborn`, `plotly`
- **App:** `streamlit`
- **Python:** 3.10+
