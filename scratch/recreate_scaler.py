"""
Regenera models/checkpoints/scaler.pkl replicando el notebook 02_preprocessing.

Requiere: data/raw/startup_data.csv (Kaggle — ver README).
No modifica data/processed/*.csv; solo recrea el StandardScaler ajustado en train (pre-SMOTE).
"""
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "startup_data.csv"
OUT_PATH = ROOT / "models" / "checkpoints" / "scaler.pkl"

COLS_DROP = [
    "Unnamed: 0",
    "Unnamed: 6",
    "id",
    "name",
    "object_id",
    "zip_code",
    "closed_at",
    "state_code.1",
    "state_code",
    "city",
    "latitude",
    "longitude",
    "founded_at",
    "first_funding_at",
    "last_funding_at",
]
COLS_SCALE = [
    "funding_total_usd",
    "age_first_funding_year",
    "age_last_funding_year",
    "age_first_milestone_year",
    "age_last_milestone_year",
    "avg_participants",
]


def load_and_prepare() -> tuple[pd.DataFrame, pd.Series]:
    if not RAW_PATH.is_file():
        raise FileNotFoundError(
            f"No se encontró {RAW_PATH}.\n"
            "Descarga startup_data.csv desde Kaggle y colócalo en data/raw/ "
            "(instrucciones en README.md sección 3)."
        )

    df = pd.read_csv(RAW_PATH)
    df_clean = df.drop(columns=COLS_DROP, errors="ignore")
    df_clean = df_clean.drop(columns=["category_code", "status"], errors="ignore")
    df_clean["age_first_milestone_year"] = df_clean["age_first_milestone_year"].fillna(0)
    df_clean["age_last_milestone_year"] = df_clean["age_last_milestone_year"].fillna(0)

    X = df_clean.drop("labels", axis=1)
    y = df_clean["labels"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def main() -> None:
    os.chdir(ROOT)
    os.makedirs(OUT_PATH.parent, exist_ok=True)

    print("Cargando y preprocesando datos (mismo flujo que notebook 02)...")
    X_train, X_test, y_train, y_test = load_and_prepare()
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[COLS_SCALE] = scaler.fit_transform(X_train[COLS_SCALE])
    X_test_scaled[COLS_SCALE] = scaler.transform(X_test[COLS_SCALE])

    # Validar contra X_test.csv ya procesado en el repo
    X_test_saved = pd.read_csv(ROOT / "data/processed" / "X_test.csv")
    diff = (X_test_scaled[COLS_SCALE].values - X_test_saved[COLS_SCALE].values)
    max_diff = abs(diff).max()
    print(f"  Máx. diferencia vs data/processed/X_test.csv (cols escaladas): {max_diff:.6e}")
    if max_diff > 1e-5:
        print(
            "  ADVERTENCIA: el scaler no coincide con los CSV procesados del repo. "
            "Revisa que startup_data.csv sea el mismo del equipo."
        )
    else:
        print("  Coincide con X_test.csv del repositorio.")

    joblib.dump(scaler, OUT_PATH)
    print(f"\nGuardado: {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
