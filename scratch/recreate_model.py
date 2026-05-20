import os
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

def main():
    # Asegurar que el directorio de salida de modelos exista
    os.makedirs('models', exist_ok=True)
    
    print("Cargando datos preprocesados...")
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
    y_test = pd.read_csv('data/processed/y_test.csv').squeeze()
    
    print(f"Dimensiones de entrenamiento: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"Dimensiones de prueba: X_test {X_test.shape}, y_test {y_test.shape}")
    
    print("Inicializando clasificador XGBoost con los hiperparámetros del notebook 03...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric='logloss'
    )
    
    print("Entrenando el modelo en todo el conjunto de entrenamiento...")
    xgb_model.fit(X_train, y_train)
    
    # Predecir probabilidades y clases
    y_pred = xgb_model.predict(X_test)
    y_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    # Calcular métricas
    test_auc = roc_auc_score(y_test, y_proba)
    test_acc = accuracy_score(y_test, y_pred)
    test_prec = precision_score(y_test, y_pred)
    test_rec = recall_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    
    print("\n=== Métricas obtenidas en el Test Set ===")
    print(f"AUC-ROC:  {test_auc:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision:{test_prec:.4f}")
    print(f"Recall:   {test_rec:.4f}")
    print(f"F1-Score: {test_f1:.4f}")
    
    # Guardar el modelo
    model_path = 'models/xgboost_model.pkl'
    print(f"\nGuardando el modelo en {model_path}...")
    joblib.dump(xgb_model, model_path)
    print("¡Modelo guardado exitosamente!")

if __name__ == '__main__':
    main()
