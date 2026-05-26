import joblib
import pandas as pd
import os

print("="*60)
print(" COMPARACION DE REPRODUCIBILIDAD (NUEVA EJECUCION vs BACKUP)")
print("="*60)

# 1. Comparar best_params.pkl
new_params_path = 'models/trained_models/best_params.pkl'
bak_params_path = 'backup/models/trained_models/best_params.pkl'

if os.path.exists(new_params_path) and os.path.exists(bak_params_path):
    new_params = joblib.load(new_params_path)
    bak_params = joblib.load(bak_params_path)
    
    print("\n--- 1. Hiperparametros Optimizados (Optuna) ---")
    if new_params == bak_params:
        print("[IDENTICOS]: Optuna llego exactamente a la misma configuracion ganadora.")
        print(f"   Modelo Ganador: {new_params.get('classifier')}")
    else:
        print("[DIFERENTES]:")
        print(f"   Nuevo:  {new_params}")
        print(f"   Backup: {bak_params}")
else:
    print("\n--- 1. Hiperparametros ---")
    print("[ERROR] Falta best_params.pkl en alguna de las carpetas.")

# 2. Comparar reportes de clasificación
new_rep_path = 'results/metrics/classification_report.csv'
bak_rep_path = 'backup/results/metrics/classification_report.csv'

if os.path.exists(new_rep_path) and os.path.exists(bak_rep_path):
    new_df = pd.read_csv(new_rep_path, index_col=0)
    bak_df = pd.read_csv(bak_rep_path, index_col=0)
    
    # Redondear a 4 decimales
    new_df = new_df.round(4)
    bak_df = bak_df.round(4)
    
    print("\n--- 2. Metricas del Modelo en Datos de Prueba ---")
    if new_df.equals(bak_df):
        print("[IDENTICAS]: El modelo final predice con exactamente la misma precision.")
        print(f"   F1-Macro: {new_df.loc['macro avg', 'f1-score']}")
    else:
        print("[DIFERENTES]:")
        print("--- Nuevo F1-Score ---")
        print(new_df[['f1-score']])
        print("--- Backup F1-Score ---")
        print(bak_df[['f1-score']])
else:
    print("\n--- 2. Metricas del Modelo ---")
    print("[ERROR] Falta classification_report.csv en alguna de las carpetas.")

print("\n" + "="*60)
