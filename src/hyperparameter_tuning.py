
#Hay unos comentarios medios raros que son del código del profe, los dejé porque son útiles para entender que hacen


import os
import logging
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib


# 1. configuración

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)

# Rutas del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
#joblib
BEST_PARAMS_FILE = os.path.join(MODELS_DIR, "best_params.pkl")


# 2. función objetivo optuna

def objective(trial, X, y):
    steps = []
    
    # 1. Optuna decide si usar PCA o no en este intento

    use_pca = trial.suggest_categorical("use_pca", [True, False])
    
    if use_pca:

        # Si decide usarlo, que decida cuánta varianza retener

        pca_variance = trial.suggest_categorical("pca_variance", [0.85, 0.90, 0.95, 0.99])
        steps.append(('pca', PCA(n_components=pca_variance, random_state=42)))
        
    # 2. Optuna decide el clasificador

    classifier_name = trial.suggest_categorical("classifier", ["SVM", "RandomForest", "XGBoost", "LightGBM", "LogisticRegression"])
    
    if classifier_name == "LogisticRegression":
        c_lr = trial.suggest_float("lr_C", 1e-3, 1e2, log=True)
        model = LogisticRegression(C=c_lr, max_iter=1000, class_weight='balanced', random_state=42)
        
    elif classifier_name == "RandomForest":
        rf_n_estimators = trial.suggest_int("rf_n_estimators", 50, 300)
        rf_max_depth = trial.suggest_int("rf_max_depth", 5, 30)
        model = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth, class_weight='balanced', random_state=42, n_jobs=-1)
        
    elif classifier_name == "SVM":
        svm_c = trial.suggest_float("svm_C", 0.1, 100.0, log=True)
        svm_kernel = trial.suggest_categorical("svm_kernel", ["rbf", "linear"])
        model = SVC(C=svm_c, kernel=svm_kernel, gamma="scale", class_weight='balanced', random_state=42)
        
    elif classifier_name == "XGBoost":
        xgb_n_estimators = trial.suggest_int("xgb_n_estimators", 50, 200)
        xgb_lr = trial.suggest_float("xgb_learning_rate", 1e-3, 0.3, log=True)
        model = XGBClassifier(n_estimators=xgb_n_estimators, learning_rate=xgb_lr, random_state=42, eval_metric='mlogloss', n_jobs=-1)
        
    else: # LightGBM
        lgb_n_estimators = trial.suggest_int("lgb_n_estimators", 50, 200)
        lgb_lr = trial.suggest_float("lgb_learning_rate", 1e-3, 0.3, log=True)
        model = LGBMClassifier(n_estimators=lgb_n_estimators, learning_rate=lgb_lr, class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
        
    # Armamos el pipeline con los pasos elegidos

    steps.append(('classifier', model))
    pipeline = Pipeline(steps)
    
    # Cross Validation (3 splits para que sea rápido)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    score = cross_val_score(pipeline, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)
    
    return score.mean()



# 3. carga de datos

def run_hyperparameter_tuning() -> None:

    print("Iniciando Optuna para Modelos, Hiperparámetros y PCA...")

    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).squeeze("columns")
    
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=30) 

    print("\n" + "="*60)
    print(f"RESULTADO GANADOR:")
    print(f"Accuracy (CV): {study.best_value:.4f}")
    print(f"Parámetros: {study.best_params}")
    print("="*60 + "\n")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(study.best_params, BEST_PARAMS_FILE)
    print(f"Mejores parámetros guardados en {BEST_PARAMS_FILE}")

if __name__ == "__main__":
    run_hyperparameter_tuning()

