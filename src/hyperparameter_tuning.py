# Hay unos comentarios medios raros que son del código del profe, los dejé porque son útiles para entender que hacen
# [Añadidos comentarios explicativos en todo el script para clarificar la lógica de Optuna y de los Modelos]

import os
import logging
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib


# ==========================================
# 1. Configuración de Entorno y Rutas
# ==========================================

# Configurar logs para visualizar el avance del script de manera limpia en terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)

# Definir rutas absolutas para evitar problemas al ejecutar el script desde distintas terminales
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Archivo final donde Optuna guardará la configuración ganadora
BEST_PARAMS_FILE = os.path.join(MODELS_DIR, "best_params.pkl")


# ==========================================
# 2. Función Objetivo de Optuna (El "Cerebro")
# ==========================================

def objective(trial, X, y):
    """
    Esta función es el núcleo de Optuna. Se ejecuta repetidamente (n_trials veces).
    En cada iteración ('trial'), Optuna 'sugiere' combinaciones de hiperparámetros de manera inteligente.
    """
    steps = [] # Lista secuencial que almacenará los componentes del Pipeline predictivo
    
    # ------------------------------------------------------
    # A. Decisión sobre Reducción de Dimensionalidad (PCA)
    # ------------------------------------------------------
    # 1. Optuna decide en cada intento si vale la pena usar PCA o no
    use_pca = trial.suggest_categorical("use_pca", [True, False])
    
    if use_pca:
        # Si decide usarlo, también evalúa cuánta varianza de los datos es óptimo retener
        pca_variance = trial.suggest_categorical("pca_variance", [0.85, 0.90, 0.95, 0.99])
        steps.append(('pca', PCA(n_components=pca_variance, random_state=42)))
        
    # ------------------------------------------------------
    # B. Elección del Algoritmo Clasificador Principal
    # ------------------------------------------------------
    # 2. Optuna elige uno de los 5 algoritmos disponibles en su catálogo
    classifier_name = trial.suggest_categorical("classifier", ["SVM", "RandomForest", "XGBoost", "LightGBM", "LogisticRegression"])
    
    # Dependiendo de cuál elija, procedemos a afinar las "perillas" de ese algoritmo en específico
    if classifier_name == "LogisticRegression":
        # suggest_float con log=True busca en saltos exponenciales (muy pequeños a muy grandes)
        c_lr = trial.suggest_float("lr_C", 1e-3, 1e2, log=True)
        # class_weight='balanced' es esencial para contrarrestar nuestro desbalance de target (70% vs 30%)
        model = LogisticRegression(C=c_lr, max_iter=1000, class_weight='balanced', random_state=42)
        
    elif classifier_name == "RandomForest":
        # suggest_int busca el número ideal de árboles y su profundidad
        rf_n_estimators = trial.suggest_int("rf_n_estimators", 50, 300)
        rf_max_depth = trial.suggest_int("rf_max_depth", 5, 30)
        model = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth, class_weight='balanced', random_state=42, n_jobs=-1)
        
    elif classifier_name == "SVM":
        svm_c = trial.suggest_float("svm_C", 0.1, 100.0, log=True)
        svm_kernel = trial.suggest_categorical("svm_kernel", ["rbf", "linear"])
        model = SVC(C=svm_c, kernel=svm_kernel, gamma="scale", class_weight='balanced', random_state=42, probability=True)

    elif classifier_name == "XGBoost":
        xgb_n_estimators = trial.suggest_int("xgb_n_estimators", 50, 200)
        xgb_lr = trial.suggest_float("xgb_learning_rate", 1e-3, 0.3, log=True)
        model = XGBClassifier(n_estimators=xgb_n_estimators, learning_rate=xgb_lr, random_state=42, eval_metric='mlogloss', n_jobs=-1)
        
    else: # LightGBM (El único que sobra)
        lgb_n_estimators = trial.suggest_int("lgb_n_estimators", 50, 200)
        lgb_lr = trial.suggest_float("lgb_learning_rate", 1e-3, 0.3, log=True)
        model = LGBMClassifier(n_estimators=lgb_n_estimators, learning_rate=lgb_lr, class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
        
    # ------------------------------------------------------
    # C. Ensamblado y Evaluación de este "Trial"
    # ------------------------------------------------------
    # Armamos el pipeline integrando el PCA (si hubo) con el clasificador
    steps.append(('classifier', model))
    pipeline = Pipeline(steps)
    
    # Cross Validation (3 splits, o cortes, de la data para probar rápido y no sobreajustar)
    # StratifiedKFold garantiza que al partir los datos, los 3 pedazos tengan exactamente un 30% de rentables
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # Entrenamos y calificamos.
    # Usar 'f1_macro' es obligatorio aquí. Evita que el modelo nos mienta sacando 70%
    # de Accuracy por el simple hecho de decir siempre "Clase 0"
    score = cross_val_score(pipeline, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)
    
    # Optuna guardará este número promedio y aprenderá de él para la siguiente iteración
    return score.mean()


# ==========================================
# 3. Orquestador de la Búsqueda
# ==========================================

def run_hyperparameter_tuning() -> None:
    print("Iniciando Optuna para Modelos, Hiperparámetros y PCA...")

    # Cargar datos limpios de entrenamiento (el Output de data_preprocessing.py)
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).squeeze("columns")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Definimos dónde guardará Optuna toda su memoria histórica para poder graficar después en Jupyter
    db_path = os.path.join(MODELS_DIR, "optuna_study.db")
    
    # Borra el estudio viejo antes de crear uno nuevo
    try:
        optuna.delete_study(
            study_name="google_ads_optimization",
            storage=f"sqlite:///{db_path}"
        )
    except KeyError:
        pass

    # Fijamos la semilla del sampler de Optuna para garantizar reproducibilidad total.
    # Sin esto, Optuna exploraría combinaciones en orden distinto en cada corrida.
    sampler = optuna.samplers.TPESampler(seed=42)
    
    # Creamos un "Estudio" nuevo indicando que queremos maximizar el puntaje
    study = optuna.create_study(
        direction="maximize", 
        study_name="google_ads_optimization",
        sampler=sampler,
        storage=f"sqlite:///{db_path}",
        load_if_exists=False
    )
    
    # Corremos 30 intentos distintos automatizados
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=30) 

    # Presentación de resultados
    print("\n" + "="*60)
    print(f"RESULTADO GANADOR:")
    print(f"F1-Macro Score (CV): {study.best_value:.4f}")
    print(f"Parámetros: {study.best_params}")
    print("="*60 + "\n")

    # Tomamos exclusivamente el diccionario ganador y lo congelamos en un .pkl
    joblib.dump(study.best_params, BEST_PARAMS_FILE)
    print(f"Mejores parámetros guardados en {BEST_PARAMS_FILE}")
    print(f"Historial del estudio guardado en {db_path}")

    study.best_trial.value  # Score del ganador actual

# Comparar scores entre modelos

if __name__ == "__main__":
    run_hyperparameter_tuning()
