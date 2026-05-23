import os
import joblib
import logging
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

# 1. Configuración básica
# logging nos permite mostrar mensajes en consola más ordenados que los clásicos 'print'
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# Definimos las rutas relativas. Esto asegura que el script funcione igual sin importar desde qué carpeta se llame.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
BEST_PARAMS_FILE = os.path.join(MODELS_DIR, "best_params.pkl")
FINAL_MODEL_FILE = os.path.join(MODELS_DIR, "final_classifier.joblib")

def reconstruct_model(params: dict) -> Pipeline:
    """
    Función clave: Toma el diccionario de los mejores hiperparámetros encontrados por Optuna
    y construye un Pipeline de scikit-learn totalmente instanciado y listo para entrenar.
    """
    steps = [] # Aquí iremos ensamblando las "piezas" de nuestro modelo predictivo
    
    # ---------------------------------------------------------
    # FASE 1: Reducción de Dimensionalidad (PCA)
    # ---------------------------------------------------------
    # Optuna nos dejó una bandera indicando si vale la pena usar PCA.
    # El .get() nos protege de errores si la llave 'use_pca' no existe.
    if params.get("use_pca", False):
        pca_var = params.get("pca_variance")
        # Si aplica PCA, agregamos este transformador matemático a nuestra lista de pasos
        steps.append(('pca', PCA(n_components=pca_var, random_state=42)))
        
    # ---------------------------------------------------------
    # FASE 2: Elección y Configuración del Algoritmo Clasificador
    # ---------------------------------------------------------
    clf_name = params.get("classifier")
    
    # Basado en el nombre que eligió Optuna, inicializamos la clase correspondiente.
    # class_weight='balanced' le indica a los modelos que presten más atención a la clase minoritaria (30%), compensando el desbalance de los datos.
    if clf_name == "LogisticRegression":
        model = LogisticRegression(C=params["lr_C"], max_iter=1000, class_weight='balanced', random_state=42)
    elif clf_name == "RandomForest":
        # n_jobs=-1 paraleliza el entrenamiento ocupando todos los núcleos de tu procesador
        model = RandomForestClassifier(n_estimators=params["rf_n_estimators"], max_depth=params["rf_max_depth"], class_weight='balanced', random_state=42, n_jobs=-1)
    elif clf_name == "SVM":
        model = SVC(C=params["svm_C"], kernel=params["svm_kernel"], gamma="scale", class_weight='balanced', random_state=42)
    elif clf_name == "XGBoost":
        # XGBoost maneja el balanceo por su cuenta al optimizar logloss
        model = XGBClassifier(n_estimators=params["xgb_n_estimators"], learning_rate=params["xgb_learning_rate"], random_state=42, eval_metric='mlogloss', n_jobs=-1)
    elif clf_name == "LightGBM":
        model = LGBMClassifier(n_estimators=params["lgb_n_estimators"], learning_rate=params["lgb_learning_rate"], class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
    else:
        raise ValueError(f"Clasificador desconocido: {clf_name}")
        
    # Finalmente, enganchamos el algoritmo predictor al final de la línea de ensamblaje
    steps.append(('classifier', model))
    
    # Retornamos el Pipeline ya armado. (Aún está "en blanco", falta hacerle .fit())
    return Pipeline(steps)

def run_model_training():
    """
    Orquestador principal que consolida el flujo de la Persona A: 
    Lee los mejores hiperparámetros, entrena con los datos procesados, y guarda el modelo final.
    """
    logging.info("Iniciando Entrenamiento del Modelo Final (Persona A)...")
    
    # 1. Validación de seguridad: Asegurarnos de que Optuna ya hizo su trabajo
    if not os.path.exists(BEST_PARAMS_FILE):
        logging.error("No se encontró best_params.pkl. Debes correr hyperparameter_tuning.py primero.")
        return
        
    # 2. Cargar el diccionario ganador de memoria secundaria
    best_params = joblib.load(BEST_PARAMS_FILE)
    logging.info(f"Parámetros ganadores cargados: {best_params}")
    
    # 3. Ingestar el dataset de entrenamiento limpio proveniente de la fase de Preprocesamiento
    # Al target (y_train) le hacemos squeeze() para pasarlo de matriz 2D (DataFrame) a matriz 1D (Series)
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).squeeze("columns")
    
    # 4. Construcción y Aprendizaje
    logging.info("Reconstruyendo y entrenando el modelo...")
    final_model = reconstruct_model(best_params)
    
    # El pipeline le pasa los datos al PCA (si existe) y luego entrena al modelo final
    final_model.fit(X_train, y_train)
    
    # 5. Guardado y Entrega a la Persona B
    # Guardamos el pipeline matemático como un binario estático para que luego se pueda cargar y predecir sin volver a entrenar.
    joblib.dump(final_model, FINAL_MODEL_FILE)
    logging.info(f"Modelo final entrenado y guardado en: {FINAL_MODEL_FILE}")
    logging.info("¡Trabajo de la Persona A terminado! Listo para que la Persona B evalúe el modelo.")

if __name__ == "__main__":
    run_model_training()
