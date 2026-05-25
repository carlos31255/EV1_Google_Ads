
import os
import joblib
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Configuración básica
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
TRAINED_MODELS_DIR = os.path.join(MODELS_DIR, "trained_models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

FINAL_MODEL_FILE = os.path.join(TRAINED_MODELS_DIR, "final_classifier.joblib")

def run_model_evaluation():
    """
    Evalúa el modelo entrenado usando el conjunto de prueba.
    Genera un reporte de clasificación, matriz de confusión y curva ROC,
    guardando los resultados en las carpetas correspondientes.

    Returns
    -------
    None
    
    Raises
    ------
    FileNotFoundError
        Si no se encuentra el modelo final o los datos de prueba.
    Exception
        Para cualquier otro error durante la evaluación o guardado de gráficos.
    """
    logging.info("Iniciando Evaluación del Modelo...")

    try:
        if not os.path.exists(FINAL_MODEL_FILE):
            raise FileNotFoundError(f"No se encontró el modelo final en {FINAL_MODEL_FILE}")

        # 2. Cargar datos de prueba
        logging.info("Cargando Test Set...")
        X_test_path = os.path.join(PROCESSED_DIR, "X_test.csv")
        y_test_path = os.path.join(PROCESSED_DIR, "y_test.csv")
        
        if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
            raise FileNotFoundError("No se encontraron los datos de prueba X_test.csv o y_test.csv.")
            
        X_test = pd.read_csv(X_test_path)
        y_test = pd.read_csv(y_test_path).squeeze("columns")

        # 3. Cargar el modelo matemático
        logging.info("Cargando el pipeline predictivo...")
        pipeline = joblib.load(FINAL_MODEL_FILE)

        # 4. Generar Predicciones
        logging.info("Realizando predicciones...")
        y_pred = pipeline.predict(X_test)
        # SVM no expone predict_proba por defecto; si falla usamos decision_function como alternativa.
        # decision_function retorna la distancia al hiperplano separador, suficiente para ordenar los casos en la curva ROC.
        try:
            y_proba = pipeline.predict_proba(X_test)[:, 1]
        except AttributeError:
            y_proba = pipeline.decision_function(X_test)

        # 5. Imprimir Métricas en Consola
        class_names = ["No Rentable", "Rentable"]
        logging.info("\n" + "="*50)
        logging.info("REPORTE DE CLASIFICACIÓN FINAL")
        logging.info("="*50)
        print(classification_report(y_test, y_pred, target_names=class_names))

        # Guardar el reporte como tabla CSV en results/metrics/
        report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        os.makedirs(METRICS_DIR, exist_ok=True)
        report_df.to_csv(os.path.join(METRICS_DIR, "classification_report.csv"))
        logging.info(f"Reporte guardado como tabla en: {METRICS_DIR}/classification_report.csv")
        
        roc_auc = roc_auc_score(y_test, y_proba)
        logging.info(f"Puntuación ROC-AUC: {roc_auc:.4f}")

        # 6. Generar y Guardar Gráficos
        os.makedirs(PLOTS_DIR, exist_ok=True)

        # --- Gráfico 1: Matriz de Confusión ---
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            confusion_matrix(y_test, y_pred),
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title('Matriz de Confusión Final')
        plt.ylabel('Realidad')
        plt.xlabel('Predicción del Modelo')
        plt.tight_layout()
        
        cm_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
        plt.savefig(cm_path)
        logging.info(f"Matriz de confusión guardada en: {cm_path}")

        # --- Gráfico 2: Curva ROC ---
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"Curva ROC (AUC = {roc_auc:.4f})", color='darkorange')
        plt.plot([0, 1], [0, 1], 'k--', label="Adivinación Aleatoria")
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title('Capacidad de Separación de Clases (ROC)')
        plt.legend(loc="lower right")
        
        roc_path = os.path.join(PLOTS_DIR, "roc_curve.png")
        plt.savefig(roc_path)
        logging.info(f"Curva ROC guardada en: {roc_path}")

        logging.info("¡Trabajo terminado exitosamente!")

    except Exception as e:
        logging.error(f"Error durante la evaluación del modelo: {e}")
        raise

if __name__ == "__main__":
    run_model_evaluation()
