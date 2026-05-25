
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
REPORTS_DIR = os.path.join(BASE_DIR, "reports", "figures")

FINAL_MODEL_FILE = os.path.join(MODELS_DIR, "final_classifier.joblib")

def run_model_evaluation():
    logging.info("Iniciando Evaluación del Modelo...")

    if not os.path.exists(FINAL_MODEL_FILE):
        logging.error("No se encontró el modelo final.")
        return

    # 2. Cargar datos de prueba
    logging.info("Cargando Test Set...")
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).squeeze("columns")

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

    # Guardar el reporte como tabla CSV en reports/
    report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    os.makedirs(os.path.join(BASE_DIR, "reports"), exist_ok=True)
    report_df.to_csv(os.path.join(BASE_DIR, "reports", "classification_report.csv"))
    logging.info(f"Reporte guardado como tabla en: reports/classification_report.csv")
    
    roc_auc = roc_auc_score(y_test, y_proba)
    logging.info(f"Puntuación ROC-AUC: {roc_auc:.4f}")

    # 6. Generar y Guardar Gráficos
    os.makedirs(REPORTS_DIR, exist_ok=True)

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
    
    cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
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
    
    roc_path = os.path.join(REPORTS_DIR, "roc_curve.png")
    plt.savefig(roc_path)
    logging.info(f"Curva ROC guardada en: {roc_path}")

    logging.info("¡Trabajo terminado exitosamente!")

if __name__ == "__main__":
    run_model_evaluation()
