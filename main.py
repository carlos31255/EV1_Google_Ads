"""
Main orchestration script — Google Ads Analytics.
Ejecuta el pipeline completo en secuencia.

Nota: también puede ejecutarse paso a paso:
    python src/data_preprocessing.py
    python src/hyperparameter_tuning.py
    python src/model_training.py
    python src/model_evaluation.py
"""

import logging
from src.data_preprocessing import run_preprocessing
from src.hyperparameter_tuning import run_hyperparameter_tuning
from src.model_training import run_model_training
from src.model_evaluation import run_model_evaluation

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

def main():
    """
    Función orquestadora que ejecuta los 4 pasos modulares del flujo de datos
    de Machine Learning (ETL, Tuning, Training, Evaluation).
    """
    logging.info("="*60)
    logging.info("🚀 INICIANDO PIPELINE DE MACHINE LEARNING: GOOGLE ADS 🚀")
    logging.info("="*60)

    # ---------------------------------------------------------
    # PASO 1: PREPROCESAMIENTO
    # Crea el Target (ROI), limpia los datos, divide en Train/Test
    # y exporta las matrices procesadas a data/processed/
    # ---------------------------------------------------------
    logging.info("\n[PASO 1/4] — Limpieza y Preprocesamiento de Datos")
    run_preprocessing()

    # ---------------------------------------------------------
    # PASO 2: OPTIMIZACIÓN (OPTUNA)
    # Ejecuta el motor de búsqueda Bayesiana para encontrar los
    # mejores hiperparámetros y guarda el resultado en best_params.pkl
    # ---------------------------------------------------------
    logging.info("\n[PASO 2/4] — Optimización de Hiperparámetros (Optuna)")
    run_hyperparameter_tuning()

    # ---------------------------------------------------------
    # PASO 3: ENTRENAMIENTO FINAL
    # Carga best_params.pkl, reconstruye el modelo ganador
    # y entrena con el 100% de los datos de Train. 
    # Guarda el modelo en final_classifier.joblib
    # ---------------------------------------------------------
    logging.info("\n[PASO 3/4] — Entrenamiento del Modelo Predictivo Final")
    run_model_training()

    # ---------------------------------------------------------
    # PASO 4: EVALUACIÓN
    # Ingiere el X_test, genera predicciones y produce las métricas
    # finales (Matriz de Confusión, Curva ROC, y Classification Report).
    # ---------------------------------------------------------
    logging.info("\n[PASO 4/4] — Evaluación de Desempeño en Test Set")
    run_model_evaluation()

    logging.info("\n" + "="*60)
    logging.info("✅ PIPELINE COMPLETO FINALIZADO CON ÉXITO ✅")
    logging.info("="*60)

if __name__ == "__main__":
    main()