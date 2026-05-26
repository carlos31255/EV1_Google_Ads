"""
data_preprocessing.py
----------------------
Orquesta el flujo completo de preprocesamiento para el dataset de Google Ads.

Pasos ejecutados en orden:
    1. Auditoria  - Verifica la integridad del archivo via checksum SHA-256 (audit.py)
    2. Carga      - Lee el CSV crudo aplicando optimizacion de memoria (optimization.py)
    3. Limpieza   - Aplica todas las transformaciones via pipeline de sklearn (pipeline.py)
    4. Division   - Separa features/target y realiza el split 80/20 Train/Test
    5. Guardado   - Persiste los splits procesados en data/processed/ para uso posterior

Uso:
    python src/data_preprocessing.py
"""

import os
import sys
import logging
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# Configuracion del sistema de logs
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)

# Permite importar modulos desde la raiz
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.audit import verify_data_integrity
from src.optimization import read_csv_in_chunks
from src.pipeline import build_preprocessing_pipeline

# ──────────────────────────────────────────────────────
# CONSTANTES Y RUTAS DEL PROYECTO
# ──────────────────────────────────────────────────────
RAW_CSV      = os.path.join(BASE_DIR, "data", "raw", "GoogleAds_DataAnalytics_Sales_Uncleaned.csv")
METADATA     = os.path.join(BASE_DIR, "data", "raw", "metadata.json")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# Variable objetivo: Conversions es la variable a predecir
# Se binariza: 1 si el anuncio generó al menos 1 conversión, 0 en caso contrario
TARGET_COLUMN  = "Conversions"

# Columnas que causan fuga de datos o no tienen valor predictivo
# Ad_ID: identificador unico (no es una caracteristica)
# Ad_Date, Cost, Sale_Amount: se usan para crear el target ANTES de ser eliminadas
# Profit_Margin: derivada directa del target, causaria leakage total
COLUMNS_TO_DROP = ["Ad_ID", "Ad_Date", "Cost", "Sale_Amount", "Profit_Margin",
                   "Conversions", "Conversion Rate"]

# Semilla fija para reproducibilidad total
RANDOM_STATE = 42
TEST_SIZE    = 0.20


def load_and_audit(raw_path: str, metadata_path: str) -> pd.DataFrame:
    """
    Verifica la integridad del dataset y carga el CSV crudo con optimizacion de memoria.

    Lanza SystemExit si la verificacion falla, evitando que se entrene un modelo
    sobre datos corruptos o manipulados.

    Parametros
    ----------
    raw_path : str
        Ruta absoluta al archivo CSV crudo.
    metadata_path : str
        Ruta absoluta al metadata.json que contiene el hash SHA-256 de referencia.

    Retorna
    -------
    pd.DataFrame
        DataFrame crudo cargado y optimizado en memoria.
    """
    logging.info("=" * 55)
    logging.info("STEP 1 — Data Integrity Audit")
    logging.info("=" * 55)

    integrity_ok = verify_data_integrity(raw_path, metadata_path)
    if not integrity_ok:
        logging.critical("Aborting: dataset failed integrity check.")
        sys.exit(1)

    logging.info("STEP 2 — Loading raw CSV with memory optimization")
    df = read_csv_in_chunks(raw_path)

    if df is None or df.empty:
        logging.critical("Aborting: dataset could not be loaded.")
        sys.exit(1)

    logging.info(f"Raw dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def create_profitable_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Crea la variable objetivo binaria 'Is_Profitable' usando una estrategia
    hibrida de tau/margen para garantizar una etiqueta balanceada y entrenable.

    Estrategia
    ----------
    1. Se calcula Profit_Margin = (Sale_Amount - Cost) / Cost sobre las columnas
       monetarias CRUDAS, antes de cualquier imputacion o winsorization (evita leakage).
    2. Solo se etiquetan las filas donde Cost Y Sale_Amount son conocidos
       (las filas con valores desconocidos se excluyen del set supervisado).
    3. Se prueban taus fijos [0.10, 0.20, 0.30] y se selecciona el que produce
       una tasa positiva mas cercana al 0.50 (split mas balanceado).
    4. Fallback dinamico: si todos los taus fijos generan un split degenerado
       (0% o 100% positivo), se usa tau = percentil 70 de los margenes conocidos.

    ¿Por qué no Conversions > 0?
    ---------------------------
    Con los datos crudos, Conversions > 0 etiqueta el 97% de las filas como clase 1,
    dejando al modelo sin nada que aprender. El target basado en margen representa
    la pregunta real del negocio: "Fue rentable esta campaña?", y produce una
    distribucion de clases estable y entrenable.

    Parametros
    ----------
    df : pd.DataFrame
        DataFrame crudo (debe contener 'Cost' y 'Sale_Amount' como strings con '$').

    Retorna
    -------
    tuple[pd.DataFrame, pd.Series]
        (X, y) donde X tiene eliminadas las columnas relacionadas al target.
    """
    logging.info("STEP 3 \u2014 Creating 'Is_Profitable' target (hybrid tau/margin strategy)")

    # Parseamos Cost y Sale_Amount a numerico desde sus versiones string con '$'
    cost_num = pd.to_numeric(
        df['Cost'].astype(str).str.replace(r'[\$,]', '', regex=True).str.strip(),
        errors='coerce'
    )
    sale_num = pd.to_numeric(
        df['Sale_Amount'].astype(str).str.replace(r'[\$,]', '', regex=True).str.strip(),
        errors='coerce'
    )

    # Calculamos el margen sobre los datos crudos ANTES de imputar
    profit_margin = (sale_num - cost_num) / cost_num.replace(0, float('nan'))

    # Mascara: solo filas con ambos valores monetarios conocidos
    known_mask = cost_num.notna() & sale_num.notna()
    known_margin = profit_margin[known_mask]
    unknown_count = (~known_mask).sum()

    # Estrategia 1: buscar el tau fijo mas balanceado
    tau_candidates = [0.10, 0.20, 0.30]
    best_tau, best_score, best_target, selection_mode = None, None, None, 'fixed'

    for tau in tau_candidates:
        y_tau = (known_margin >= tau).astype(int)
        pos_rate = y_tau.mean()
        score = abs(pos_rate - 0.5)      # cuanto se aleja de 50/50
        if 0 < pos_rate < 1 and (best_score is None or score < best_score):
            best_score, best_tau, best_target = score, tau, y_tau

    # Estrategia 2: fallback dinamico al percentil 70 si ningun tau fijo sirve
    if best_target is None:
        finite_margin = known_margin.replace([float('inf'), float('-inf')], float('nan')).dropna()
        best_tau = float(finite_margin.quantile(0.70))
        best_target = (known_margin >= best_tau).astype(int)
        selection_mode = 'dynamic_fallback'

    logging.info(f"Tau selected: {best_tau:.4f} (mode: {selection_mode})")
    logging.info(f"Supervised rows: {known_mask.sum():,} | Excluded (unknown monetary): {unknown_count:,}")
    logging.info(f"Target distribution \u2014 Profitable (1): {best_target.mean()*100:.1f}% | Not (0): {(1-best_target.mean())*100:.1f}%")

    # Aplicamos el target y filtramos filas sin label
    df = df.loc[known_mask].copy()
    df['Is_Profitable'] = best_target.values

    # Separamos y eliminamos columnas que causarian fuga de datos
    y = df['Is_Profitable']
    X = df.drop(columns=COLUMNS_TO_DROP + ['Is_Profitable'], errors='ignore')

    return X, y


def clean_and_transform(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """
    Ajusta el pipeline completo de sklearn SOLO sobre X_train y luego lo aplica
    a ambos conjuntos para prevenir la fuga de datos (data leakage).

    Parametros
    ----------
    X_train : pd.DataFrame
        Conjunto de entrenamiento con features crudos.
    X_test : pd.DataFrame
        Conjunto de prueba con features crudos.

    Retorna
    -------
    tuple
        (X_train_procesado, X_test_procesado, pipeline_ajustado)
    """
    logging.info("STEP 5 — Building and fitting preprocessing pipeline on Train set")

    pipeline = build_preprocessing_pipeline(columns_to_drop=COLUMNS_TO_DROP)

    # fit() solo sobre entrenamiento para prevenir fuga de datos al test set
    X_train_proc = pipeline.fit_transform(X_train)

    # transform() sobre test usando las estadisticas aprendidas del train
    X_test_proc  = pipeline.transform(X_test)

    logging.info(f"Pipeline fitted. Train shape after processing: {X_train_proc.shape}")
    logging.info(f"Test shape after processing:  {X_test_proc.shape}")

    return X_train_proc, X_test_proc, pipeline


def save_splits(X_train, X_test, y_train, y_test, pipeline, output_dir: str) -> None:
    """
    Persiste los splits Train/Test procesados como archivos CSV y guarda el pipeline.

    Parametros
    ----------
    X_train, X_test : array-like
        Matrices de features procesadas.
    y_train, y_test : pd.Series
        Vectores de la variable objetivo.
    pipeline : sklearn.pipeline.Pipeline
        El pipeline ajustado.
    output_dir : str
        Directorio de destino para los archivos de salida.
    """
    logging.info("PASO 6 \u2014 Guardando particiones procesadas y pipeline en disco")
    os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame(X_train).to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    pd.DataFrame(X_test).to_csv(os.path.join(output_dir,  "X_test.csv"),  index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False, header=True)
    y_test.to_csv(os.path.join(output_dir,  "y_test.csv"),  index=False, header=True)

    # Guardar el pipeline ajustado para reproducibilidad
    pipeline_path = os.path.join(output_dir, "preprocessing_pipeline.joblib")
    joblib.dump(pipeline, pipeline_path)

    logging.info(f"Archivos guardados en: {output_dir}")
    logging.info(f"  X_train.csv \u2014 {X_train.shape}")
    logging.info(f"  X_test.csv  \u2014 {X_test.shape}")
    logging.info(f"  Pipeline    \u2014 {pipeline_path}")


def run_preprocessing() -> None:
    """
    Punto de entrada principal. Ejecuta el pipeline completo de preprocesamiento.
    Orquestador principal del preprocesamiento.
    """
    logging.info("=" * 55)
    logging.info("Iniciando Preprocesamiento de Datos (Google Ads)")
    logging.info("=" * 55)

    # PASO 1 & 2: Auditoria + Carga
    df = load_and_audit(RAW_CSV, METADATA)

    # PASO 3: Crear target Is_Profitable y separar features
    X, y = create_profitable_target(df)

    # PASO 4: Particionamiento 80/20 estratificado ANTES de limpiar (evita leakage)
    logging.info("PASO 4 \u2014 Particionamiento 80/20 estratificado (Train/Test)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    logging.info(f"Set de Entrenamiento: {X_train.shape[0]:,} filas | Set de Prueba: {X_test.shape[0]:,} filas")

    # PASO 5: Limpiar y transformar (fit solo en train)
    X_train_proc, X_test_proc, pipeline_ajustado = clean_and_transform(X_train, X_test)

    # PASO 6: Guardar resultados y el pipeline
    save_splits(X_train_proc, X_test_proc, y_train, y_test, pipeline_ajustado, PROCESSED_DIR)

    logging.info("=" * 55)
    logging.info("Preprocesamiento finalizado. Los datos estan listos para modelado.")
    logging.info("=" * 55)


if __name__ == "__main__":
    run_preprocessing()
