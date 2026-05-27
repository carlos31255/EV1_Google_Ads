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
RAW_CSV       = os.path.join(BASE_DIR, "data", "raw", "GoogleAds_DataAnalytics_Sales_Uncleaned.csv")
METADATA      = os.path.join(BASE_DIR, "data", "raw", "metadata.json")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# Columnas que causan fuga de datos o no tienen valor predictivo.
# NOTA: Cost y Sale_Amount se eliminan DESPUÉS de usarlas para crear
# los ratios de eficiencia y el target. Los ratios SÍ pueden quedarse
# porque no revelan el profit directamente, sino la eficiencia relativa.
COLUMNS_TO_DROP = [
    "Ad_ID", "Ad_Date",
    "Cost", "Sale_Amount",          # valores absolutos — leakage directo
    "Profit_Margin",                 # es el target crudo
    "Conversions", "Conversion Rate" # resultado final — leakage
]

RANDOM_STATE = 42
TEST_SIZE    = 0.20


def load_and_audit(raw_path: str, metadata_path: str) -> pd.DataFrame:
    """
    Verifica la integridad del dataset y carga el CSV crudo con optimizacion de memoria.
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


def engineer_efficiency_features(df: pd.DataFrame,
                                  cost_num: pd.Series,
                                  sale_num: pd.Series) -> pd.DataFrame:
    """
    Crea ratios de eficiencia que capturan rentabilidad relativa sin exponer
    los valores absolutos de Cost ni Sale_Amount (evita leakage directo).

    ¿Por qué estos ratios y no Cost/Sale_Amount directos?
    ------------------------------------------------------
    Cost y Sale_Amount en bruto revelan el profit de forma casi directa
    (profit = sale - cost), por lo que incluirlos como features sería leakage.
    Los ratios, en cambio, miden EFICIENCIA: cuánto costó conseguir cada clic,
    cuánto ingresó por cada lead, etc. Esa información sí es predictiva y no
    trivialmente derivable del target.

    Features creadas
    ----------------
    - cost_per_click      : Costo promedio por cada clic recibido.
    - cost_per_lead       : Costo promedio por cada lead generado.
    - revenue_per_click   : Ingreso promedio por cada clic recibido.
    - revenue_per_lead    : Ingreso promedio por cada lead generado.
    - ctr                 : Click-through rate (Clicks / Impressions).
    - lead_rate           : Tasa de conversión a lead (Leads / Clicks).

    Parametros
    ----------
    df       : DataFrame crudo (debe tener columnas Clicks, Impressions, Leads).
    cost_num : Serie numérica con el costo ya parseado (sin símbolo $).
    sale_num : Serie numérica con el ingreso ya parseado (sin símbolo $).

    Retorna
    -------
    pd.DataFrame con las nuevas columnas agregadas.
    """
    logging.info("STEP 3a — Engineering efficiency ratio features")

    clicks      = pd.to_numeric(df['Clicks'],      errors='coerce').replace(0, float('nan'))
    impressions = pd.to_numeric(df['Impressions'], errors='coerce').replace(0, float('nan'))
    leads       = pd.to_numeric(df['Leads'],       errors='coerce').replace(0, float('nan'))

    # Ratios de costo (cuánto se gastó por unidad de resultado)
    df['cost_per_click']    = cost_num / clicks
    df['cost_per_lead']     = cost_num / leads

    # Ratios de ingreso (cuánto generó cada unidad de resultado)
    df['revenue_per_click'] = sale_num / clicks
    df['revenue_per_lead']  = sale_num / leads

    # Ratios de eficiencia de tráfico (independientes del dinero)
    df['ctr']               = clicks / impressions   # % de impresiones que generaron clic
    df['lead_rate']         = leads  / clicks        # % de clics que generaron lead

    nuevas = ['cost_per_click', 'cost_per_lead',
              'revenue_per_click', 'revenue_per_lead',
              'ctr', 'lead_rate']
    logging.info(f"Features de eficiencia creadas: {nuevas}")

    return df


def create_profitable_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Crea la variable objetivo binaria 'Is_Profitable' y agrega features de
    eficiencia ANTES de eliminar Cost y Sale_Amount.

    Estrategia del target
    ---------------------
    1. Se parsean Cost y Sale_Amount a numérico desde sus versiones string con '$'.
    2. Se calculan los ratios de eficiencia (ver engineer_efficiency_features).
    3. Se calcula Profit_Margin = (Sale_Amount - Cost) / Cost sobre datos CRUDOS.
    4. Solo se etiquetan filas donde Cost Y Sale_Amount son conocidos.
    5. Se prueban taus fijos [0.10, 0.20, 0.30] eligiendo el más balanceado.
    6. Fallback dinámico al percentil 70 si ningún tau fijo sirve.
    7. Se eliminan Cost, Sale_Amount y demás columnas de leakage.
    """
    logging.info("STEP 3 — Creating 'Is_Profitable' target + efficiency features")

    # ── Parseo monetario ──────────────────────────────────────────────────────
    cost_num = pd.to_numeric(
        df['Cost'].astype(str).str.replace(r'[\$,]', '', regex=True).str.strip(),
        errors='coerce'
    )
    sale_num = pd.to_numeric(
        df['Sale_Amount'].astype(str).str.replace(r'[\$,]', '', regex=True).str.strip(),
        errors='coerce'
    )

    # ── Ratios de eficiencia (ANTES de eliminar Cost/Sale_Amount) ─────────────
    # Importante: se crean aquí usando los valores crudos, luego Cost y
    # Sale_Amount se eliminan en el drop final para evitar leakage directo.
    df = engineer_efficiency_features(df, cost_num, sale_num)

    # ── Construcción del target ───────────────────────────────────────────────
    profit_margin = (sale_num - cost_num) / cost_num.replace(0, float('nan'))

    known_mask    = cost_num.notna() & sale_num.notna()
    known_margin  = profit_margin[known_mask]
    unknown_count = (~known_mask).sum()

    tau_candidates = [0.10, 0.20, 0.30]
    best_tau, best_score, best_target, selection_mode = None, None, None, 'fixed'

    for tau in tau_candidates:
        y_tau    = (known_margin >= tau).astype(int)
        pos_rate = y_tau.mean()
        score    = abs(pos_rate - 0.5)
        if 0 < pos_rate < 1 and (best_score is None or score < best_score):
            best_score, best_tau, best_target = score, tau, y_tau

    if best_target is None:
        finite_margin = known_margin.replace([float('inf'), float('-inf')], float('nan')).dropna()
        best_tau      = float(finite_margin.quantile(0.70))
        best_target   = (known_margin >= best_tau).astype(int)
        selection_mode = 'dynamic_fallback'

    logging.info(f"Tau selected: {best_tau:.4f} (mode: {selection_mode})")
    logging.info(f"Supervised rows: {known_mask.sum():,} | Excluded: {unknown_count:,}")
    logging.info(f"Target — Profitable (1): {best_target.mean()*100:.1f}% | Not (0): {(1-best_target.mean())*100:.1f}%")

    # ── Filtrar solo filas con label conocido y asignar target ────────────────
    df = df.loc[known_mask].copy()
    df['Is_Profitable'] = best_target.values

    # ── Eliminar columnas de leakage (Cost y Sale_Amount se van aquí) ─────────
    y = df['Is_Profitable']
    X = df.drop(columns=COLUMNS_TO_DROP + ['Is_Profitable'], errors='ignore')

    logging.info(f"Features finales para el modelo: {X.shape[1]} columnas")
    logging.info(f"Columnas: {X.columns.tolist()}")

    return X, y


def clean_and_transform(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """
    Ajusta el pipeline de sklearn SOLO sobre X_train y lo aplica a ambos sets.
    """
    logging.info("STEP 5 — Building and fitting preprocessing pipeline on Train set")

    pipeline = build_preprocessing_pipeline(columns_to_drop=COLUMNS_TO_DROP)

    X_train_proc = pipeline.fit_transform(X_train)
    X_test_proc  = pipeline.transform(X_test)

    logging.info(f"Train shape after processing: {X_train_proc.shape}")
    logging.info(f"Test shape after processing:  {X_test_proc.shape}")

    return X_train_proc, X_test_proc, pipeline


def save_splits(X_train, X_test, y_train, y_test, pipeline,
                feature_names, output_dir: str) -> None:
    """
    Persiste los splits Train/Test procesados como CSV conservando los nombres
    de columna, y guarda el pipeline ajustado.

    CORRECCIÓN respecto a la versión anterior:
    ------------------------------------------
    La versión anterior hacía pd.DataFrame(X_train) sobre el array numpy que
    devuelve el pipeline, lo que perdía los nombres de columna (quedaban 0,1,2...).
    Ahora se pasan los feature_names obtenidos con pipeline.get_feature_names_out()
    para que los CSV conserven nombres legibles e interpretables.
    """
    logging.info("PASO 6 — Guardando particiones procesadas y pipeline en disco")
    os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame(X_train, columns=feature_names).to_csv(
        os.path.join(output_dir, "X_train.csv"), index=False)
    pd.DataFrame(X_test,  columns=feature_names).to_csv(
        os.path.join(output_dir, "X_test.csv"),  index=False)

    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False, header=True)
    y_test.to_csv(os.path.join(output_dir,  "y_test.csv"),  index=False, header=True)

    pipeline_path = os.path.join(output_dir, "preprocessing_pipeline.joblib")
    joblib.dump(pipeline, pipeline_path)

    logging.info(f"Archivos guardados en: {output_dir}")
    logging.info(f"  X_train.csv — {X_train.shape} | Features: {list(feature_names)}")
    logging.info(f"  X_test.csv  — {X_test.shape}")
    logging.info(f"  Pipeline    — {pipeline_path}")


def run_preprocessing() -> None:
    """
    Punto de entrada principal. Ejecuta el pipeline completo de preprocesamiento.
    """
    logging.info("=" * 55)
    logging.info("Iniciando Preprocesamiento de Datos (Google Ads)")
    logging.info("=" * 55)

    # PASO 1 & 2: Auditoria + Carga
    df = load_and_audit(RAW_CSV, METADATA)

    # PASO 3: Crear ratios de eficiencia + target Is_Profitable
    X, y = create_profitable_target(df)

    # PASO 4: Split 80/20 estratificado ANTES de limpiar (evita leakage)
    logging.info("PASO 4 — Particionamiento 80/20 estratificado (Train/Test)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    logging.info(f"Entrenamiento: {X_train.shape[0]:,} filas | Prueba: {X_test.shape[0]:,} filas")

    # PASO 5: Limpiar y transformar (fit solo en train)
    X_train_proc, X_test_proc, pipeline_ajustado = clean_and_transform(X_train, X_test)

    # ── NUEVA EXTRACCIÓN ROBUSTA DE FEATURE NAMES ─────────────────────────────
    # Extraemos directamente los nombres generados por el ColumnTransformer final
    try:
        # Accedemos al último paso ('preprocessing') del pipeline ajustado
        preprocessor_step = pipeline_ajustado.named_steps['preprocessing']
        feature_names = preprocessor_step.get_feature_names_out()
        logging.info(f"Nombres de características extraídos con éxito del Preprocesador ({len(feature_names)} columnas).")
    except Exception as e:
        # Fallback seguro en caso de cualquier asimetría imprevista en el mapeo de sklearn
        feature_names = [f"feature_{i}" for i in range(X_train_proc.shape[1])]
        logging.warning(f"No se pudo extraer get_feature_names_out de forma nativa ({e}) — usando índices indexados.")

    # PASO 6: Guardar resultados con nombres de columna
    save_splits(X_train_proc, X_test_proc, y_train, y_test,
                pipeline_ajustado, feature_names, PROCESSED_DIR)

    logging.info("=" * 55)
    logging.info("Preprocesamiento finalizado. Los datos estan listos para modelado.")
    logging.info("=" * 55)


if __name__ == "__main__":
    run_preprocessing()