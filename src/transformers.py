# Módulo de Transformaciones de Datos.

#
# Objetivos implementados aquí:
#   A. Estandarización de Fechas          — columna Ad_Date     
#   B. Limpieza de Texto / Fuzzy Matching — columnas Campaign_Name, Location
#   C. Transformación de Tipos Numéricos  — limpieza de columnas monetarias (Cost, Sale_Amount)
#   D. Imputación Inteligente de Valores Faltantes — lógica de negocio + pipeline de mediana
#
# Objetivos del compañero: (ordenar luego de finalizar el proyecto)
#   A. Estandarización de Fechas          — columna Ad_Date     
#   B. Limpieza de Texto / Fuzzy Matching — columnas Campaign_Name, Location

import logging
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# C. TRANSFORMACIÓN DE TIPOS NUMÉRICOS

# Esta funcion elimina el símbolo '$' y las comas de una columna de texto monetario y la convierte a tipo float.
def clean_monetary_column(series: pd.Series) -> pd.Series:
    # Ejemplo: '$1,234.56' -> 1234.56
    return (
        series
        .astype(str)
        .str.replace(r'[$,]', '', regex=True)
        .str.strip()
        .replace('', np.nan)        # strings vacíos -> NaN en lugar de error
        .astype(float)
    )

# Esta funcion aplica clean_monetary_column a una o más columnas del DataFrame.
def transform_monetary_columns(df: pd.DataFrame,
                                columns: list = None) -> pd.DataFrame:
    # Aplica clean_monetary_column a una o más columnas del DataFrame.
    
    if columns is None:
        columns = ['Cost', 'Sale_Amount']
    # Copia el DataFrame para no modificar el original
    df = df.copy()
    for col in columns:
        if col in df.columns:
            before_nulls = df[col].isna().sum()
            df[col] = clean_monetary_column(df[col])
            after_nulls = df[col].isna().sum()
            new_nulls = after_nulls - before_nulls
            logging.info(
                f"[C] '{col}' convertida a float. "
                f"Nuevos NaN introducidos (valores vacíos/inválidos): {new_nulls}"
            )
        else:
            logging.warning(f"[C] Columna '{col}' no encontrada en el DataFrame. Se omite.")

    return df


# D. IMPUTACIÓN INTELIGENTE DE VALORES FALTANTES

# Esta funcion recalcula los valores faltantes de Conversion_Rate usando lógica de negocio:
# Conversion_Rate = Conversions / Clicks
def impute_conversion_rate(df: pd.DataFrame,
                            rate_col: str = 'Conversion_Rate',
                            conversions_col: str = 'Conversions',
                            clicks_col: str = 'Clicks') -> pd.DataFrame:

    df = df.copy()
    # Verificamos que la columna Conversion_Rate exista en el DataFrame
    if rate_col not in df.columns:
        logging.warning(f"[D] Columna '{rate_col}' no encontrada. Se omite la imputación por negocio.")
        return df

    # Identificamos las filas donde SÍ podemos calcular la tasa
    missing_mask = df[rate_col].isna()
    can_calculate = (
        missing_mask
        & df[conversions_col].notna()
        & df[clicks_col].notna()
        & (df[clicks_col] > 0)
    )
    # Contamos cuántos valores se pueden recalcular
    recalculated = can_calculate.sum()
    # Recalculamos los valores faltantes
    df.loc[can_calculate, rate_col] = (
        df.loc[can_calculate, conversions_col] / df.loc[can_calculate, clicks_col]
    )
    # Contamos cuántos valores quedan faltantes
    still_missing = df[rate_col].isna().sum()
    # Mostramos los resultados
    logging.info(
        f"[D] '{rate_col}': {recalculated} valores recalculados por lógica de negocio. "
        f"NaN restantes: {still_missing}"
    )
    return df
    


def build_median_imputation_pipeline(numeric_columns: list) -> Pipeline:
    # Construye un Pipeline de scikit-learn con SimpleImputer de mediana
    # para las columnas numéricas especificadas.
    #
    # Uso
    # ---
    # pipeline = build_median_imputation_pipeline(['Cost', 'CTR', 'Conversion_Rate'])
    # imputed_array = pipeline.fit_transform(df[numeric_columns])
    #
    # Retorna
    # -------
    # Pipeline de sklearn listo para fit/transform.

    pipeline = Pipeline(steps=[
        ('imputer_mediana', SimpleImputer(strategy='median'))
    ])
    logging.info(
        f"[D] Pipeline de imputación por mediana creado para columnas: {numeric_columns}"
    )
    return pipeline


def impute_remaining_numeric(df: pd.DataFrame,
                              columns: list = None) -> pd.DataFrame:
    # Aplica imputación por mediana (vía Pipeline de sklearn) a los NaN
    # restantes en las columnas numéricas especificadas.
    #
    # Parámetros
    # ----------
    # df      : DataFrame de entrada (ya con imputación por lógica de negocio aplicada).
    # columns : Columnas numéricas a imputar. Por defecto: todas las columnas numéricas.
    #
    # Retorna
    # -------
    # DataFrame con valores imputados por mediana (copia).

    df = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Solo procesamos columnas que realmente tienen valores faltantes
    cols_with_nulls = [c for c in columns if c in df.columns and df[c].isna().any()]

    if not cols_with_nulls:
        logging.info("[D] No se encontraron NaN restantes en columnas numéricas. Nada que imputar.")
        return df

    pipeline = build_median_imputation_pipeline(cols_with_nulls)
    imputed_values = pipeline.fit_transform(df[cols_with_nulls])
    df[cols_with_nulls] = imputed_values

    logging.info(
        f"[D] Imputación por mediana aplicada a {len(cols_with_nulls)} columna(s): "
        f"{cols_with_nulls}"
    )
    return df


def apply_full_imputation(df: pd.DataFrame,
                           rate_col: str = 'Conversion_Rate',
                           conversions_col: str = 'Conversions',
                           clicks_col: str = 'Clicks',
                           numeric_columns: list = None) -> pd.DataFrame:
    # Pipeline completo de imputación (Pasos D1 + D2):
    #   1. Recálculo por lógica de negocio para Conversion_Rate.
    #   2. Imputación por mediana para los NaN numéricos restantes.
    #
    # Parámetros
    # ----------
    # df              : DataFrame crudo o parcialmente limpiado.
    # rate_col        : Nombre de la columna Conversion Rate.
    # conversions_col : Nombre de la columna Conversions.
    # clicks_col      : Nombre de la columna Clicks.
    # numeric_columns : Columnas sobre las que aplicar imputación por mediana.
    #                   Si es None, se usan todas las columnas numéricas.
    #
    # Retorna
    # -------
    # DataFrame completamente imputado (copia).

    logging.info("[D] Iniciando pipeline completo de imputación...")
    df = impute_conversion_rate(df, rate_col, conversions_col, clicks_col)
    df = impute_remaining_numeric(df, numeric_columns)
    logging.info("[D] Pipeline completo de imputación finalizado.")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# PRUEBA RÁPIDA
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Muestra sintética mínima para verificar ambos transformadores localmente
    sample = pd.DataFrame({
        'Cost':            ['$1,200.50', '$800.00', None,       '$450.75'],
        'Sale_Amount':     ['$3,000.00', None,       '$1,500.00', '$2,200.00'],
        'Clicks':          [200,          150,        300,         None],
        'Conversions':     [10,           None,       30,          20],
        'Conversion_Rate': [None,         0.05,       None,        None],
        'CTR':             [0.04,         None,       0.06,        0.03],
    })

    print("=== Antes ===")
    print(sample, "\n")

    # C — Columnas monetarias
    sample = transform_monetary_columns(sample, columns=['Cost', 'Sale_Amount'])

    # D — Imputación
    sample = apply_full_imputation(sample)

    print("=== Después ===")
    print(sample)
