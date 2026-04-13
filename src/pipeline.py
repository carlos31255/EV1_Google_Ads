# Módulo de construcción del Pipeline de Preprocesamiento.

# Objetivos implementados aquí:
#   A. build_preprocessing_pipeline() — construye y retorna el pipeline completo
#                                       listo para fit_transform() o para integrarse
#                                       con un modelo en un pipeline final de ML.

# Las columnas numéricas y categóricas se detectan dinámicamente con
# make_column_selector, por lo que el pipeline se adapta si el dataset cambia.


import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.transformers import (
    DropColumnsTransformer,
    MonetaryCleanerTransformer,
    DropHighMissingTransformer,
    SmartImputerTransformer,
    OutlierCapper,
    DropZeroVarianceTransformer,
)


# A. CONSTRUCCIÓN DEL PIPELINE DE PREPROCESAMIENTO

# Esta función construye y retorna el pipeline completo de preprocesamiento
# adaptado al dataset de Google Ads. Detecta automáticamente qué columnas son
# numéricas y cuáles son categóricas después de los pasos de limpieza previos.
def build_preprocessing_pipeline(columns_to_drop=None):

    # Si no se especifican columnas a eliminar, usamos las del dataset de Google Ads
    # que causan Data Leakage directo (fuente del target Is_Profitable)
    if columns_to_drop is None:
        columns_to_drop = ['Ad_ID', 'Ad_Date', 'Cost', 'Sale_Amount']

    # 1. Ruta para números: Capping -> Varianza Cero -> Escalar
    num_pipe = Pipeline([
        ('capper',        OutlierCapper(apply_capping=True)),
        ('zero_variance', DropZeroVarianceTransformer()),
        ('scaler',        StandardScaler()),
    ])

    # 2. Ruta para textos: OneHot encoding
    cat_pipe = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    # 3. El enrutador maestro: detecta columnas dinámicamente tras la limpieza previa
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipe, make_column_selector(dtype_include='number')),
            ('cat', cat_pipe, make_column_selector(dtype_exclude='number')),
        ],
        remainder='drop'
    )

    # 4. El Súper Pipeline completo
    full_pipeline = Pipeline([
        # Eliminar columnas que generan Data Leakage
        ('drop_leaks',       DropColumnsTransformer(columns_to_drop=columns_to_drop)),
        # Convertir columnas monetarias '$X,XXX.XX' a float
        ('monetary_clean',   MonetaryCleanerTransformer(columns=['Cost', 'Sale_Amount'])),
        # Eliminar columnas con > 80% de nulos
        ('drop_high_nan',    DropHighMissingTransformer(threshold=0.80)),
        # Imputación inteligente según porcentaje de nulos
        ('smart_imputer',    SmartImputerTransformer(low_threshold=0.10)),
        # Preprocesamiento numérico y categórico
        ('preprocessing',    preprocessor),
    ])

    return full_pipeline
