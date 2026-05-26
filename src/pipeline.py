"""
Pipeline module for Google Ads Dataset.
Orchestrates structural cleaning, missing value handling, and statistical scaling.
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold

from .transformers import (
    DateStandardizerTransformer, TextNormalizerTransformer,
    DropColumnsTransformer, MonetaryCleanerTransformer,
    DropHighMissingTransformer, SmartImputerTransformer
)

def build_preprocessing_pipeline(columns_to_drop=None):
    """
    Builds the complete scikit-learn preprocessing pipeline.
    Dynamically identifies numeric and categorical features.
    """
    if columns_to_drop is None:
        columns_to_drop = ['Ad_ID', 'Ad_Date', 'Cost', 'Sale_Amount']

    # Ruta de procesamiento numérico
    num_pipe = Pipeline([
        ('zero_variance', VarianceThreshold(threshold=0.0)),
        ('scaler', StandardScaler())
    ])

    # Ruta de procesamiento categórico
    cat_pipe = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Enrutador dinámico (Detecta columnas sobrevivientes a la limpieza previa)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipe, make_column_selector(dtype_include='number')),
            ('cat', cat_pipe, make_column_selector(dtype_exclude='number')),
        ],
        remainder='drop'
    )

    # Pipeline Maestro
    full_pipeline = Pipeline([
        ('date_standardizer', DateStandardizerTransformer()),
        ('text_normalizer',   TextNormalizerTransformer()),
        ('monetary_clean',    MonetaryCleanerTransformer(columns=['Cost', 'Sale_Amount'])),
        ('drop_leaks',        DropColumnsTransformer(columns_to_drop=columns_to_drop)),
        ('drop_high_nan',     DropHighMissingTransformer(threshold=0.80)),
        ('smart_imputer',     SmartImputerTransformer()),
        ('preprocessing',     preprocessor),
    ])

    return full_pipeline
