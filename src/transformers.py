"""
Custom Scikit-Learn transformers for the Google Ads dataset.
Includes text normalization, monetary cleaning, and leakage-free imputation.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class DateStandardizerTransformer(BaseEstimator, TransformerMixin):
    """Standardizes date formats and extracts temporal features (month, DOW)."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        if 'Ad_Date' in X_copy.columns:
            # Parseo robusto de fechas y generación de variables temporales
            dates = pd.to_datetime(X_copy['Ad_Date'], errors='coerce', dayfirst=True)
            X_copy['Ad_Month'] = dates.dt.month
            X_copy['Ad_DOW'] = dates.dt.dayofweek
            X_copy['Is_Weekend'] = dates.dt.dayofweek.isin([5, 6]).astype(int)
        return X_copy

class TextNormalizerTransformer(BaseEstimator, TransformerMixin):
    """Normalizes text capitalization to group similar categorical strings."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        # Estandarización de texto para evitar categorías duplicadas (ej. 'Móvil' vs 'móvil')
        text_cols = ['Campaign_Name', 'Location', 'Device']
        for col in text_cols:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].astype(str).str.title().str.strip()
        return X_copy

class MonetaryCleanerTransformer(BaseEstimator, TransformerMixin):
    """Converts monetary strings (e.g., '$1,234.50') into clean float values."""
    def __init__(self, columns=None):
        self.columns = columns if columns else ['Cost', 'Sale_Amount']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        # Limpieza de símbolos financieros para permitir operaciones matemáticas
        for col in self.columns:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].replace(r'[\$,]', '', regex=True)
                X_copy[col] = pd.to_numeric(X_copy[col], errors='coerce')
        return X_copy

class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """Drops columns that cause data leakage or have no predictive value."""
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop if columns_to_drop else []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop, errors='ignore')

class DropHighMissingTransformer(BaseEstimator, TransformerMixin):
    """Drops columns exceeding a defined threshold of missing values."""
    def __init__(self, threshold=0.80):
        self.threshold = threshold
        self.cols_to_drop_ = []

    def fit(self, X, y=None):
        missing_pct = X.isnull().mean()
        self.cols_to_drop_ = missing_pct[missing_pct > self.threshold].index.tolist()
        return self

    def transform(self, X):
        return X.drop(columns=self.cols_to_drop_, errors='ignore')

class SmartImputerTransformer(BaseEstimator, TransformerMixin):
    """
    Leakage-free adaptive imputer. 
    Learns median/mode during fit() and safely applies them during transform().
    """
    def __init__(self, low_threshold=0.10):
        self.low_threshold = low_threshold
        self.impute_values_ = {}

    def fit(self, X, y=None):
        # PREVENCIÓN DE FUGA DE DATOS: Aprendemos las métricas solo con el set de entrenamiento
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                self.impute_values_[col] = X[col].median()
            else:
                self.impute_values_[col] = X[col].mode()[0] if not X[col].mode().empty else 'Unknown'
        return self

    def transform(self, X):
        # Aplicamos los valores aprendidos para rellenar los vacíos en datos nuevos
        return X.fillna(self.impute_values_)
class OutlierCapper(BaseEstimator, TransformerMixin):
    """Caps outliers in numeric columns using the 1.5 * IQR rule."""
    def __init__(self, apply_capping=True):
        self.apply_capping = apply_capping
        self.bounds_ = {}

    def fit(self, X, y=None):
        if self.apply_capping:
            for col in X.columns:
                if pd.api.types.is_numeric_dtype(X[col]):
                    q1 = X[col].quantile(0.25)
                    q3 = X[col].quantile(0.75)
                    iqr = q3 - q1
                    self.bounds_[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        return self

    def transform(self, X):
        X_copy = X.copy()
        if self.apply_capping:
            for col, (lower, upper) in self.bounds_.items():
                if col in X_copy.columns:
                    X_copy[col] = X_copy[col].clip(lower=lower, upper=upper)
        return X_copy

class DropZeroVarianceTransformer(BaseEstimator, TransformerMixin):
    """Drops numeric columns with zero variance."""
    def __init__(self):
        self.zero_var_cols_ = []

    def fit(self, X, y=None):
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                # Maneja arrays con un solo elemento único como 0 variance
                if X[col].nunique() <= 1:
                    self.zero_var_cols_.append(col)
        return self

    def transform(self, X):
        return X.drop(columns=self.zero_var_cols_, errors='ignore')
