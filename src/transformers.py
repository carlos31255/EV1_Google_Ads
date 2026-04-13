# Módulo de Transformadores Personalizados de Scikit-Learn.

#
# Objetivos implementados aquí (Carlos):
#   C. Limpieza Monetaria                  — columnas Cost, Sale_Amount ('$1,234' -> float)
#   D. Eliminación de Columnas Leak        — Ad_ID, Ad_Date, Cost, Sale_Amount
#   E. Eliminación por Nulos Altos         — columnas con > 80% de valores faltantes
#   F. Capping de Outliers (IQR)           — con interruptor apply_capping on/off
#   G. Eliminación de Varianza Cero        — columnas numéricas constantes
#   H. Imputación Inteligente              — mediana/moda según porcentaje de nulos
#
# Objetivos del compañero: (pendiente de integrar al pipeline luego de finalizar)
#   A. Estandarización de Fechas           — columna Ad_Date
#   B. Limpieza de Texto / Fuzzy Matching  — columnas Campaign_Name, Location, Device
#

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# C. LIMPIEZA DE COLUMNAS MONETARIAS

# Esta clase convierte columnas de texto monetario tipo '$1,234.56' a valores float.
# Es necesaria porque Cost y Sale_Amount llegan como object (string) en el dataset crudo.
class MonetaryCleanerTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):
        # Por defecto actúa sobre las dos columnas monetarias del dataset de Google Ads
        self.columns = columns if columns is not None else ['Cost', 'Sale_Amount']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            if col in X_copy.columns:
                # Ejemplo: '$1,892.00' -> 1892.0
                X_copy[col] = (
                    X_copy[col]
                    .astype(str)
                    .str.replace(r'[$,]', '', regex=True)
                    .str.strip()
                    .replace('', np.nan)
                    .astype(float)
                )
        return X_copy

# D. ELIMINACIÓN DE COLUMNAS (Anti Data-Leakage)

# Esta clase elimina columnas especificadas del DataFrame para prevenir Data Leakage.
# Ejemplo: quitar 'Ad_ID' (identificador) y 'Cost'/'Sale_Amount' (fuente del target).
class DropColumnsTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        # Solo elimina si la columna realmente existe en el dataset
        cols = [col for col in self.columns_to_drop if col in X_copy.columns]
        return X_copy.drop(columns=cols)

# E. ELIMINACIÓN DE COLUMNAS CON ALTO PORCENTAJE DE NULOS

# Esta clase elimina columnas cuyo porcentaje de valores faltantes supera el umbral indicado.
# Ejemplo: si una columna tiene > 80% de NaN, no aporta información útil al modelo.
class DropHighMissingTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, threshold=None):
        self.threshold = threshold
        self.cols_to_drop_ = []

    def fit(self, X, y=None):
        # Aprende qué columnas superan el límite (ej. > 80% nulos)
        # Si no se especifica threshold, usa 0.8 como valor por defecto
        umbral = self.threshold if self.threshold is not None else 0.8
        pct_nulos = X.isnull().mean()
        self.cols_to_drop_ = pct_nulos[pct_nulos > umbral].index.tolist()
        return self

    def transform(self, X):
        X_copy = X.copy()
        cols = [c for c in self.cols_to_drop_ if c in X_copy.columns]
        return X_copy.drop(columns=cols)


# F. CAPPING DE OUTLIERS (IQR)

# Esta clase recorta los valores atípicos numéricos usando el método del Rango Intercuartílico (IQR).
# Puede desactivarse con apply_capping=False para comparar resultados con y sin capping.
class OutlierCapper(BaseEstimator, TransformerMixin):

    def __init__(self, apply_capping=True):
        self.apply_capping = apply_capping
        self.bounds_ = {}

    def fit(self, X, y=None):
        if not self.apply_capping:
            return self  # Si está apagado, no aprende nada y pasa de largo
        # Calcula y guarda los límites inferior y superior para cada columna numérica
        for col in X.select_dtypes(include=['number']).columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            self.bounds_[col] = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        return self

    def transform(self, X):
        X_copy = X.copy()
        if not self.apply_capping:
            return X_copy  # Si está apagado, devuelve los datos intactos
        # Aplica el recorte (capping) a los valores que se salen de los límites
        for col, (lower, upper) in self.bounds_.items():
            if col in X_copy.columns:
                X_copy[col] = np.clip(X_copy[col], lower, upper)
        return X_copy

    def get_feature_names_out(self, input_features=None):
        return input_features


# G. ELIMINACIÓN DE VARIANZA CERO

# Esta clase elimina columnas numéricas cuya desviación estándar es exactamente 0 (valores constantes).
# Ejemplo: una columna donde todos los registros tienen el mismo valor no aporta información.
class DropZeroVarianceTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.cols_to_drop_ = []

    def fit(self, X, y=None):
        # Buscamos columnas numéricas cuya desviación estándar sea exactamente 0
        num_cols = X.select_dtypes(include=['number']).columns
        self.cols_to_drop_ = [col for col in num_cols if X[col].std() == 0]
        return self

    def transform(self, X):
        X_copy = X.copy()
        cols = [c for c in self.cols_to_drop_ if c in X_copy.columns]
        return X_copy.drop(columns=cols)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        # Devuelve solo las columnas que NO fueron eliminadas
        return np.array([f for f in input_features if f not in self.cols_to_drop_])


# H. IMPUTACIÓN INTELIGENTE DE VALORES FALTANTES

# Esta clase decide la estrategia de imputación según el porcentaje de nulos de cada columna:
#   - 0% a 10%  -> Imputación simple: Mediana para numéricas, Moda para texto.
#   - 10% a 80% -> Imputación compleja: fallback temporal a simple (pendiente: KNN/Iterative).
#   - > 80%     -> Ignorado (manejado previamente por DropHighMissingTransformer).
class SmartImputerTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, low_threshold=0.10):
        self.low_threshold = low_threshold
        self.cols_simples_ = []
        self.cols_complejas_ = []

    def fit(self, X, y=None):
        porcentaje_nulos = X.isnull().mean()
        self.cols_simples_ = []
        self.cols_complejas_ = []

        for col in X.columns:
            pct = porcentaje_nulos[col]
            if 0 < pct <= self.low_threshold:
                self.cols_simples_.append(col)
            elif pct > self.low_threshold:
                self.cols_complejas_.append(col)

        print(f"[SmartImputer] Simples  (<10%): {self.cols_simples_}")
        print(f"[SmartImputer] Complejas (>10%): {self.cols_complejas_} (PENDIENTE -> KNN/Iterative)")
        return self

    def transform(self, X):
        X_copy = X.copy()

        # 1. Imputación Simple (< 10% nulos)
        for col in self.cols_simples_:
            if col not in X_copy.columns:
                continue
            if pd.api.types.is_numeric_dtype(X_copy[col]):
                X_copy[col] = X_copy[col].fillna(X_copy[col].median())
            else:
                X_copy[col] = X_copy[col].fillna(X_copy[col].mode()[0])

        # 2. Imputación Compleja (> 10% nulos) — fallback temporal a simple
        for col in self.cols_complejas_:
            if col not in X_copy.columns:
                continue
            if pd.api.types.is_numeric_dtype(X_copy[col]):
                X_copy[col] = X_copy[col].fillna(X_copy[col].median())
            else:
                X_copy[col] = X_copy[col].fillna(X_copy[col].mode()[0])

        return X_copy

    def get_feature_names_out(self, input_features=None):
        return input_features
