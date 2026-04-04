"""
Data Transformers Module.
Applies cleaning and transformation steps to the raw Google Ads dataset.

Objectives implemented here:
    C. Numeric Type Transformation  — clean monetary columns (Cost, Sale_Amount)
    D. Intelligent Missing Value Imputation — business logic + median pipeline

Objectives handled by teammate:
    A. Date Standardization          — Ad_Date column
    B. Text Cleaning / Fuzzy Matching — Campaign_Name, Location columns
"""

import logging
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# ──────────────────────────────────────────────────────────────────────────────
# C. NUMERIC TYPE TRANSFORMATION
# ──────────────────────────────────────────────────────────────────────────────

def clean_monetary_column(series: pd.Series) -> pd.Series:
    """
    Strips the '$' symbol and commas from a monetary text column
    and converts it to float.

    Example: '$1,234.56' -> 1234.56
    """
    return (
        series
        .astype(str)
        .str.replace(r'[$,]', '', regex=True)
        .str.strip()
        .replace('', np.nan)        # empty strings -> NaN instead of error
        .astype(float)
    )


def transform_monetary_columns(df: pd.DataFrame,
                                columns: list = None) -> pd.DataFrame:
    """
    Applies clean_monetary_column to one or more columns in the DataFrame.

    Parameters
    ----------
    df      : Input DataFrame.
    columns : List of column names to clean. Defaults to ['Cost', 'Sale_Amount'].

    Returns
    -------
    DataFrame with cleaned numeric columns (copy, original unchanged).
    """
    if columns is None:
        columns = ['Cost', 'Sale_Amount']

    df = df.copy()
    for col in columns:
        if col in df.columns:
            before_nulls = df[col].isna().sum()
            df[col] = clean_monetary_column(df[col])
            after_nulls = df[col].isna().sum()
            new_nulls = after_nulls - before_nulls
            logging.info(
                f"[C] '{col}' converted to float. "
                f"New NaNs introduced (empty/invalid values): {new_nulls}"
            )
        else:
            logging.warning(f"[C] Column '{col}' not found in DataFrame. Skipping.")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# D. INTELLIGENT MISSING VALUE IMPUTATION
# ──────────────────────────────────────────────────────────────────────────────

def impute_conversion_rate(df: pd.DataFrame,
                            rate_col: str = 'Conversion_Rate',
                            conversions_col: str = 'Conversions',
                            clicks_col: str = 'Clicks') -> pd.DataFrame:
    """
    Recalculates missing Conversion_Rate values using business logic:
        Conversion_Rate = Conversions / Clicks

    Only fills rows where Conversion_Rate is NaN AND both Conversions
    and Clicks are available and Clicks > 0.

    Parameters
    ----------
    df              : Input DataFrame.
    rate_col        : Name of the Conversion Rate column.
    conversions_col : Name of the Conversions column.
    clicks_col      : Name of the Clicks column.

    Returns
    -------
    DataFrame with business-logic-imputed Conversion_Rate (copy).
    """
    df = df.copy()

    if rate_col not in df.columns:
        logging.warning(f"[D] Column '{rate_col}' not found. Skipping business imputation.")
        return df

    # Identify rows where we CAN calculate the rate
    missing_mask = df[rate_col].isna()
    can_calculate = (
        missing_mask
        & df[conversions_col].notna()
        & df[clicks_col].notna()
        & (df[clicks_col] > 0)
    )

    recalculated = can_calculate.sum()
    df.loc[can_calculate, rate_col] = (
        df.loc[can_calculate, conversions_col] / df.loc[can_calculate, clicks_col]
    )

    still_missing = df[rate_col].isna().sum()
    logging.info(
        f"[D] '{rate_col}': {recalculated} values recalculated via business logic. "
        f"Remaining NaNs: {still_missing}"
    )
    return df


def build_median_imputation_pipeline(numeric_columns: list) -> Pipeline:
    """
    Builds a scikit-learn Pipeline with a median SimpleImputer
    for the specified numeric columns.

    Usage
    -----
    pipeline = build_median_imputation_pipeline(['Cost', 'CTR', 'Conversion_Rate'])
    imputed_array = pipeline.fit_transform(df[numeric_columns])

    Returns
    -------
    sklearn Pipeline ready to fit/transform.
    """
    pipeline = Pipeline(steps=[
        ('median_imputer', SimpleImputer(strategy='median'))
    ])
    logging.info(
        f"[D] Median imputation pipeline created for columns: {numeric_columns}"
    )
    return pipeline


def impute_remaining_numeric(df: pd.DataFrame,
                              columns: list = None) -> pd.DataFrame:
    """
    Applies median imputation (via sklearn Pipeline) to any remaining
    NaN values in the specified numeric columns.

    Parameters
    ----------
    df      : Input DataFrame (should already have business-logic imputation applied).
    columns : Numeric columns to impute. Defaults to all numeric columns.

    Returns
    -------
    DataFrame with median-imputed values (copy).
    """
    df = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Only process columns that actually have missing values
    cols_with_nulls = [c for c in columns if c in df.columns and df[c].isna().any()]

    if not cols_with_nulls:
        logging.info("[D] No remaining NaNs found in numeric columns. Nothing to impute.")
        return df

    pipeline = build_median_imputation_pipeline(cols_with_nulls)
    imputed_values = pipeline.fit_transform(df[cols_with_nulls])
    df[cols_with_nulls] = imputed_values

    logging.info(
        f"[D] Median imputation applied to {len(cols_with_nulls)} column(s): "
        f"{cols_with_nulls}"
    )
    return df


def apply_full_imputation(df: pd.DataFrame,
                           rate_col: str = 'Conversion_Rate',
                           conversions_col: str = 'Conversions',
                           clicks_col: str = 'Clicks',
                           numeric_columns: list = None) -> pd.DataFrame:
    """
    Full imputation pipeline (Steps D1 + D2):
        1. Business-logic recalculation for Conversion_Rate.
        2. Median imputation for any remaining NaN numeric values.

    Parameters
    ----------
    df              : Raw or partially cleaned DataFrame.
    rate_col        : Conversion Rate column name.
    conversions_col : Conversions column name.
    clicks_col      : Clicks column name.
    numeric_columns : Columns to apply median imputation to.
                      If None, all numeric columns are used.

    Returns
    -------
    Fully imputed DataFrame (copy).
    """
    logging.info("[D] Starting full imputation pipeline...")
    df = impute_conversion_rate(df, rate_col, conversions_col, clicks_col)
    df = impute_remaining_numeric(df, numeric_columns)
    logging.info("[D] Full imputation pipeline complete.")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Minimal synthetic sample to verify both transformers locally
    sample = pd.DataFrame({
        'Cost':            ['$1,200.50', '$800.00', None,       '$450.75'],
        'Sale_Amount':     ['$3,000.00', None,       '$1,500.00', '$2,200.00'],
        'Clicks':          [200,          150,        300,         None],
        'Conversions':     [10,           None,       30,          20],
        'Conversion_Rate': [None,         0.05,       None,        None],
        'CTR':             [0.04,         None,       0.06,        0.03],
    })

    print("=== Before ===")
    print(sample, "\n")

    # C — Monetary columns
    sample = transform_monetary_columns(sample, columns=['Cost', 'Sale_Amount'])

    # D — Imputation
    sample = apply_full_imputation(sample)

    print("=== After ===")
    print(sample)
