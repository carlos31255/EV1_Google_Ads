"""
Main orchestration script for Google Ads Analytics.
Handles target engineering (ROI), preprocessing, and data exportation.

Usage:
    python main.py
"""

import sys
import pandas as pd
from pathlib import Path

# Imports locales del proyecto
from src.audit import verify_data_integrity
from src.optimization import optimize_memory
from src.pipeline import build_preprocessing_pipeline

def build_profit_target(cost_raw, sale_raw, tau=0.2):
    """
    Engineers the target variable (Is_Profitable) based on Return on Investment (ROI).
    If Profit_Margin >= tau, it's considered profitable (1), else (0).
    """
    cost_clean = pd.to_numeric(cost_raw.replace(r'[\$,]', '', regex=True), errors='coerce')
    sale_clean = pd.to_numeric(sale_raw.replace(r'[\$,]', '', regex=True), errors='coerce')
    
    # Cálculo del margen de ganancia (ROI)
    profit_margin = (sale_clean - cost_clean) / cost_clean
    
    # Binarización basada en el umbral 'tau'
    target = (profit_margin >= tau).astype(int)
    
    # Marcamos como nulos aquellos donde faltaba información financiera
    mask_valid = cost_clean.notna() & sale_clean.notna()
    target = target.where(mask_valid, pd.NA)
    
    return target, mask_valid

def main():
    """Executes the complete ETL and Machine Learning preparation pipeline."""
    print("="*60)
    print("📈 PIPELINE DE DATOS: GOOGLE ADS ANALYTICS")
    print("="*60)

    try:
        # 1. Carga de Datos
        print("\n📥 Fase 1: Extracción de datos")
        raw_path = Path("data/raw/GoogleAds_DataAnalytics_Sales_Uncleaned.csv")
        
        if not raw_path.exists():
            print(f"❌ Error: Archivo no encontrado en {raw_path}")
            return
            
        df_raw = pd.read_csv(raw_path)

        # 2. Ingeniería de la Variable Objetivo (ROI)
        print("\n🎯 Fase 2: Construcción de Variable Objetivo (Is_Profitable)")
        if 'Cost' in df_raw.columns and 'Sale_Amount' in df_raw.columns:
            # Aplicamos la lógica de negocio para crear el Target
            target_series, valid_mask = build_profit_target(df_raw['Cost'], df_raw['Sale_Amount'])
            
            # Filtramos solo los registros con métricas financieras válidas
            df_valid = df_raw[valid_mask].copy()
            y = target_series[valid_mask].astype(int)
        else:
            print("❌ Error: Faltan columnas financieras para calcular el ROI.")
            return

        # 3. Optimización de Memoria
        print("\n⚙️  Fase 3: Optimización de recursos")
        df_opt = optimize_memory(df_valid)

        # 4. Pipeline de Preprocesamiento
        print("\n🏗️  Fase 4: Transformación y limpieza (Pipeline)")
        columns_to_drop = ['Ad_ID', 'Ad_Date', 'Cost', 'Sale_Amount']
        pipeline = build_preprocessing_pipeline(columns_to_drop=columns_to_drop)
        
        # Procesamiento matemático (El pipeline ya no filtrará datos al target)
        processed_matrix = pipeline.fit_transform(df_opt)

        # Reconstrucción de Dataframe
        try:
            feature_names = pipeline.named_steps['preprocessing'].get_feature_names_out()
            feature_names = [n.split('__')[-1] for n in feature_names]
        except Exception:
            feature_names = [f"feat_{i}" for i in range(processed_matrix.shape[1])]

        df_processed = pd.DataFrame(processed_matrix, columns=feature_names, index=df_opt.index)
        
        # Reacoplamos el target limpio
        df_processed['Is_Profitable'] = y.values

        # 5. Guardado Final
        print("\n💾 Fase 5: Exportación de datos limpios")
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        output_path = processed_dir / "GoogleAds_Processed.csv"

        df_processed.to_csv(output_path, index=False)
        
        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
        print("="*60)
        print(f"📊 Dataset final: {df_processed.shape[0]} filas x {df_processed.shape[1]} columnas")
        print(f"📁 Guardado en:   {output_path}\n")

    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")

if __name__ == "__main__":
    main()