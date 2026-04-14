# Punto de entrada del proyecto Google Ads Analytics.
# Orquesta el pipeline completo: auditoría, carga, optimización, preprocesamiento y guardado.
#
# Uso:
#   python main.py

import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from pathlib import Path

from src.audit import create_metadata_file, verify_data_integrity
from src.optimization import optimize_memory
from src.pipeline import build_preprocessing_pipeline
from src.transformers import MonetaryCleanerTransformer



def main():
    """
    Orquesta el pipeline completo de ETL: auditoría, carga, optimización, creación
    de la variable objetivo, preprocesamiento y guardado del dataset procesado.
    """
    print("--- Iniciando Pipeline de Datos ---\n")

    try:
        # 1. Auditoría — verifica que el CSV no haya sido alterado desde su origen
        raw_dir = Path("data/raw")
        csv_file = list(raw_dir.glob("*.csv"))[0]
        metadata_path = raw_dir / "metadata.json"

        print("1. Auditando integridad del dataset...")
        if not metadata_path.exists():
            print("   Metadata no encontrada — generando por primera vez...")
            create_metadata_file(str(csv_file), str(metadata_path))

        if not verify_data_integrity(str(csv_file), str(metadata_path)):
            print("\n❌ Pipeline detenido: fallo en la verificación de integridad.")
            return

        # 2. Carga dinámica — toma el primer CSV que encuentre en data/raw/
        print(f"\n2. Cargando datos desde {csv_file.name}...")
        df_raw = pd.read_csv(csv_file)
        print(f"   Shape original: {df_raw.shape[0]:,} filas x {df_raw.shape[1]} columnas")

        # 3. Optimización de memoria antes de procesar
        print("\n3. Optimizando memoria del DataFrame...")
        df_opt = optimize_memory(df_raw)

        # 4. Creación de la variable objetivo Is_Profitable
        # Lógica de negocio: rentable si Sale_Amount > Cost
        # Usamos MonetaryCleanerTransformer para no duplicar la lógica de limpieza
        print("\n4. Creando variable objetivo Is_Profitable...")
        cleaner = MonetaryCleanerTransformer(columns=['Cost', 'Sale_Amount'])
        df_clean = cleaner.fit_transform(df_opt)
        df_opt['Is_Profitable'] = np.where(
            df_clean['Sale_Amount'] > df_clean['Cost'],
            1, 0
        )
        dist = df_opt['Is_Profitable'].value_counts(normalize=True) * 100
        print(f"   Rentable (1): {dist.get(1, 0):.1f}%  |  No rentable (0): {dist.get(0, 0):.1f}%")

        # 5. Construcción y aplicación del pipeline de preprocesamiento
        print("\n5. Construyendo y aplicando pipeline de preprocesamiento...")
        X = df_opt.drop(columns=['Is_Profitable'])
        pipeline = build_preprocessing_pipeline()
        processed_matrix = pipeline.fit_transform(X)

        # 6. Guardado del dataset procesado
        print("\n6. Guardando dataset procesado...")
        feature_names = pipeline.named_steps['preprocessing'].get_feature_names_out()
        feature_names = [n.replace('num__', '').replace('cat__', '') for n in feature_names]

        df_processed = pd.DataFrame(processed_matrix, columns=feature_names)
        df_processed['Is_Profitable'] = df_opt['Is_Profitable'].values

        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        output_path = processed_dir / "GoogleAds_Processed.csv"

        df_processed.to_csv(output_path, index=False)
        print(f"✅ Dataset procesado guardado en {output_path}")
        print(f"   Dimensiones finales: {df_processed.shape[0]:,} filas x {df_processed.shape[1]} columnas")

    # Manejo de errores específicos
    except IndexError:
        print("\n❌ ERROR CRITICO: No se encontró ningún archivo CSV en 'data/raw/'.")
    except FileNotFoundError as e:
        print(f"\n❌ ERROR CRITICO: Archivo o directorio no encontrado: {e}")
    except pd.errors.EmptyDataError:
        print("\n❌ ERROR CRITICO: El archivo CSV está completamente vacío.")
    except pd.errors.ParserError:
        print("\n❌ ERROR CRITICO: Pandas no pudo leer el CSV. Revisa el separador o formato.")
    except Exception as e:
        print(f"\n❌ ERROR FATAL: El pipeline falló inesperadamente: {e}")


if __name__ == "__main__":
    main()
