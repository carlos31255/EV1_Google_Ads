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


def build_profit_target(cost_raw, sale_raw, tau_candidates=(0.1, 0.2, 0.3)):
    """
    Construye la variable objetivo `Is_Profitable` usando margen y selección de tau.

    Reglas:
    - Profit_Margin = (Sale_Amount - Cost) / Cost
    - Is_Profitable = 1 si Profit_Margin >= tau, 0 en caso contrario

    Selección de tau:
    1. Evalúa candidatos fijos (`tau_candidates`).
    2. Elige el que conserve ambas clases y quede más cerca de 50/50.
    3. Si todos colapsan en clase única, aplica fallback dinámico con el
       cuantil 0.70 del margen para obtener una separación útil.

    Los registros con `Cost` o `Sale_Amount` nulos quedan con target
    desconocido (`NA`) para excluirse del set supervisado.

    Returns
    -------
    tuple[pd.Series, pd.Series, float, str]
        target nullable Int64, máscara de target conocido, tau elegido y modo
        de selección (`fixed` o `dynamic_fallback`).
    """
    known_target_mask = cost_raw.notna() & sale_raw.notna()
    margin = (sale_raw - cost_raw) / cost_raw

    best_tau = None
    best_score = None
    best_target = None
    selection_mode = "fixed"

    known_margin = margin[known_target_mask]
    for tau in tau_candidates:
        y_tau = (known_margin >= tau).astype(int)
        pos_rate = y_tau.mean()
        score = abs(pos_rate - 0.5)
        if 0 < pos_rate < 1:
            if best_score is None or score < best_score:
                best_score = score
                best_tau = tau
                best_target = y_tau

    if best_target is None:
        finite_margin = known_margin.replace([np.inf, -np.inf], np.nan).dropna()
        if finite_margin.empty:
            raise ValueError("No hay margen válido para construir Is_Profitable.")
        best_tau = float(finite_margin.quantile(0.70))
        best_target = (known_margin >= best_tau).astype(int)
        selection_mode = "dynamic_fallback"

    target = pd.Series(pd.NA, index=cost_raw.index, dtype='Int64')
    target.loc[known_target_mask] = best_target.astype('Int64')

    return target, known_target_mask, best_tau, selection_mode


def main():
    """
    Orquesta el pipeline completo de ETL: auditoría, carga, optimización,
    creación del target por margen y tau, preprocesamiento y guardado.
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

        # 4. Creación de la variable objetivo Is_Profitable (antes de imputación/capping)
        # Se calcula sobre valores monetarios crudos parseados.
        # Si Cost o Sale_Amount están nulos, el target se marca como desconocido.
        print("\n4. Creando variable objetivo Is_Profitable...")
        cost_raw = pd.to_numeric(
            df_opt['Cost'].astype(str).str.replace(r'[$,]', '', regex=True),
            errors='coerce'
        )
        sale_raw = pd.to_numeric(
            df_opt['Sale_Amount'].astype(str).str.replace(r'[$,]', '', regex=True),
            errors='coerce'
        )
        df_opt['Is_Profitable'], known_target_mask, best_tau, selection_mode = build_profit_target(
            cost_raw=cost_raw,
            sale_raw=sale_raw,
            tau_candidates=(0.1, 0.2, 0.3),
        )

        unknown_count = int((~known_target_mask).sum())
        known_count = int(known_target_mask.sum())
        dist = (
            df_opt.loc[known_target_mask, 'Is_Profitable']
            .value_counts(normalize=True)
            .mul(100)
        )
        print(f"   Tau seleccionado: {best_tau:.4f} ({selection_mode})")
        print(f"   Targets conocidos: {known_count}  |  Desconocidos excluibles: {unknown_count}")
        print(f"   Rentable (1): {dist.get(1, 0):.1f}%  |  No rentable (0): {dist.get(0, 0):.1f}%")

        # 5. Construcción y aplicación del pipeline de preprocesamiento
        # Excluimos del set supervisado los registros con target desconocido.
        print("\n5. Construyendo y aplicando pipeline de preprocesamiento...")
        df_supervised = df_opt.loc[known_target_mask].copy()
        X = df_supervised.drop(columns=['Is_Profitable'])
        y = df_supervised['Is_Profitable'].astype(int)
        pipeline = build_preprocessing_pipeline()
        processed_matrix = pipeline.fit_transform(X)

        # 6. Guardado del dataset procesado
        print("\n6. Guardando dataset procesado...")
        feature_names = pipeline.named_steps['preprocessing'].get_feature_names_out()
        feature_names = [n.replace('num__', '').replace('cat__', '') for n in feature_names]

        df_processed = pd.DataFrame(processed_matrix, columns=feature_names)
        df_processed['Is_Profitable'] = y.values

        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        output_path = processed_dir / "GoogleAds_Processed.csv"

        df_processed.to_csv(output_path, index=False)
        print(f"✅ Dataset procesado guardado en {output_path}")
        print(f"   Dimensiones finales (supervisado): {df_processed.shape[0]:,} filas x {df_processed.shape[1]} columnas")

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
