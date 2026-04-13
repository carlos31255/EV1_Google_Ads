# Módulo de Optimización de Memoria.

#
# Objetivos implementados aquí:
#   A. Optimización de Tipos Numéricos  — reduce el footprint del DataFrame en RAM
#   B. Procesamiento por Chunks         — permite leer archivos grandes sin agotar la memoria
#

import pandas as pd
import numpy as np


# A. OPTIMIZACIÓN DE TIPOS NUMÉRICOS

# Esta función reduce el uso de memoria de un DataFrame bajando la precisión
# de los tipos numéricos al mínimo necesario para representar sus valores.
# Ejemplo: una columna int64 que solo tiene valores 0-100 pasa a int8 (8x menos memoria).
def optimize_memory(df):

    try:
        original_mem = df.memory_usage(deep=True).sum() / 1024**2
        print(f"Memoria original: {original_mem:.2f} MB")

        df_opt = df.copy()

        for col in df_opt.select_dtypes(include=['int', 'float']).columns:
            try:
                orig_type = df_opt[col].dtype
                c_min = df_opt[col].min()
                c_max = df_opt[col].max()

                # Conversión de enteros: int64 -> int8 / int16 / int32 según rango
                if str(orig_type).startswith('int'):
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df_opt[col] = df_opt[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df_opt[col] = df_opt[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df_opt[col] = df_opt[col].astype(np.int32)

                # Conversión de decimales: float64 -> float32 si el rango lo permite
                elif str(orig_type).startswith('float'):
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df_opt[col] = df_opt[col].astype(np.float32)

            except BaseException as e:
                # Si una columna falla, advertimos pero el ciclo for continúa
                print(f"WARNING: No se pudo optimizar la columna '{col}': {e}")
                continue

        final_mem = df_opt.memory_usage(deep=True).sum() / 1024**2
        savings = 100 * (original_mem - final_mem) / original_mem

        print(f"Memoria optimizada: {final_mem:.2f} MB")
        print(f"Ahorro total: {savings:.1f}%")
        return df_opt

    except Exception as e:
        print(f"❌ ERROR CRITICO en optimización de memoria: {e}")
        # En caso de fallo total, devolvemos el DataFrame original intacto
        return df


# B. PROCESAMIENTO POR CHUNKS

# Esta función lee un archivo CSV grande en fragmentos (chunks) para evitar
# agotar la memoria RAM. Aplica optimize_memory a cada chunk antes de unirlos.
# Ejemplo: un archivo de 2 GB puede procesarse en chunks de 10,000 filas.
def read_csv_in_chunks(filepath, chunksize=10_000, sep=','):

    chunks = []

    try:
        print(f"Leyendo '{filepath}' en chunks de {chunksize:,} filas...")

        for i, chunk in enumerate(pd.read_csv(filepath, sep=sep, chunksize=chunksize)):
            # Optimizamos la memoria de cada chunk antes de acumularlo
            chunk_opt = optimize_memory(chunk)
            chunks.append(chunk_opt)
            print(f"  Chunk {i+1} procesado: {len(chunk):,} filas")

        df_final = pd.concat(chunks, ignore_index=True)
        print(f"Lectura completada: {df_final.shape[0]:,} filas x {df_final.shape[1]} columnas")
        return df_final

    except Exception as e:
        print(f"❌ ERROR CRITICO al leer el archivo por chunks: {e}")
        return None
