"""
Módulo de Auditoría de Datos.
Verifica la integridad y procedencia del dataset mediante metadata y hashing SHA-256.

NOTA: El paso de generación de metadata (create_metadata_file) debe ejecutarse
UNA SOLA VEZ cuando se obtiene el dataset crudo por primera vez, para establecer
el hash oficial de referencia.
"""
import hashlib
import logging
import os
import json
from typing import Optional, Dict

# 1. Configuración del monitor de eventos (logs)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def generate_checksum(file_path: str) -> Optional[str]:
    """
    Genera el hash SHA-256 de un archivo.

    Parameters
    ----------
    file_path : str
        Ruta absoluta al archivo a hashear.

    Returns
    -------
    Optional[str]
        Cadena hexadecimal del hash, o None si el archivo no existe.
    """
    try:
        # Abrimos el archivo en modo lectura binaria ('rb') para leer bytes, no texto
        with open(file_path, "rb") as file:
            file_bytes = file.read()
            # Normalizamos finales de línea (CRLF de Windows a LF nativo) 
            # para evitar que Git cambie el hash al descargar en otra PC
            file_bytes = file_bytes.replace(b'\r\n', b'\n')
            # Generamos y retornamos la cadena hexadecimal única del archivo
            return hashlib.sha256(file_bytes).hexdigest()
    except FileNotFoundError:
        logging.error(f"Archivo no encontrado en: {file_path}")
        return None

def get_file_metadata(file_path: str) -> Optional[Dict]:
    """
    Obtiene el tamaño del archivo (en MB) y su hash SHA-256.

    Parameters
    ----------
    file_path : str
        Ruta absoluta al archivo del que se extraerá la metadata.

    Returns
    -------
    Optional[Dict]
        Diccionario con nombre, tamaño y checksum del archivo, o None si no existe.
    """
    logging.info(f"Extrayendo metadata de: {file_path}...")
    try:
        # Calculamos el tamaño del archivo y lo pasamos a Megabytes
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        file_hash = generate_checksum(file_path)
        
        # Estructuramos la información en un diccionario
        return {
            "file_name": os.path.basename(file_path),
            "size_mb": round(size_mb, 2),
            "sha256_checksum": file_hash
        }
    except FileNotFoundError:
        return None

def create_metadata_file(file_path: str, metadata_path: str) -> None:
    """
    Crea el archivo oficial metadata.json (debe ejecutarse una sola vez).

    Parameters
    ----------
    file_path : str
        Ruta al archivo CSV crudo del que se generará el hash oficial.
    metadata_path : str
        Ruta donde se guardará el archivo metadata.json resultante.
    """
    metadata = get_file_metadata(file_path)
    if metadata:
        # Guardamos el diccionario como un archivo JSON físico
        with open(metadata_path, 'w') as json_file:
            json.dump(metadata, json_file, indent=4)
        logging.info(f"Metadata oficial guardada correctamente en: {metadata_path}")

def verify_data_integrity(file_path: str, metadata_path: str) -> bool:
    """
    Compara el hash actual del archivo contra el hash oficial en metadata.json.

    Parameters
    ----------
    file_path : str
        Ruta al archivo CSV cuya integridad se quiere verificar.
    metadata_path : str
        Ruta al archivo metadata.json con el hash oficial de referencia.

    Returns
    -------
    bool
        True si el archivo no fue modificado, False si se detectó alteración.
    """
    logging.info(f"Verificando integridad de: {file_path} contra {metadata_path}")
    
    # 1. Leemos el hash oficial que guardamos previamente en el JSON
    try:
        with open(metadata_path, 'r') as json_file:
            official_metadata = json.load(json_file)
            expected_hash = official_metadata.get("sha256_checksum")
    except FileNotFoundError:
        logging.error(f"Archivo de metadata no encontrado en {metadata_path}. Ejecuta create_metadata_file primero.")
        return False

    # 2. Calculamos el hash del archivo CSV actual en este exacto momento
    current_hash = generate_checksum(file_path)
    
    # 3. Comparamos ambos hashes para asegurar que nadie alteró el CSV
    if current_hash == expected_hash:
        logging.info("ÉXITO: Integridad de datos verificada. No se detectó corrupción.")
        return True
    else:
        logging.critical("ADVERTENCIA: ¡Se detectó corrupción o manipulación del dataset!")
        logging.critical(f"Hash esperado: {expected_hash}")
        logging.critical(f"Hash actual:   {current_hash}")
        return False

if __name__ == "__main__":
    # Obtenemos la ruta base del proyecto (un nivel arriba de /src)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Rutas de prueba usando rutas absolutas
    dataset_path = os.path.join(base_dir, "data", "raw", "GoogleAds_DataAnalytics_Sales_Uncleaned.csv")
    metadata_path = os.path.join(base_dir, "data", "raw", "metadata.json")
    
    # Si es la primera vez y no hay JSON, lo creamos
    if not os.path.exists(metadata_path):
        logging.info("No se encontró metadata. Generando archivo de metadata oficial...")
        create_metadata_file(dataset_path, metadata_path)
    
    # Verificamos que el dataset crudo coincida con el JSON
    logging.info("--- Probando función de verificación ---")
    verify_data_integrity(dataset_path, metadata_path)
