"""
Data Audit Module.
Verifies data integrity and provenance using metadata and SHA-256 hashing.

NOTE: The metadata generation step (create_metadata_file) should be executed 
ONLY ONCE when the raw data is first obtained to establish the ground truth.
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
    Generates a SHA-256 hash of a file.
    """
    try:
        # Abrimos el archivo en modo lectura binaria ('rb') para leer bytes, no texto
        with open(file_path, "rb") as file:
            file_bytes = file.read()
            # Generamos y retornamos la cadena hexadecimal única del archivo
            return hashlib.sha256(file_bytes).hexdigest()
    except FileNotFoundError:
        logging.error(f"File not found at: {file_path}")
        return None

def get_file_metadata(file_path: str) -> Optional[Dict]:
    """
    Retrieves file size (MB) and its SHA-256 hash.
    """
    logging.info(f"Extracting metadata for {file_path}...")
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
    Creates the official metadata.json file (runs only once).
    """
    metadata = get_file_metadata(file_path)
    if metadata:
        # Guardamos el diccionario como un archivo JSON físico
        with open(metadata_path, 'w') as json_file:
            json.dump(metadata, json_file, indent=4)
        logging.info(f"Official metadata saved securely at: {metadata_path}")

def verify_data_integrity(file_path: str, metadata_path: str) -> bool:
    """
    Compares current file hash against the official metadata.json.
    """
    logging.info(f"Verifying integrity for: {file_path} against {metadata_path}")
    
    # 1. Leemos el hash oficial que guardamos previamente en el JSON
    try:
        with open(metadata_path, 'r') as json_file:
            official_metadata = json.load(json_file)
            expected_hash = official_metadata.get("sha256_checksum")
    except FileNotFoundError:
        logging.error(f"Metadata file missing at {metadata_path}. Run create_metadata_file first.")
        return False

    # 2. Calculamos el hash del archivo CSV actual en este exacto momento
    current_hash = generate_checksum(file_path)
    
    # 3. Comparamos ambos hashes para asegurar que nadie alteró el CSV
    if current_hash == expected_hash:
        logging.info("SUCCESS: Data integrity verified. No corruption detected.")
        return True
    else:
        logging.critical("WARNING: Data corruption or manipulation detected!")
        logging.critical(f"Expected Hash: {expected_hash}")
        logging.critical(f"Current Hash:  {current_hash}")
        return False

if __name__ == "__main__":
    # Obtenemos la ruta base del proyecto (un nivel arriba de /src)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Rutas de prueba usando rutas absolutas
    dataset_path = os.path.join(base_dir, "data", "raw", "GoogleAds_DataAnalytics_Sales_Uncleaned.csv")
    metadata_path = os.path.join(base_dir, "data", "raw", "metadata.json")
    
    # Si es la primera vez y no hay JSON, lo creamos
    if not os.path.exists(metadata_path):
        logging.info("No metadata found. Generating official metadata file...")
        create_metadata_file(dataset_path, metadata_path)
    
    # Verificamos que el dataset crudo coincida con el JSON
    logging.info("--- Testing verification function ---")
    verify_data_integrity(dataset_path, metadata_path)
