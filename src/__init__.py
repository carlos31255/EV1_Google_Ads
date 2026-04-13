# Paquete src — expone las funciones públicas de audit, transformers y optimization
from .audit import (
    generate_checksum,
    get_file_metadata,
    create_metadata_file,
    verify_data_integrity,
)

from .transformers import (
    DropColumnsTransformer,
    MonetaryCleanerTransformer,
    DropHighMissingTransformer,
    OutlierCapper,
    DropZeroVarianceTransformer,
    SmartImputerTransformer,
)

from .optimization import (
    optimize_memory,
    read_csv_in_chunks,
)
