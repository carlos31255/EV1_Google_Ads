# Google Ads Analytics

Análisis exploratorio, limpieza y preprocesamiento de datos de campañas publicitarias de Google Ads, orientado a identificar patrones de rendimiento, eficiencia del gasto y oportunidades de optimización. Incluye un pipeline de ML automatizado para predecir rentabilidad de campañas (`Is_Profitable`).

## Estado actual del target

- `Is_Profitable` se calcula con margen de rentabilidad: `(Sale_Amount - Cost) / Cost`.
- Se evalúan `tau = {0.1, 0.2, 0.3}`; si hay clase degenerada, se aplica `tau` dinámico (cuantil 70 del margen).
- Registros con `Cost` o `Sale_Amount` nulos se marcan como target desconocido y se excluyen del set supervisado.
- Resultado actual del set supervisado: 2,366 filas, con distribución aproximada 70% clase `0` y 30% clase `1`.

## Estructura del proyecto

```
google_ads_analytics/
├── data/
│   ├── raw/               # Dataset original sin modificar
│   └── processed/         # Datos limpios y transformados
├── notebooks/             # Análisis exploratorio y construcción del pipeline
│   ├── 01_EDA_ML_GoogleAds.ipynb
│   └── 02_Pipelines.ipynb
├── src/
│   ├── __init__.py
│   ├── audit.py           # Auditoría e integridad del dataset
│   ├── transformers.py    # Transformers personalizados de scikit-learn
│   ├── pipeline.py        # Función build_preprocessing_pipeline()
│   └── optimization.py    # Optimización de memoria y lectura por chunks
├── outputs/               # Visualizaciones generadas
├── docs/                  # Informe técnico y documentación extra
├── main.py                # Punto de entrada — ejecuta el ETL completo
├── .gitignore
├── requirements.txt
└── README.md
```

## Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/carlos31255/google_ads_analytics.git
   cd google_ads_analytics
   ```

2. Crea un entorno virtual e instala las dependencias:
   ```bash
   python -m venv google_ads
   google_ads\Scripts\activate       # Windows
   pip install -r requirements.txt
   ```

## Ejecución

Para correr el pipeline completo de ETL y preprocesamiento de forma automática:

```bash
python main.py
```

Esto ejecuta en orden:
1. Carga del dataset crudo
2. Auditoría e integridad del archivo
3. Optimización de memoria
4. Creación temprana de `Is_Profitable` sobre datos monetarios crudos
5. Exclusión de registros con target desconocido
6. Pipeline de preprocesamiento (limpieza + imputación + encoding + escalado)
7. Guardado del dataset procesado en `data/processed/`

## Flujo EDA actualizado

El notebook principal `notebooks/01_EDA_ML_GoogleAds.ipynb` ahora sigue un flujo explícito de diagnóstico y validación:

1. Inspección inicial del dataset crudo
2. Visualización inicial (antes de limpieza)
3. Limpieza de datos y creación del target supervisado
4. EDA posterior a limpieza
5. Comparaciones antes/después (dispersión y correlación)

Visualizaciones relevantes generadas en `outputs/`:

- `boxplot_inicial.png`
- `correlacion_inicial.png`
- `boxplot_comparativo.png`
- `correlacion_comparativa.png`
- `outlier_capping.png`
- `dist_objetivo.png`
- `dist_numericas_por_target.png`

## Dependencias principales

| Librería | Uso |
|---|---|
| `pandas` | Manipulación y análisis de datos |
| `numpy` | Operaciones numéricas |
| `scikit-learn` | Pipeline, transformers y modelado |
| `matplotlib` / `seaborn` | Visualización |
| `jupyter` | Exploración en notebooks |

## Dataset

- **Fuente:** [Google Ads Sales Dataset — Kaggle](https://www.kaggle.com/datasets/nayakganesh007/google-ads-sales-dataset?resource=download)
- **Archivo original:** `data/raw/GoogleAds_DataAnalytics_Sales_Uncleaned.csv`
- **Descripción:** Datos de rendimiento de anuncios incluyendo impresiones, clics, conversiones, costos y métricas de ventas.

## Módulos (`src/`)

- **`audit.py`** — Verificación de integridad del dataset: checksum, metadata y validación.
- **`transformers.py`** — Transformers personalizados compatibles con `sklearn.Pipeline`: limpieza monetaria, imputación inteligente, capping de outliers, etc.
- **`pipeline.py`** — Función `build_preprocessing_pipeline()` que ensambla el pipeline completo reutilizable.
- **`optimization.py`** — Reducción del footprint en memoria y lectura de archivos grandes por chunks.
