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
├── notebooks/             # Análisis exploratorio y documentación del proceso
│   ├── 01_exploratory_analysis.ipynb     # EDA inicial
│   ├── 02_Pipelines.ipynb                # Construcción del pipeline (Evaluación 1)
│   ├── 03_unsupervised_modeling.ipynb    # Clustering y PCA (Evaluación 2)
│   └── 04_hyperparameter_optimization.ipynb  # Visualización Optuna (Evaluación 2)
├── src/
│   ├── __init__.py
│   ├── audit.py               # Auditoría e integridad del dataset
│   ├── transformers.py        # Transformers personalizados de scikit-learn
│   ├── pipeline.py            # Función build_preprocessing_pipeline()
│   ├── optimization.py        # Optimización de memoria y lectura por chunks
│   ├── data_preprocessing.py  # Limpieza, split y serialización del pipeline
│   ├── hyperparameter_tuning.py  # Búsqueda de hiperparámetros con Optuna
│   └── model_training.py      # Entrenamiento final con los mejores parámetros
├── models/
│   └── trained_models/    # Modelos y artefactos serializados (.joblib, .pkl, .db)
├── results/               # Resultados experimentales
│   ├── metrics/           # Reportes en CSV
│   └── plots/             # Gráficos (ROC, matriz de confusión)
├── outputs/               # Visualizaciones generadas en etapas previas
├── docs/                  # Informe técnico y documentación extra
├── main.py                # Pipeline ETL de la Evaluación 1
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

### Evaluación 1 — ETL y Preprocesamiento básico

```bash
python main.py
```

### Evaluación 2 — Modelado y Optimización (Persona A)

Los scripts se ejecutan **uno a uno** para no sobrecargar el sistema, ya que cada fase puede ser costosa computacionalmente. Se recomienda esperar a que termine cada paso antes de continuar.

**Paso 1 — Preprocesamiento y generación de splits:**
```bash
google_ads\Scripts\python src\data_preprocessing.py
```
> Genera `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv` y `preprocessing_pipeline.joblib` en `data/processed/`.

**Paso 2 — Búsqueda de hiperparámetros con Optuna (30 trials):**
```bash
google_ads\Scripts\python src\hyperparameter_tuning.py
```
> Evalúa 5 algoritmos (SVM, Random Forest, XGBoost, LightGBM, Regresión Logística) con y sin PCA. Guarda el historial completo en `models/trained_models/optuna_study.db` y la receta ganadora en `models/trained_models/best_params.pkl`.

**Paso 3 — Entrenamiento del modelo final (Persona A):**
```bash
google_ads\Scripts\python src\model_training.py
```
> Reconstruye el pipeline ganador con los parámetros de `best_params.pkl`, lo entrena con el set completo de entrenamiento y lo guarda en `models/trained_models/final_classifier.joblib`.

**Paso 4 — Evaluación sobre datos nunca vistos (Persona B):**
```bash
google_ads\Scripts\python src\model_evaluation.py
```
> Carga el modelo final (`final_classifier.joblib`) y lo evalúa contra el set de prueba apartado (`X_test.csv`). Genera un reporte tabular en `results/metrics/classification_report.csv` y gráficos (matriz de confusión y curva ROC) en `results/plots/`.

**Exploración de resultados (notebooks):**
- `notebooks/03_unsupervised_modeling.ipynb` — Clustering y PCA exploratorio
- `notebooks/04_hyperparameter_optimization.ipynb` — Visualizaciones interactivas de Optuna

## Flujo EDA actualizado

El notebook principal `notebooks/01_exploratory_analysis.ipynb` ahora sigue un flujo explícito de diagnóstico y validación:

1. Inspección inicial del dataset crudo
2. Visualización inicial (antes de limpieza: dispersión, nulos y correlación)
3. Limpieza de datos y creación del target supervisado
4. EDA posterior a limpieza
5. Comparaciones antes/después (dispersión y correlación)

Visualizaciones relevantes generadas en `outputs/`:

- `boxplot_inicial.png`
- `nulos_heatmap_inicial.png`
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
| `optuna` | Optimización automática de hiperparámetros |
| `xgboost` / `lightgbm` | Modelos de gradient boosting |
| `joblib` | Serialización de modelos y pipelines |
| `plotly` | Gráficos interactivos (visualizaciones Optuna) |
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
