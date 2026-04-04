# Google Ads Analytics

Análisis exploratorio y limpieza de datos de campañas publicitarias de Google Ads, orientado a identificar patrones de rendimiento, eficiencia del gasto y oportunidades de optimización.

## Estructura del proyecto

```
google_ads_analytics/
├── data/
│   ├── raw/               # Dataset original sin modificar
│   └── processed/         # Datos limpios y transformados
├── notebooks/             # Análisis exploratorio (Jupyter Notebooks)
├── src/
│   ├── __init__.py
│   ├── audit.py           # Funciones de auditoría y validación de datos
│   └── transformers.py    # Transformaciones y limpieza de datos
├── docs/                  # Documentación adicional
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
   python -m venv .venv
   .venv\Scripts\activate       # Windows
   pip install -r requirements.txt
   ```

3. Inicia Jupyter:
   ```bash
   jupyter notebook
   ```

## Dependencias principales

| Librería | Uso |
|---|---|
| `pandas` | Manipulación y análisis de datos |
| `numpy` | Operaciones numéricas |
| `scikit-learn` | Modelado y métricas |
| `matplotlib` / `seaborn` | Visualización |
| `jupyter` | Entorno de notebooks |

## Dataset

- **Fuente:** [Google Ads Sales Dataset — Kaggle](https://www.kaggle.com/datasets/nayakganesh007/google-ads-sales-dataset?resource=download)
- **Archivo original:** `data/raw/GoogleAds_DataAnalytics_Sales_Uncleaned.csv`
- **Descripción:** Datos de rendimiento de anuncios incluyendo impresiones, clics, conversiones, costos y métricas de ventas.

> Los archivos CSV no se incluyen en el repositorio (ver `.gitignore`). Descarga el dataset desde Kaggle y colócalo en `data/raw/`.

## Módulos (`src/`)

- **`audit.py`** — Funciones para auditar la calidad del dataset: valores nulos, duplicados, tipos de datos, etc.
- **`transformers.py`** — Transformaciones de datos: normalización, codificación, ingeniería de features.

