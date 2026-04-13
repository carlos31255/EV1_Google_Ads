# Guia de Trabajo con GitHub

Este archivo explica cómo clonar el proyecto, trabajar localmente y subir cambios a GitHub.

## Configuracion inicial (solo la primera vez)

**1. Clonar el repositorio**
```bash
git clone https://github.com/carlos31255/google_ads_analytics.git
cd google_ads_analytics
```

**2. Crear y activar el entorno virtual**
```bash
python -m venv google_ads
google_ads\Scripts\activate
```

**3. Instalar dependencias**
```bash
pip install -r requirements.txt
```

**4. Descargar el dataset**

Descarga el archivo desde [Kaggle](https://www.kaggle.com/datasets/nayakganesh007/google-ads-sales-dataset?resource=download)
y colócalo en:
```
data/raw/GoogleAds_DataAnalytics_Sales_Uncleaned.csv
```

---

## Flujo de trabajo diario

Cada vez que vayas a trabajar en el proyecto, sigue estos pasos:

**1. Traer los cambios mas recientes del repositorio**
```bash
git pull
```

**2. Hacer tus cambios localmente** (editar archivos, agregar funciones, etc.)

**3. Ver qué archivos modificaste**
```bash
git status
```

**4. Agregar los archivos al commit**
```bash
git add nombre_archivo.py   # un archivo especifico
git add .                   # o todos los archivos modificados
```

**5. Hacer el commit con un mensaje descriptivo**
```bash
git commit -m "tipo: descripcion corta de lo que hiciste"
```

**6. Subir los cambios a GitHub**
```bash
git push
```

---

## Convencion de mensajes de commit

Usar prefijos cortos para mantener el historial ordenado:

| Prefijo | Uso |
|---|---|
| `feat:` | Nueva funcion o feature |
| `fix:` | Correccion de un error |
| `refactor:` | Cambio de estructura sin modificar la logica |
| `docs:` | Cambios en documentacion o README |
| `data:` | Cambios relacionados al dataset o metadata |

Ejemplos:
```bash
git commit -m "feat: agregar funcion de limpieza de fechas"
git commit -m "fix: corregir calculo de conversion rate"
git commit -m "docs: actualizar README con nuevas instrucciones"
```

---

## Archivos que NO se suben a GitHub

El archivo `.gitignore` excluye automaticamente:

- `data/raw/*.csv` y `data/processed/*.csv` — el dataset no se sube, se descarga desde Kaggle
- `google_ads/` — el entorno virtual se crea localmente con `pip install -r requirements.txt`
- `__pycache__/`, `.ipynb_checkpoints/` — archivos temporales de Python y Jupyter
