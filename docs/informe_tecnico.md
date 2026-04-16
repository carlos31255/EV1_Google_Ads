# Informe Técnico — Proyecto Google Ads Analytics
## Limpieza, Transformación y Preprocesamiento de Datos para Predicción de Rentabilidad de Campañas

---

**Proyecto:** Google Ads Analytics — Clasificación Binaria de Rentabilidad  
**Dataset:** Google Ads Sales Dataset (Kaggle)  
**Fecha:** Abril 2026  
**Repositorio:** https://github.com/carlos31255/google_ads_analytics

---

## 1. Resumen Ejecutivo

El presente informe documenta el proceso completo de análisis exploratorio de datos (EDA), limpieza y transformación aplicado al dataset de campañas publicitarias de Google Ads. El objetivo central del proyecto es construir un pipeline de preprocesamiento reproducible y automatizado que transforme datos crudos con múltiples problemas de calidad en una matriz numérica lista para entrenar modelos de Machine Learning.

### Variable Objetivo

Se definió la variable binaria `Is_Profitable` mediante lógica de negocio directa:

> **Profit_Margin = (Sale_Amount - Cost) / Cost**
>
> **Is_Profitable = 1** si `Profit_Margin >= tau`, **0** en caso contrario.
>
> Para este dataset se evalúan `tau = {0.1, 0.2, 0.3}` y, si esas opciones generan una clase degenerada, se aplica un `tau` dinámico (cuantil 70 del margen) para mantener una distribución útil de clases.

Esta variable no existía en el dataset original y fue ingeniería propia del proyecto. Además, cuando `Cost` o `Sale_Amount` están nulos en crudo, el target se marca como desconocido y el registro se excluye del set supervisado.

### Resultados Clave

| Métrica | Valor |
|---|---|
| Filas originales | 2,600 |
| Columnas originales | 13 |
| Columnas finales procesadas | 22 features + 1 target |
| Reducción de memoria | 1.44 MB → 1.38 MB (3.5%) |
| Filas finales supervisadas | 2,366 (se excluyen 234 targets desconocidos) |
| Campañas rentables (clase 1) | 710 (30.0%) |
| Campañas no rentables (clase 0) | 1,656 (70.0%) |
| Transformers personalizados implementados | 8 clases sklearn |

El pipeline fue automatizado en un script `main.py` ejecutable con un solo comando (`python main.py`), que orquesta auditoría, carga, optimización de memoria, creación del target, preprocesamiento y guardado del dataset procesado.

---

## 2. Análisis Exploratorio Inicial

### 2.1 Descripción General del Dataset

El dataset `GoogleAds_DataAnalytics_Sales_Uncleaned.csv` contiene registros de anuncios individuales de una campaña de Marketing Digital para un curso de Data Analytics. Cada fila representa un anuncio con sus métricas de rendimiento.

**Dimensiones originales:** 2,600 filas × 13 columnas

| Columna | Tipo original | Descripción |
|---|---|---|
| `Ad_ID` | object | Identificador único del anuncio |
| `Campaign_Name` | object | Nombre de la campaña (con typos) |
| `Clicks` | float64 | Número de clics recibidos |
| `Impressions` | float64 | Número de impresiones del anuncio |
| `Cost` | object | Costo del anuncio (formato `$XXX.XX`) |
| `Leads` | float64 | Número de leads generados |
| `Conversions` | float64 | Número de conversiones |
| `Conversion Rate` | float64 | Tasa de conversión (Conversions / Clicks) |
| `Sale_Amount` | object | Monto de venta (formato `$X,XXX`) |
| `Ad_Date` | object | Fecha del anuncio (formatos múltiples) |
| `Location` | object | Ciudad objetivo (con variantes) |
| `Device` | object | Dispositivo (con variantes de capitalización) |
| `Keyword` | object | Palabra clave del anuncio |

### 2.2 Estadísticas Descriptivas — Variables Numéricas

| Variable | Count | Media | Std | Min | Q1 | Mediana | Q3 | Max |
|---|---|---|---|---|---|---|---|---|
| Clicks | 2,488 | 138.96 | 34.62 | 80 | 110 | 139 | 169 | 199 |
| Impressions | 2,546 | 4,523.28 | 869.93 | 3,000 | 3,764 | 4,518 | 5,279 | 5,999 |
| Leads | 2,552 | 20.00 | 6.03 | 10 | 15 | 20 | 25 | 30 |
| Conversions | 2,526 | 6.52 | 2.27 | 3 | 5 | 7 | 9 | 10 |
| Conversion Rate | 1,974 | 0.049 | 0.020 | 0.015 | 0.035 | 0.046 | 0.058 | 0.123 |

> **Observación:** `Conversion Rate` presenta el mayor porcentaje de valores faltantes (24.08%), lo que la convierte en la columna más problemática del dataset.

### 2.3 Problemas de Calidad Detectados

El análisis exploratorio reveló los siguientes problemas, ordenados por severidad y tipo:

#### A. Valores Faltantes (NaN)

| Columna | Nulos | Porcentaje |
|---|---|---|
| Conversion Rate | 626 | **24.08%** |
| Sale_Amount | 139 | 5.35% |
| Clicks | 112 | 4.31% |
| Cost | 97 | 3.73% |
| Conversions | 74 | 2.85% |
| Impressions | 54 | 2.08% |
| Leads | 48 | 1.85% |

#### B. Columnas Monetarias Codificadas como Texto

- `Cost` y `Sale_Amount` almacenadas como strings tipo `'$231.88'` y `'$1,892'`
- Impiden cualquier operación aritmética directa
- Presentes en 2,503 y 2,461 registros respectivamente (descontando nulos)

#### C. Inconsistencias de Texto (Typos y Capitalización)

| Columna | Variantes encontradas | Valor correcto |
|---|---|---|
| `Campaign_Name` | `'DataAnalyticsCourse'`, `'Data Anlytics Corse'`, `'Data Analytcis Course'`, `'Data Analytics Corse'` | Una sola campaña |
| `Location` | `'hyderabad'`, `'HYDERABAD'`, `'Hyderbad'`, `'hydrebad'` | `hyderabad` |
| `Device` | `'desktop'`, `'Desktop'`, `'DESKTOP'`, `'mobile'`, `'Mobile'`, `'MOBILE'`, `'tablet'`, `'Tablet'`, `'TABLET'` | 3 dispositivos únicos |

#### D. Fechas con Formatos Heterogéneos

La columna `Ad_Date` presentó al menos tres formatos distintos:
- `'2024-11-16'` → ISO 8601 (formato correcto)
- `'20-11-2024'` → DD-MM-YYYY
- `'2024/11/16'` → con barras en vez de guiones

#### E. Columna Identificadora sin Valor Predictivo

- `Ad_ID`: identificador único por anuncio. No aporta información al modelo y su inclusión causaría Data Leakage de tipo identificador.

### 2.4 Distribución de la Variable Objetivo

| Is_Profitable | Frecuencia | Porcentaje |
|---|---|---|
| 1 (Rentable) | 710 | 30.0% |
| 0 (No rentable) | 1,656 | 70.0% |

> **Nota importante:** La distribución final del target se calcula únicamente sobre registros con `Cost` y `Sale_Amount` válidos (2,366 filas). Los 234 registros con target desconocido se excluyen del set supervisado para evitar etiquetas artificiales.

### 2.5 Diagnóstico Visual en Datos Crudos

Se incorporó una etapa explícita de visualización **antes de la limpieza** dentro del notebook `01_EDA_ML_GoogleAds.ipynb`, con el objetivo de separar el diagnóstico inicial de la validación posterior al procesamiento.

Visualizaciones iniciales agregadas:

- `outputs/boxplot_inicial.png`: boxplot horizontal de variables numéricas crudas (`Clicks`, `Impressions`, `Leads`, `Conversions`, `Conversion Rate`) para detectar dispersión y posibles atípicos.
- `outputs/nulos_heatmap_inicial.png`: mapa de calor de valores faltantes en el dataset crudo para identificar concentración de nulos por columna (especialmente en `Conversion Rate`).
- `outputs/correlacion_inicial.png`: matriz de correlación en datos crudos para medir relaciones previas entre variables numéricas.

Esta separación mejora la trazabilidad analítica: primero se identifica el problema real del dataset y luego se verifica el efecto de las transformaciones aplicadas.

En particular, el mapa de calor de nulos reforzó la decisión metodológica de imputación al evidenciar que la ausencia de datos no era homogénea entre columnas.

---

## 3. Metodología de Transformación

### 3.1 Arquitectura del Pipeline

Se diseñó un pipeline secuencial de scikit-learn (`sklearn.Pipeline`) que garantiza la reproducibilidad y previene Data Leakage entre conjuntos de entrenamiento y prueba. Cada paso del pipeline es un transformer personalizado que hereda de `BaseEstimator` y `TransformerMixin`.

```
Datos Crudos (2,600 × 13)
      |
   |-- [Target] Is_Profitable por margen en crudo + seleccion de tau
   |-- [Target] Exclusión de registros con target desconocido
   |
      |-- [A] DateStandardizerTransformer   → estandariza fechas + extrae features
      |-- [B] TextNormalizerTransformer      → normaliza texto con fuzzy matching
      |-- [C] DropColumnsTransformer         → elimina columnas de leakage
      |-- [D] MonetaryCleanerTransformer     → convierte '$X,XXX' → float
      |-- [E] DropHighMissingTransformer     → elimina columnas > 80% nulos
      |-- [F] SmartImputerTransformer        → imputa según porcentaje de nulos
      |-- [G] ColumnTransformer
              |-- num_pipe: [OutlierCapper → DropZeroVariance → StandardScaler]
              |-- cat_pipe: [OneHotEncoder]
      |
Datos Procesados Supervisados (2,366 × 22 features)
```

### 3.2 Transformers Implementados

#### Transformer A — DateStandardizerTransformer

**Problema que resuelve:** La columna `Ad_Date` llegaba en múltiples formatos (`YYYY-MM-DD`, `DD-MM-YYYY`, `YYYY/MM/DD`), siendo incompatible con procesamiento numérico.

**Técnica empleada:** Parseo secuencial con `pd.to_datetime`, primero con `dayfirst=False` (ISO) y en caso de fallo con `dayfirst=True`. Luego se extraen tres features temporales numéricas:
- `Ad_Date_month`: mes del anuncio (1-12)
- `Ad_Date_day_of_week`: día de la semana (0=Lunes, 6=Domingo)
- `Ad_Date_is_weekend`: indicador binario de fin de semana

**Justificación:** El comportamiento de las campañas varía por estacionalidad (mes) y por tipo de día (semana vs. fin de semana). Extraer estos features convierte información temporal en variables predictoras útiles. La columna original se elimina posteriormente por `DropColumnsTransformer`.

#### Transformer B — TextNormalizerTransformer

**Problema que resuelve:** Las columnas `Campaign_Name`, `Location` y `Device` presentaban múltiples variantes por typos y capitalización inconsistente. Si se dejaban sin corregir, el `OneHotEncoder` generaría dummies independientes para lo que debería ser un mismo valor.

**Técnica empleada:** Durante `fit()`, se aprende la lista canónica de valores normalizando a minúsculas. Durante `transform()`, cada valor se mapea al canónico más cercano usando `difflib.get_close_matches()` con un umbral de similitud de 0.6.

**Justificación:** La normalización por similitud de cadenas es más robusta que la normalización por mayúsculas/minúsculas simple, ya que resuelve typos estructurales (`'Hyderbad'` → `'hyderabad'`). El módulo `difflib` de la librería estándar de Python permite hacerlo sin dependencias externas.

**Impacto medible:** Antes de la normalización, `Device` tenía 9 categorías únicas (3 dispositivos × 3 variantes de capitalización). Después, 3 categorías únicas.

#### Transformer C — MonetaryCleanerTransformer

**Problema que resuelve:** `Cost` y `Sale_Amount` almacenadas como texto tipo `'$231.88'` impiden comparaciones y operaciones numéricas.

**Técnica empleada:** Remoción del símbolo `$` y comas mediante expresión regular, seguido de conversión a `float`. Los strings vacíos se convierten a `NaN` para preservar la información de ausencia.

**Justificación:** Esta limpieza es necesaria para transformar variables monetarias en numéricas dentro del pipeline de features. La variable objetivo se calcula antes, sobre valores monetarios crudos parseados, para preservar la lógica de negocio sin contaminación por imputación o capping.

#### Transformer D — DropColumnsTransformer

**Problema que resuelve:** Columnas que causarían Data Leakage directo o que no aportan valor predictivo.

**Columnas eliminadas:**
- `Ad_ID`: identificador único, no tiene poder predictivo
- `Ad_Date`: reemplazada por las tres features temporales del Transformer A
- `Cost` y `Sale_Amount`: son la fuente directa de `Is_Profitable` (si el modelo las viera, conocería la respuesta de antemano)

**Justificación:** Incluir `Cost` o `Sale_Amount` como features en un modelo que predice `Is_Profitable` (derivado directamente del margen monetario) sería fuga de información directa. El modelo aprendería la regla de etiquetado y no generalizaría.

#### Transformer E — DropHighMissingTransformer

**Problema que resuelve:** Columnas con proporciones extremas de valores faltantes aportan más ruido que señal.

**Umbral utilizado:** 80% de nulos. Ninguna columna del dataset superó este umbral en los datos actuales, por lo que ninguna fue eliminada en esta ejecución. Sin embargo, el transformer actúa como salvaguarda para versiones futuras del dataset.

**Justificación:** Una columna con más del 80% de valores faltantes requeriría imputar la mayoría de sus valores, lo que introduciría más sesgo que beneficio.

#### Transformer F — SmartImputerTransformer

**Problema que resuelve:** El dataset contiene entre 1.85% y 24.08% de valores faltantes en columnas numéricas clave.

**Estrategia de decisión:**
- **< 10% nulos (simple):** Imputación por mediana para numéricas, moda para categóricas. Afecta: `Clicks` (4.3%), `Impressions` (2.1%), `Leads` (1.9%), `Conversions` (2.9%).
- **10% - 80% nulos (compleja):** Fallback temporal a mediana. Afecta: `Conversion Rate` (24.1%), `Ad_Date_month` y `Ad_Date_day_of_week` (fechas no parseables).

**Justificación:** La separación por umbrales permite aplicar distintas estrategias en función del impacto potencial del sesgo de imputación. Columnas con pocos nulos son seguras con la mediana; columnas con muchos nulos requieren métodos más sofisticados. La implementación actual es un fallback temporal con documentación explícita de la deuda técnica (`PENDIENTE → KNN/Iterative`).

#### Pipeline Numérico — OutlierCapper + StandardScaler

**OutlierCapper (IQR):** Recorta valores fuera del rango `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`. Incluye un interruptor `apply_capping=True/False` para experimentación controlada.

**StandardScaler:** Normaliza cada columna numérica a media 0 y desviación estándar 1. Garantiza que no haya dominancia de escala entre variables (ej. `Impressions` en miles vs. `Conversions` en unidades).

#### Pipeline Categórico — OneHotEncoder

**OneHotEncoder:** Convierte variables categóricas en variables dummy binarias. El parámetro `handle_unknown='ignore'` permite que el pipeline maneje categorías no vistas durante el entrenamiento sin causar errores en producción.

### 3.3 Prevención de Data Leakage

El orden del pipeline garantiza que:
1. La normalización de texto (B) ocurre antes de codificar con OneHot
2. Las fechas se procesan antes de eliminar la columna original
3. `Cost` y `Sale_Amount` se eliminan antes de que el ColumnTransformer las vea
4. La creación de `Is_Profitable` ocurre en `main.py` antes de llamar al pipeline, usando margen en datos crudos
5. Los registros con target desconocido (`Cost` o `Sale_Amount` nulos) se excluyen del set supervisado antes del `fit_transform`
6. La columna objetivo se separa como `y` antes de pasar `X` al pipeline

### 3.4 Módulos de Soporte

#### audit.py — Integridad del Dataset

Implementa verificación mediante hash SHA-256 del archivo CSV crudo. Al primer uso genera un `metadata.json` con el hash oficial. En cada ejecución posterior compara el hash actual contra el oficial, detectando cualquier modificación accidental o intencional del dataset.

#### optimization.py — Optimización de Memoria

Implementa downcasting de tipos numéricos: `int64 → int8/16/32` y `float64 → float32` según el rango real de cada columna. También implementa lectura de archivos grandes por chunks para proyectos que escalen más allá de la memoria disponible.

---

## 4. Resultados y Validación

### 4.1 Transformación Dimensional

| Etapa | Filas | Columnas |
|---|---|---|
| Dataset crudo | 2,600 | 13 |
| Después de crear target y excluir desconocidos | 2,366 | 13 + target |
| Después de Transformers A+B (nuevas cols de fecha) | 2,366 | 16 |
| Después de DropColumnsTransformer | 2,366 | 12 |
| Después de SmartImputer (sin nulos) | 2,366 | 12 |
| Dataset procesado final (con encoding) | 2,366 | 22 features + 1 target |

### 4.2 Reducción de Nulos

| Columna | Nulos antes | Nulos después |
|---|---|---|
| Clicks | 112 (4.3%) | 0 ✅ |
| Impressions | 54 (2.1%) | 0 ✅ |
| Leads | 48 (1.9%) | 0 ✅ |
| Conversions | 74 (2.9%) | 0 ✅ |
| Conversion Rate | 626 (24.1%) | 0 ✅ (imputada por mediana) |
| Cost | 97 (3.7%) | Eliminada (leakage) |
| Sale_Amount | 139 (5.4%) | Eliminada (leakage) |

### 4.3 Corrección de Inconsistencias de Texto

| Columna | Valores únicos antes | Valores únicos después |
|---|---|---|
| Campaign_Name | 4 variantes de 1 campaña | 1 valor canónico |
| Location | 4 variantes de 1 ciudad | 1 valor canónico |
| Device | 9 variantes de 3 dispositivos | 3 valores canónicos |
| Keyword | 6 valores (sin typos graves) | 6 valores |

### 4.4 Validación de Integridad

El módulo `audit.py` verificó mediante SHA-256 que el archivo CSV crudo no fue modificado desde su obtención original:

```
INFO: Verifying integrity for: data/raw/GoogleAds_DataAnalytics_Sales_Uncleaned.csv
INFO: SUCCESS: Data integrity verified. No corruption detected.
```

### 4.5 Ejecución del Pipeline Completo

La ejecución de `python main.py` produce la siguiente salida verificada:

```
--- Iniciando Pipeline de Datos ---

1. Auditando integridad del dataset...
   INFO: SUCCESS: Data integrity verified. No corruption detected.

2. Cargando datos desde GoogleAds_DataAnalytics_Sales_Uncleaned.csv...
   Shape original: 2,600 filas x 13 columnas

3. Optimizando memoria del DataFrame...
   Memoria original: 1.43 MB
   Memoria optimizada: 1.38 MB
   Ahorro total: 3.5%

4. Creando variable objetivo Is_Profitable...
   Tau seleccionado: 6.9320 (dynamic_fallback)
   Targets conocidos: 2366  |  Desconocidos excluibles: 234
   Rentable (1): 30.0%  |  No rentable (0): 70.0%

5. Construyendo y aplicando pipeline de preprocesamiento...
   [SmartImputer] Simples  (<10%): ['Clicks', 'Impressions', 'Leads', 'Conversions']
   [SmartImputer] Complejas (>10%): ['Conversion Rate', 'Ad_Date_month', 'Ad_Date_day_of_week']

6. Guardando dataset procesado...
   ✅ Dataset procesado guardado en data/processed/GoogleAds_Processed.csv
   Dimensiones finales (supervisado): 2,366 filas x 23 columnas
```

### 4.6 Validación Visual Antes vs Después de la Limpieza

Además del EDA inicial y del EDA posterior a limpieza, se consolidó una sección comparativa para evaluar cambios estructurales:

- `outputs/nulos_heatmap_inicial.png`: diagnóstico visual inicial de nulos para justificar la estrategia de imputación.
- `outputs/boxplot_comparativo.png`: comparación lado a lado de dispersión en variables numéricas crudas vs procesadas.
- `outputs/correlacion_comparativa.png`: comparación de matriz de correlación inicial vs final.
- `outputs/outlier_capping.png`: comparación de distribuciones por variable antes/después del recorte IQR.

Hallazgos principales de la validación visual:

1. La limpieza no distorsiona de forma abrupta la forma global de las variables; mantiene patrones de dispersión coherentes entre el estado crudo y el procesado.
2. La estructura de correlaciones base (`Clicks`, `Impressions`, `Leads`, `Conversions`, `Conversion Rate`) se conserva en términos generales.
3. Al incorporar variables derivadas de negocio (`Profit_Margin`, `Is_Profitable`), emergen relaciones fuertes esperadas con `Sale_Amount`, validando la lógica de etiquetado y su consistencia estadística.
4. La visualización de nulos en crudo confirmó concentración en columnas críticas y respaldó la imputación selectiva aplicada en el pipeline.

---

## 5. Conclusiones y Recomendaciones

### 5.1 Conclusiones

**1. El dataset presentó una calidad de datos significativamente degradada** en múltiples dimensiones simultáneas: tipos incorrectos, formatos inconsistentes, valores faltantes, typos y columnas identificadoras sin valor predictivo. Estos problemas, de no ser corregidos, habrían producido modelos sesgados o directamente incorrectos.

**2. La arquitectura de pipeline resultó ser la decisión técnica más valiosa del proyecto.** Al encapsular cada transformación en una clase sklearn independiente y componerlas en un pipeline secuencial, se garantizan tres propiedades fundamentales para el ML:
   - **Reproducibilidad:** el mismo pipeline aplicado a datos nuevos produce resultados equivalentes
   - **Prevención de leakage:** el escalado y la imputación aprenden exclusivamente de los datos de entrenamiento
   - **Mantenibilidad:** cada transformer puede ser reemplazado o ajustado sin afectar el resto

**3. La variable objetivo `Is_Profitable` no existía en el dataset y fue creada mediante una regla de negocio robusta.** Se definió a partir del margen de rentabilidad y un umbral `tau`, calculados sobre datos monetarios crudos (antes de imputación/capping). Además, los registros con `Cost` o `Sale_Amount` faltantes se trataron como target desconocido y se excluyeron del set supervisado para evitar etiquetas artificiales.

**4. El target quedó en una distribución operativa (70% / 30%) apta para clasificación supervisada**, evitando el colapso a clase única observado con reglas más simples o umbrales fijos insuficientes.

### 5.2 Dificultades Encontradas

- **Formatos de fecha múltiples:** La columna `Ad_Date` presentó tres formatos distintos que requirieron parseo secuencial con doble pasada (dayfirst=False y dayfirst=True).
- **Columnas monetarias como strings:** `Cost` y `Sale_Amount` almacenadas con símbolo `$` impidieron su uso aritmético directo; fue necesario un transformer dedicado antes de crear el target.
- **Encoding Windows y UTF-8:** La terminal de Windows (cp1252) no soporta emojis por defecto, requiriendo `sys.stdout.reconfigure(encoding='utf-8')` en `main.py`.
- **Colapso de clase con umbrales fijos:** Los umbrales iniciales `tau = 0.1, 0.2, 0.3` mantuvieron el target en clase única para los registros conocidos. Se resolvió con un fallback dinámico (cuantil 70 del margen), logrando una distribución 70/30.
- **Definición robusta de la variable objetivo (`Is_Profitable`):** Se detectó que calcular o validar el target después de transformaciones monetarias (imputación/capping) podía desalinear la etiqueta con su regla de negocio original y, además, colapsar la clase en un 100% de positivos. Se corrigió calculando `Is_Profitable` sobre valores monetarios crudos, marcando como desconocidos los registros con `Cost` o `Sale_Amount` nulos para excluirlos del set supervisado, y calibrando un umbral de margen (tau) para conservar una distribución de clases útil.
- **Verificación de integridad cruzada (Cross-Platform Hash):** Las conversiones automáticas de salto de línea de Git (`CRLF` a `LF`) alteraban el hash SHA-256 del dataset crudo al clonarlo en sistemas Windows, marcando falsos positivos de corrupción. Se solucionó normalizando los bytes de los saltos de línea dentro de `audit.py` de forma previa al cálculo.

### 5.3 Recomendaciones

- **Mantener trazabilidad del target:** Versionar y documentar explícitamente el `tau` usado para etiquetar (`fixed` o `dynamic_fallback`) en cada corrida para asegurar reproducibilidad del conjunto supervisado.
- **Mejorar la imputación de `Conversion Rate`:** Reemplazar el fallback actual (mediana) por `KNNImputer` o `IterativeImputer`, dado que esta columna tiene un 24% de nulos.
- **Añadir tests unitarios:** Implementar pruebas con `pytest` para cada transformer, garantizando que el pipeline se comporte correctamente ante datos inesperados.

---

## Anexo — Estructura del Proyecto

```
google_ads_analytics/
├── data/
│   ├── raw/
│   │   ├── GoogleAds_DataAnalytics_Sales_Uncleaned.csv
│   │   └── metadata.json              # Hash SHA-256 para integridad
│   └── processed/
│       └── GoogleAds_Processed.csv    # Dataset limpio y transformado
├── notebooks/
│   ├── 01_EDA_ML_GoogleAds.ipynb      # EDA completo con visualizaciones
│   └── 02_Pipelines.ipynb             # Construcción paso a paso del pipeline
├── src/
│   ├── __init__.py
│   ├── audit.py                       # Verificación SHA-256 de integridad
│   ├── transformers.py                # 8 transformers sklearn personalizados
│   ├── pipeline.py                    # Función build_preprocessing_pipeline()
│   └── optimization.py               # Optimización de memoria y chunks
├── outputs/                           # Visualizaciones generadas (EDA inicial, nulos, final y comparativas)
├── docs/
│   └── informe_tecnico.md             # Este documento
├── main.py                            # Punto de entrada — python main.py
├── requirements.txt
└── README.md
```

## Anexo — Dependencias

| Librería | Versión | Uso |
|---|---|---|
| pandas | 2.2.1 | Manipulación de datos |
| numpy | 1.26.4 | Operaciones numéricas |
| scikit-learn | 1.4.1.post1 | Pipeline, transformers, preprocessing |
| matplotlib | 3.8.3 | Visualizaciones |
| seaborn | 0.13.2 | Visualizaciones estadísticas |
| difflib | stdlib | Fuzzy matching de texto |
| hashlib | stdlib | SHA-256 para auditoría |
