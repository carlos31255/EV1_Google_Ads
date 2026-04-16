# Informe Adicional — Interpretacion de Graficos para Presentacion

---

## 1. Objetivo del informe

Este documento resume que explica cada grafico seleccionado para la presentacion del proyecto Google Ads Analytics y como usarlo de forma narrativa, clara y sin saturar.

Graficos contemplados:
- `nulos_heatmap_inicial.png`
- `boxplot_comparativo.png`
- `dist_objetivo.png`
- `dist_numericas_por_target.png` (solo subgraficos de `Conversion Rate` y `Conversions`)
- `correlacion_comparativa.png`
- `outlier_capping.png` (solo dos vistas de `Conversion Rate`: original con atipicos y recortada)

---

## 2. Lectura e interpretacion por grafico

### 2.0 `nulos_heatmap_inicial.png`

**Que muestra en este contexto**
- Muestra visualmente en que columnas y en que volumen aparecen valores nulos en el dataset crudo.
- Las zonas con mayor intensidad de color concentran mas ausencia de datos.

**Para que sirve en la presentacion**
- Justifica por que la imputacion fue necesaria (especialmente en `Conversion Rate`).
- Introduce el problema de calidad de datos antes de mostrar transformaciones mas avanzadas.

**Mensaje clave recomendado (1 frase)**
- "Este mapa de calor muestra que los nulos no estan distribuidos al azar: se concentran en columnas clave, por eso la etapa de imputacion fue obligatoria para preparar el dataset."

---

### 2.1 `boxplot_comparativo.png`

**Que muestra en este contexto**
- Compara la distribucion de variables numericas antes y despues del procesamiento.
- En este caso particular, el cambio visual es pequeno o casi nulo entre antes y despues.

**Para que sirve en la presentacion**
- Funciona como control de estabilidad: la limpieza no distorsiono artificialmente la estructura global de las variables.
- Ayuda a aclarar que el mayor impacto del pipeline estuvo en calidad de datos (nulos, texto, leakage, target) mas que en mover fuertemente la forma de todas las distribuciones.

**Mensaje clave recomendado (1 frase)**
- "Aqui el boxplot cambia poco, y eso es bueno: indica que el pipeline limpio el dato sin alterar de forma agresiva su comportamiento global."

---

### 2.2 `dist_objetivo.png`

**Que muestra en este contexto**
- Presenta la distribucion final de `Is_Profitable` (aprox. 70% no rentable y 30% rentable).
- Refleja el resultado de la logica de negocio para construir el target y del ajuste de `tau`.

**Para que sirve en la presentacion**
- Ancla todo el caso: define claramente el problema supervisado real (clasificacion binaria desbalanceada).
- Justifica decisiones posteriores de modelado (metricas y estrategias para desbalance).

**Mensaje clave recomendado (1 frase)**
- "Vemos 70% de campañas no rentables y 30% rentables. El modelo debe aprender a identificar ese pequeño grupo rentable, porque aunque son minoría, ahí está el valor del negocio."

---

### 2.3 `correlacion_comparativa.png`

**Que muestra en este contexto**
- Contrasta la estructura de correlaciones numericas en datos crudos frente a datos procesados.
- Ayuda a validar que el pipeline limpia y transforma sin destruir relaciones estadisticas relevantes.

**Para que sirve en la presentacion**
- Respalda la idea de integridad analitica: el pipeline mejora calidad sin "romper" el comportamiento entre variables.
- Funciona como soporte tecnico cuando te preguntan si la limpieza altero la logica del negocio.

**Mensaje clave recomendado (1 frase)**
- "En la matriz de correlación, los números van de -1 a 1. Rojo significa variables que se mueven juntas (fuerte relación). Azul significa relación débil o inversa. Al comparar antes y después, las relaciones importantes se mantienen igual: la limpieza no rompió la lógica de los datos."

---

### 2.4 `outlier_capping.png` (uso acotado)

**Alcance que vas a usar**
- Solo se mostraran las dos vistas de `Conversion Rate`:
  1. `Conversion Rate` con atipicos originales.
  2. `Conversion Rate` con atipicos recortados por IQR.

**Que muestra en este contexto**
- El efecto directo del capping sobre la variable numerica mas sensible en dispersion.
- Como se reduce el impacto de valores extremos sin cambiar la naturaleza de la variable.

**Para que sirve en la presentacion**
- Respuesta concreta a la pregunta: "Que hizo exactamente el recorte de outliers?"
- Buen grafico de respaldo tecnico para preguntas del jurado o cierre metodologico.

**Mensaje clave recomendado (1 frase)**
- "En Conversion Rate vemos valores extremos muy altos y muy bajos. El capping los domestica dentro de rangos razonables. Las campañas normales no cambian, pero los extremos se recortan para que cuando entrenes el modelo, no pierda tiempo aprendiendo casos raros y se enfoque en patrones reales de rentabilidad."

---

### 2.5 `dist_numericas_por_target.png` (uso selectivo: solo 2 subgraficos)

**Subgrafico 1: `Conversion Rate` vs `Is_Profitable`**
- **Que muestra:** el grado de separacion de la tasa de conversion entre clases rentables y no rentables.
- **Por que mostrarlo:** conecta directo con tu narrativa de calidad de dato (nulos y outliers) y aporta senal predictiva interpretable.
- **Mensaje clave:** "Conversion Rate muestra diferencias claras entre campañas rentables y no rentables. Por eso fue prioritaria en la limpieza: para que los datos estén listos cuando entrenes un modelo que prediga rentabilidad."

**Subgrafico 2: `Conversions` vs `Is_Profitable`**
- **Que muestra:** diferencias en volumen de conversiones entre ambas clases del target.
- **Por que mostrarlo:** traduce el analisis a impacto de negocio (resultado comercial), no solo estadistica.
- **Mensaje clave:** "Las campañas rentables tienen más conversiones. Eso es útil porque el modelo puede aprender ese patrón para predecir cuáles campañas serán rentables."

**Nota de uso en presentacion**
- Usar estos dos subgraficos como respaldo tecnico breve.
- Evitar mostrar todos los paneles de `dist_numericas_por_target.png` para no saturar la diapositiva.

---

## 3. Orden sugerido para exponer estos graficos

1. `nulos_heatmap_inicial.png` (diagnostico rapido de calidad de datos en crudo)
2. `dist_objetivo.png` (definicion del problema supervisado final)
3. `dist_numericas_por_target.png` solo con `Conversion Rate` y `Conversions` (senal predictiva por clase)
4. `correlacion_comparativa.png` (validacion estadistica del procesamiento)
5. `boxplot_comparativo.png` como apoyo breve (control de estabilidad, no de cambio grande)
6. `outlier_capping.png` solo en modo tecnico (Conversion Rate original vs recortada)

---

## 4. Guion corto de exposicion (45-60 segundos)

"Primero, con la distribucion del objetivo mostramos que el problema final es clasificacion binaria desbalanceada 70/30. Luego, con Conversion Rate y Conversions por target, evidenciamos senal predictiva util para separar clases. Despues, la correlacion comparativa confirma que el pipeline mejora calidad sin destruir relaciones relevantes. El boxplot comparativo se usa como control de estabilidad: cambia poco y confirma que no deformamos el dato. Finalmente, outlier capping queda como respaldo tecnico en Conversion Rate antes y despues."

---

## 5. Explicación del Pipeline — Qué Hace Cada Función

Para que entiendas el flujo completo de transformación, aquí van las funciones en orden:

1. **Target Constructor (`Is_Profitable`)**
   - Lee `Cost` y `Sale_Amount` en bruto.
   - Calcula margen de rentabilidad: `(Sale_Amount - Cost) / Cost`.
   - Etiqueta como rentable si margen >= `tau` (umbral ajustado para evitar clase única).
   - Excluye registros con `Cost` o `Sale_Amount` nulos antes de todo lo que sigue.
   
   **La más importante de mencionar:** El ajuste dinámico de `tau`. Sin él, la distribución se colapsa en una sola clase. Mostrar que se prueban umbrales fijos y se aplica fallback dinámico demuestra rigor metodológico.

2. **DateStandardizerTransformer**
   - Parsea fechas en múltiples formatos (`YYYY-MM-DD`, `DD-MM-YYYY`, `YYYY/MM/DD`).
   - Extrae tres features numéricas: mes, día de la semana, es_fin_de_semana.
   - Elimina la columna original de fecha (ya no sirve).
   
   **La más importante de mencionar:** Que convierte una columna problemática (múltiples formatos) en tres nuevas variables analíticamente útiles. Muestra aprovechamiento de información sin perder datos.

3. **TextNormalizerTransformer**
   - Detecta valores canónicos en columnas de texto (`Campaign_Name`, `Location`, `Device`).
   - Por fuzzy matching, corrige typos y capitalización (`'HYDERABAD'` → `'hyderabad'`).
   - Resultado: categorías limpias sin duplicados por caso/typo.
   
   **La más importante de mencionar:** `Device` pasó de 9 categorías (falsas) a 3 reales. Esto evita que el encoding cree 9 columnas dummy cuando hay solo 3 dispositivos. Un ejemplo concreto que muchos entienden inmediatamente.

4. **DropColumnsTransformer**
   - Elimina `Ad_ID` (identificador, sin valor predictivo).
   - Elimina `Cost` y `Sale_Amount` (son la fuente del target, causarían data leakage).
   - Elimina `Ad_Date` (ya fue reemplazada por las 3 features temporales).
   
   **La más importante de mencionar:** Eliminar `Cost` y `Sale_Amount` por data leakage. Es el punto donde explicas que la limpieza no es solo quitar suciedad, sino también diseño inteligente para evitar que el modelo haga trampa.

5. **MonetaryCleanerTransformer**
   - Convierte strings monetarios (`'$231.88'`, `'$1,892'`) a números flotantes.
   - Quita `$` y comas, luego parsea a `float`.
   
   **La más importante de mencionar:** Sin esta conversión, no podría haber cálculo del target (margen). Es una transformación mínima pero absolutamente bloqueante; sin ella el análisis no avanza.

6. **DropHighMissingTransformer**
   - Revisa si alguna columna tiene > 80% de nulos.
   - Si la hay, la descarta (no sucedió en este dataset, pero es salvaguarda).
   
   **La más importante de mencionar:** Es una decisión de arquitectura. Mostrar que existe un umbral explícito (80%) demuestra que no es magia, sino reglas claras y predecibles.

7. **SmartImputerTransformer**
   - Columnas con < 10% de nulos: imputa con mediana (numéricas) o moda (categóricas).
   - Columnas con 10%-80% de nulos: imputa con mediana (estrategia fallback).
   - Ejemplo: `Conversion Rate` con 24% de nulos se imputa con mediana.
   
   **La más importante de mencionar:** `Conversion Rate` con 24% de nulos (la variable más crítica) se salva por imputación. Sin esto estaría inutilizable. Conecta directamente con el gráfico de Conversion Rate que vas a mostrar después.

8. **ColumnTransformer (rama numérica)**
   
   Esta función aplica tres transformaciones en secuencia a todas las columnas numéricas:
   
   - **OutlierCapper:** Detecta valores extremos (muy altos o muy bajos) y los domestica dentro de rangos razonables. Así evita que valores anómalos distorsionen el modelo.
   
   - **DropZeroVariance:** Elimina columnas que tienen el mismo valor en todos los registros (sin variación). Si una columna siempre dice "3", no le aporta información al modelo para distinguir entre rentable y no rentable.
   
   - **StandardScaler:** Convierte todas las variables numéricas a una escala común (media 0, rango aproximado -3 a 3). Esto es crucial porque variables en diferentes escalas (Clicks en 0-200 vs Impressions en 3000-6000) pueden dominar artificialmente el modelo.
   
   **Flujo:** datos crudos → recorte de extremos → eliminación de columnas sin variación → escalado uniforme → listo para modelado.
   
   **La más importante de mencionar:** StandardScaler. Sin escalado uniforme, el modelo interpretaría Impressions (números grandes: 3000-6000) como más importantes que Clicks (números pequeños: 80-200), simplemente por la magnitud. Eso sería un sesgo artificial, no una decisión inteligente.

9. **ColumnTransformer (rama categórica)**
   - **OneHotEncoder:** Convierte variables categóricas en dummies binarias.
   - Parámetro `handle_unknown='ignore'`: si ve una categoría nueva, la trata como valor desconocido sin error.
   
   **La más importante de mencionar:** El parámetro `handle_unknown='ignore'`. Muestra que el diseño es robusto: el modelo no explota si llegan datos con categorías nuevas. Es production-ready, no frágil.

**Resultado final:** Dataset limpio, sin nulos, sin leakage, listo para entrenar un modelo.

### Concepto Clave: Data Leakage

**¿Qué es el data leakage?**  
Es cuando el modelo de machine learning accede accidentalmente a información que no debería tener durante el entrenamiento. En otras palabras:
- Usas columnas que directamente nos dicen la respuesta (la variable objetivo).
- El modelo "hace trampa" aprendiendo la regla de etiquetado, no patrones reales.
- Resultado: en producción (datos nuevos), el modelo falla porque esa información no está disponible.

**Ejemplo en este proyecto:**  
- `Cost` y `Sale_Amount` son la fuente directa de `Is_Profitable` (margen = (Sale_Amount - Cost) / Cost).
- Si dejamos esas columnas como features, el modelo simplemente diría: "si veo Cost y Sale_Amount, calculo el margen y listo".
- Pero en producción no tendrías esas columnas disponibles, así que el modelo sería inútil.
- **Solución:** Eliminamos `Cost` y `Sale_Amount` del dataset de entrenamiento (las usamos solo para crear el target, luego se descartan).

Lo mismo aplica a `Ad_ID`: es un identificador único que no tiene poder predictivo real, así que se descarta.

### Las más importantes en este caso

Para el dataset de Google Ads, estas funciones fueron **críticas**:

1. **Target Constructor** — Sin ella no hay problem statement. La regla de rentabilidad por margen fue el eje central.
2. **TextNormalizerTransformer** — `Device` tenía 9 variantes, `Location` y `Campaign_Name` tenían typos. Sin esto, el encoding generaría columnas falsas.
3. **MonetaryCleanerTransformer** — `Cost` y `Sale_Amount` eran strings. Sin conversión numérica, ningún análisis funcionaba.
4. **SmartImputerTransformer** — `Conversion Rate` con 24% de nulos habría sido inutilizable. La imputación por mediana lo salvó.
5. **OutlierCapper** — `Conversion Rate` tenía extremos severos. El capping fue esencial para estabilizar la distribución.

Las demás funciones (DateStandardizer, DropHighMissing, DropZeroVariance) fueron complementarias pero necesarias para completitud.

---

## 6. Recomendaciones de uso en diapositivas

- Usa un solo mensaje por grafico (evitar sobrecarga de texto).
- Mantener `outlier_capping.png` fuera del cuerpo principal y mostrarlo solo si piden detalle tecnico.
- Si el tiempo es corto, priorizar: `dist_objetivo.png` + `dist_numericas_por_target.png` (Conversion Rate y Conversions).
- Dejar `boxplot_comparativo.png` como apoyo opcional, aclarando que su valor aqui es de estabilidad mas que de contraste fuerte.
