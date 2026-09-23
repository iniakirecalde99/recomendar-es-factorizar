# Spec — Demo "Recomendar es factorizar"

## 1. Objetivo
Implementar la factorización R ≈ U·Vᵀ de la matriz usuario–película de
MovieLens 100K con dos métodos independientes, ALS y descenso de gradiente, y
mostrar en vivo: la convergencia de ambos, su comparación y recomendaciones
concretas para un usuario. El código tiene que reproducir los ejemplos
numéricos del informe.

## 2. Alcance
Dentro: carga y preprocesamiento de MovieLens 100K; modelo y pérdida; ALS;
descenso de gradiente completo; comparación; recomendaciones; ejemplos chicos
del informe como tests.
Fuera: regularización, descenso estocástico o por mini-lotes, sesgos por
usuario o película, otros datasets, interfaz gráfica, optimización de
rendimiento más allá de numpy vectorizado.

## 3. Datos
- Fuente: MovieLens 100K de GroupLens (943 usuarios, 1682 películas, 100.000
  calificaciones enteras de 1 a 5).
- Script `scripts/descargar_movielens.py`: descarga el zip oficial a `data/`,
  lo descomprime e informa si ya existe. Si no hay red, falla con un mensaje
  claro indicando dónde descargarlo a mano.
- Archivos usados: `u.data` (usuario, película, calificación, timestamp,
  separados por tab) para el dataset completo; `u1.base` y `u1.test` para la
  separación entrenamiento/prueba; `u.item` (codificación latin-1, separado por
  `|`) para los títulos.
- Los ids del archivo empiezan en 1; internamente se reindexan desde 0 y se
  guarda el mapeo para poder mostrar títulos.

## 4. Preprocesamiento
- Construir R (m × n, float, `np.nan` en huecos) y M (m × n, bool).
- Filtro por k: eliminar las películas con menos de k calificaciones en
  entrenamiento, y los usuarios con menos de k calificaciones. Repetir hasta que
  no se elimine nada. Motivo: sin regularización, el paso de ALS de una fila con
  menos de k datos no tiene solución única.
- Reindexar después del filtro. Descartar del conjunto de prueba los pares cuyo
  usuario o película fue filtrado.
- Loguear (WARNING) cuántas películas, usuarios y calificaciones se eliminaron,
  en entrenamiento y en prueba. Ese número va a la sección 8 del informe.
- El umbral del filtro es el mismo k de la factorización, no un parámetro
  aparte: en ALS el sistema de cada fila es de k × k y necesita al menos k
  datos observados.

## 5. Modelo (informe sección 4)
- Estimación: R̂ = U·Vᵀ, con U de m × k y V de n × k; r̂ᵢⱼ = uᵢ · vⱼ.
- Pérdida: SCE(U, V) = Σ_{(i,j) ∈ Ω} (rᵢⱼ − uᵢ · vⱼ)², suma de cuadrados
  del error sin factor 1/2, igual que en el informe (sección 3.5).
- Inicialización: U y V con valores aleatorios uniformes en [0, escala), a
  partir de una semilla. Para comparar, ALS y GD arrancan de la MISMA U y V
  iniciales.

## 6. ALS (informe secciones 3.5 y 5)
Una función `resolver_factor(R, M, F)` que, para cada fila i de R, toma las
filas de F correspondientes a las columnas observadas de esa fila y resuelve
las ecuaciones normales de mínimos cuadrados para obtener la fila i del factor
nuevo.
Algoritmo:
1. Inicializar U, V; calcular f₀.
2. U ← resolver_factor(R, M, V).
3. V ← resolver_factor(R.T, M.T, U).
4. Calcular f(t+1). Si |f(t+1) − f(t)| < epsilon, terminar.
5. Si se alcanzó max_iter, terminar con WARNING. Si no, volver al paso 2.
Si un sistema resulta singular, lanzar `SistemaSingularError` indicando la fila
y la cantidad de datos observados.

## 7. Descenso de gradiente (informe secciones 3.4 y 6)
Con E = M ⊙ (R − U·Vᵀ) (error solo sobre Ω, cero fuera):
- ∇_U f = −2 · E · V
- ∇_V f = −2 · Eᵀ · U
Algoritmo:
1. Inicializar U, V; calcular f₀.
2. Calcular ambos gradientes con U y V de la iteración t.
3. U ← U − eta · ∇_U f;  V ← V − eta · ∇_V f (simultáneo).
4. Calcular f(t+1). Si |f(t+1) − f(t)| < epsilon, terminar.
5. Si f(t+1) > f(t), loguear WARNING (eta posiblemente grande). Si se alcanzó
   max_iter, terminar con WARNING. Si no, volver al paso 2.
Implementar además `descenso_gradiente(f, grad_f, x0, eta, epsilon, max_iter)`
genérico para funciones de ℝⁿ, usado en el ejemplo de 3.4.

## 8. Resultados de un entrenamiento
Cada método devuelve un objeto con: U, V, historial de f por iteración,
cantidad de iteraciones, tiempo total, y el motivo del corte (tolerancia o
max_iter).

## 9. Evaluación y comparación (informe secciones 7 y 8)
- Pérdida sobre Ω de entrenamiento y suma de errores al cuadrado sobre el
  conjunto de prueba, para ambos métodos, con la misma inicialización y el
  mismo k.
- Gráfico: f vs. iteración para ALS y GD en el mismo eje (escala log en y).
- Tabla por consola: iteraciones, tiempo, f final de entrenamiento, error de
  prueba.

## 10. Demo (`python -m src.demo`)
Argumentos: `--metodo {als,gd,ambos}`, `--k`, `--eta`, `--epsilon`,
`--max-iter`, `--semilla`, `--usuario`, `--top-n`.
Salidas:
- Resumen del preprocesamiento (filtrados).
- Progreso por iteración (logging INFO).
- Comparación (sección 9).
- Top-N recomendaciones para `--usuario`: películas no calificadas por él con
  mayor r̂ᵢⱼ, con título.
- Para cada factor latente, las 5 películas con mayor y menor valor en esa
  columna de V (con títulos), sin etiquetarlas: la interpretación la hace el
  grupo en el oral.

## 11. Criterios de aceptación (tests)
- CA-01 Datos: después de cargar `u.data` sin filtrar, R es 943 × 1682 y M
  tiene 100.000 valores True.
- CA-02 Huecos: modificar cualquier valor fuera de Ω (incluido reemplazar NaN
  por un número) no cambia f.
- CA-03 Filtro: después del preprocesamiento, toda fila y columna de M tiene al
  menos k valores True.
- CA-04 Descenso de gradiente genérico sobre f(x,y) = (x−1)² + (y−2)² +
  (x+y−6)², desde (0,0), con eta = 0,1 y epsilon = 1e-10: converge a (2, 3)
  con f = 3 (tolerancia 1e-4). [La tabla 2 del informe se genera con esta
  función; el punto inicial y los parámetros tienen que coincidir.]
- CA-05 Simetría de ALS: resolver_factor(R.T, M.T, U) da exactamente el mismo
  resultado que calcular el paso de V fila por fila a mano, sin transponer.
- CA-06 Monotonía de ALS: el historial de f es no creciente.
- CA-07 Gradiente: ∇_U f y ∇_V f coinciden con una aproximación por diferencias
  finitas en una matriz chica (error relativo < 1e-5).
- CA-08 Simultaneidad de GD: una iteración de GD es igual a calcular ambos
  gradientes con (U, V) de la iteración t y actualizar; test contra una
  implementación de referencia en el propio test.
- CA-09 Ejemplo de 4 × 5 del informe con k = 2:
      Ana:   5  4  ?  1  ?
      Bruno: ?  5  3  ?  1
      Carla: 1  ?  3  5  ?
      Diego: ?  1  ?  4  5
  (columnas: Toy Story, Star Wars, Fargo, Titanic, Scream). ALS no lanza
  SistemaSingularError y f decrece. Esta parte del criterio ya es exigible.
  [PENDIENTE: fijar la V₀ del ejemplo de la sección 5 del informe; hasta
  entonces, un test adicional que compare la primera iteración contra la
  tabla del informe queda marcado `skip` con el motivo escrito.]
- CA-10 Reproducibilidad: misma semilla, mismos resultados.
- CA-11 Recomendaciones: el top-N nunca incluye películas ya calificadas por el
  usuario.
- CA-12 Sin librerías prohibidas: ningún archivo de `src/` importa Surprise,
  implicit ni scikit-learn, ni usa inv, lstsq o pinv.

## 12. Decisiones pendientes
- V₀ del ejemplo de ALS a mano (sección 5 del informe).
- Valores por defecto de k, eta, epsilon y max_iter para MovieLens: se
  calibran en una tarea específica y se documentan en `config.py` con el
  criterio usado.