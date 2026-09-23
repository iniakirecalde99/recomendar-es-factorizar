# Plan — Demo "Recomendar es factorizar"

Diseño derivado de `CLAUDE.md` y `specs/spec.md`. No incluye código de
implementación: solo estructura, firmas, docstrings y contenido literal de
`config.py` y `errores.py` (constantes y clases de excepción, no algoritmos).

## 0. Decisiones incorporadas en esta versión

Recibidas del usuario, ya reflejadas abajo (no se vuelven a plantear como
preguntas):

1. **Sin factor ½.** La pérdida es la SCE (suma de cuadrados del error, informe
   3.5), sin ½. La función se llama `sce` en el código.
2. **RMSE afuera.** No hay `calcular_rmse_en_prueba` ni flag `--rmse`. La
   comparación en la sección 9 se hace con SCE sobre Ω del conjunto completo
   filtrado, igual para ambos métodos.
3. **V₀ del ejemplo de ALS sigue pendiente.** CA-09 solo verifica que no haya
   `SistemaSingularError` y que la SCE decrezca. La comparación contra la
   tabla del informe queda como test `skip` con el motivo escrito.
4. **Defaults de MovieLens: se eligen corriendo la demo a mano.** No hay
   script de calibración (`scripts/calibrar.py` queda fuera de alcance).
   `config.py` deja los valores marcados como pendientes hasta correr la
   demo a mano; el criterio usado para elegirlos queda como comentario en
   `config.py`, no en un script aparte.
5. **`--k` como umbral y como dimensión latente es intencional.** El sistema
   de cada fila en ALS es k×k y necesita al menos k datos observados para
   tener solución única; el umbral de filtro es k por construcción, no un
   parámetro independiente.
6. **`inicializar_factores` vive en `modelo.py`, compartida.** CLAUDE.md regla
   3 ya está ampliada para incluir la inicialización de U y V entre lo que
   comparten ALS y descenso de gradiente.
7. **`TOP_N_DEFECTO = 10`.**
8. **Tabla por consola: texto plano con f-strings, sin dependencias nuevas**
   (nada de pandas ni tabulate). Columnas: método, iteraciones, motivo de
   corte, tiempo (s), SCE final. Sin partición entrenamiento/prueba (ver
   decisión 2): una sola SCE por método, sobre Ω del conjunto completo
   filtrado. Los decimales usan punto en el código; el pasaje a coma para
   pegar en el informe es manual, fuera del código.

## 1. Estructura de carpetas

```
recomendar-es-factorizar/
├── CLAUDE.md
├── data/                        # .gitignore — dataset descargado, no se commitea
├── scripts/
│   └── descargar_movielens.py
├── specs/
│   ├── spec.md
│   ├── plan.md
│   └── tasks.md                 # (a crear en una tarea posterior)
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── errores.py
│   ├── datos.py
│   ├── modelo.py
│   ├── als.py
│   ├── gradiente.py
│   ├── comparacion.py
│   ├── recomendaciones.py
│   └── demo.py
└── tests/
    ├── conftest.py
    ├── test_datos.py
    ├── test_modelo.py
    ├── test_gradiente_generico.py
    ├── test_als.py
    ├── test_gd_factorizacion.py
    ├── test_reproducibilidad.py
    ├── test_recomendaciones.py
    └── test_dependencias.py
```

Justificación de la separación de módulos: CLAUDE.md regla 3 exige que ALS y
descenso de gradiente vivan en módulos separados (`als.py`, `gradiente.py`) y
que solo compartan carga de datos (`datos.py`), inicialización de U y V,
predicción y SCE (`modelo.py`). `comparacion.py` y `recomendaciones.py`
corresponden a las secciones 9 y 10 del informe/spec, posteriores a ambos
métodos y no son "lógica de actualización", por lo que no violan la regla.

## 2. Módulos y funciones públicas

### `src/config.py`
No define funciones: ver contenido en la sección 3.

### `src/errores.py`
No define funciones: ver contenido en la sección 4.

### `src/datos.py` — carga y preprocesamiento (spec §3–4)

```python
@dataclass
class ConjuntoCalificaciones:
    R: np.ndarray                       # m × n, float, NaN en huecos
    M: np.ndarray                       # m × n, bool
    id_usuario_a_indice: dict[int, int]     # id crudo del archivo -> índice 0-based
    id_pelicula_a_indice: dict[int, int]
```

- `cargar_calificaciones(ruta_archivo: Path) -> ConjuntoCalificaciones`
  Carga un archivo estilo `u.data` (usuario, película, calificación,
  timestamp separados por tab), reindexa ids de usuario y película desde 0 y
  arma R (NaN en huecos) y M. No aplica ningún filtro.
  **Cubre:** CA-01.

- `filtrar_por_minimo(datos: ConjuntoCalificaciones, k: int) -> ConjuntoCalificaciones`
  Elimina iterativamente filas y columnas de M con menos de k valores True
  hasta alcanzar un punto fijo, reindexa el resultado desde 0 y devuelve
  mapeos actualizados (solo con los ids que sobrevivieron). k es el mismo
  valor que la dimensión latente del modelo (ver decisión 5): el umbral no es
  arbitrario, es el mínimo de datos por fila que necesita el sistema k×k de
  ALS para tener solución única. Loguea (WARNING) cuántas películas, usuarios
  y calificaciones se eliminaron. Lanza `ErrorDatosInsuficientes` si no queda
  ninguna fila o columna.
  **Cubre:** CA-03.

- `cargar_titulos(ruta_u_item: Path) -> dict[int, str]`
  Carga `u.item` (codificación latin-1, separado por `|`) y devuelve el
  mapeo id de película crudo → título.

- `construir_indice_a_titulo(id_pelicula_a_indice: dict[int, int], titulos_por_id: dict[int, str]) -> dict[int, str]`
  Invierte `id_pelicula_a_indice` y lo combina con `titulos_por_id` para
  obtener índice de columna de V → título.

- `preparar_datos_movielens(ruta_u_data: Path, ruta_u_item: Path, k: int) -> DatosPreparados`
  Orquesta `cargar_calificaciones`, `filtrar_por_minimo`, `cargar_titulos` y
  `construir_indice_a_titulo`; pensada para ser el único punto de entrada que
  usa `demo.py`. `DatosPreparados` es un dataclass con: `calificaciones:
  ConjuntoCalificaciones`, `titulos_por_indice: dict[int, str]`.

### `src/modelo.py` — módulo compartido (spec §5, §8; CLAUDE.md regla 3)

- `predecir(U: np.ndarray, V: np.ndarray) -> np.ndarray`
  Calcula la estimación R̂ = U·Vᵀ (informe sección 4).

- `sce(R: np.ndarray, M: np.ndarray, U: np.ndarray, V: np.ndarray) -> float`
  Calcula SCE(U,V) = Σ_{(i,j)∈Ω} (rᵢⱼ − uᵢ·vⱼ)², suma de cuadrados del error
  sin factor ½ (informe 3.5), sumando solo sobre las posiciones donde M es
  True. En las condiciones de corte y en los historiales de ALS/GD se sigue
  llamando f al valor que devuelve esta función (notación de CLAUDE.md regla
  7).
  **Cubre:** CA-02.

- `inicializar_factores(m: int, n: int, k: int, escala: float, generador: np.random.Generator) -> tuple[np.ndarray, np.ndarray]`
  Genera U (m×k) y V (n×k) con valores aleatorios uniformes en [0, escala)
  usando el `Generator` recibido (informe sección 4). Función compartida:
  quien orqueste la comparación (`demo.py`) la llama una sola vez y pasa el
  mismo par (o copias) a `entrenar_als` y `entrenar_gd`, para que ambos
  arranquen literalmente de la misma U₀, V₀ (spec §5).

```python
@dataclass
class ResultadoEntrenamiento:
    U: np.ndarray
    V: np.ndarray
    historial_f: list[float]
    n_iteraciones: int
    tiempo_segundos: float
    motivo_corte: str   # "tolerancia" | "max_iter"
```
(spec §8, usado por `als.py` y `gradiente.py`)

### `src/als.py` (spec §6, informe 3.5 y 5)

- `resolver_factor(R: np.ndarray, M: np.ndarray, F: np.ndarray) -> np.ndarray`
  Para cada fila i de R, toma las columnas j con `M[i, j] = True`, arma las
  ecuaciones normales de mínimos cuadrados con las filas correspondientes de
  F y resuelve con `np.linalg.solve` la fila i del nuevo factor. Llamada con
  `(R, M, V)` calcula el paso de U; llamada con `(R.T, M.T, U)` calcula el
  paso de V (misma función, sin duplicar lógica). Lanza
  `SistemaSingularError(fila, n_observados)` si el sistema es singular.
  **Cubre:** CA-05, CA-06, CA-09.

- `entrenar_als(R: np.ndarray, M: np.ndarray, U0: np.ndarray, V0: np.ndarray, epsilon: float, max_iter: int) -> ResultadoEntrenamiento`
  Ejecuta ALS (informe 3.5 y 5) alternando `resolver_factor(R, M, V)` y
  `resolver_factor(R.T, M.T, U)` a partir de `U0`, `V0`, usando `modelo.sce`
  para calcular f en cada iteración, hasta `|f(t+1) − f(t)| < epsilon` o
  `max_iter`. Loguea INFO por iteración y WARNING si corta por `max_iter`.
  **Cubre:** CA-06, CA-09 (a nivel algoritmo); soporta CA-10.

### `src/gradiente.py` (spec §7, informe 3.4 y 6)

```python
@dataclass
class ResultadoDescensoGenerico:
    x: np.ndarray
    historial_f: list[float]
    n_iteraciones: int
    motivo_corte: str
```

- `descenso_gradiente(f: Callable[[np.ndarray], float], grad_f: Callable[[np.ndarray], np.ndarray], x0: np.ndarray, eta: float, epsilon: float, max_iter: int) -> ResultadoDescensoGenerico`
  Descenso de gradiente genérico para funciones de ℝⁿ (informe 3.4): actualiza
  `x ← x − eta · grad_f(x)` hasta `|f(t+1) − f(t)| < epsilon` o `max_iter`.
  Loguea WARNING si f aumenta entre iteraciones o si se alcanza `max_iter`.
  **Cubre:** CA-04.

- `gradiente_sce(R: np.ndarray, M: np.ndarray, U: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]`
  Calcula ∇_U f = −2·E·V y ∇_V f = −2·Eᵀ·U con E = M ⊙ (R − U·Vᵀ) (informe
  3.4/6), ambos evaluados en el mismo (U, V) recibido (gradiente de `sce`,
  consistente con la ausencia de factor ½).
  **Cubre:** CA-07, CA-08.

- `entrenar_gd(R: np.ndarray, M: np.ndarray, U0: np.ndarray, V0: np.ndarray, eta: float, epsilon: float, max_iter: int) -> ResultadoEntrenamiento`
  Descenso de gradiente completo sobre la factorización (informe 3.4 y 6): en
  cada iteración calcula `gradiente_sce` con (U, V) de la iteración t y
  actualiza ambos simultáneamente, usando `modelo.sce` para el criterio de
  corte. Loguea INFO por iteración, WARNING si f aumenta o si corta por
  `max_iter`.
  **Cubre:** CA-08 (a nivel algoritmo); soporta CA-10.

### `src/comparacion.py` (spec §9)

```python
@dataclass
class FilaComparacion:
    metodo: str
    n_iteraciones: int
    motivo_corte: str
    tiempo_segundos: float
    sce_final: float
```

- `comparar_metodos(resultado_als: ResultadoEntrenamiento, resultado_gd: ResultadoEntrenamiento) -> list[FilaComparacion]`
  Arma una fila por método con iteraciones, motivo de corte, tiempo y SCE
  final (último valor de `historial_f`). Sin RMSE (decisión 2) y sin
  partición entrenamiento/prueba: la SCE es la del conjunto completo
  filtrado, la misma que ya calculó `entrenar_als`/`entrenar_gd` sobre Ω.

- `formatear_tabla_comparacion(filas: list[FilaComparacion]) -> str`
  Arma la tabla de la sección 9 como texto plano alineado con f-strings, sin
  agregar dependencias (nada de pandas ni tabulate — decisión 8). Columnas:
  método, iteraciones, motivo de corte, tiempo (s), SCE final. Decimales con
  punto; la conversión a coma para pegar en el informe es manual, fuera del
  código. Devuelve el texto listo para imprimir — `demo.py` es quien hace el
  `print` (único módulo con `print`).

- `graficar_convergencia(resultado_als: ResultadoEntrenamiento, resultado_gd: ResultadoEntrenamiento, ruta_salida: Path | None) -> None`
  Grafica f (SCE) vs. iteración para ALS y GD en el mismo eje, escala
  logarítmica en y (matplotlib); si `ruta_salida` no es None, guarda la
  figura ahí.

### `src/recomendaciones.py` (spec §10)

- `recomendar_top_n(U: np.ndarray, V: np.ndarray, M: np.ndarray, indice_usuario: int, n: int, titulos_por_indice: dict[int, str]) -> list[tuple[str, float]]`
  Entre las películas j con `M[indice_usuario, j] = False`, devuelve las n
  con mayor r̂ᵢⱼ, junto con su título.
  **Cubre:** CA-11.

- `extremos_por_factor(V: np.ndarray, titulos_por_indice: dict[int, str], cantidad: int = 5) -> list[tuple[int, list[tuple[str, float]], list[tuple[str, float]]]]`
  Para cada columna (factor latente) de V, devuelve las `cantidad` películas
  con mayor y con menor valor en esa columna, con título, sin interpretar ni
  etiquetar el factor.

### `src/demo.py` (spec §10) — único módulo con `print`

- `construir_parser() -> argparse.ArgumentParser`
  Define los argumentos `--datos`, `--grafico`, `--k`, `--eta`, `--epsilon`,
  `--max-iter`, `--semilla`, `--usuario`, `--top-n`, con los defaults de
  `config.py` (`RUTA_DATOS_DEFECTO`, `RUTA_GRAFICO_DEFECTO`, `K_DEFECTO`,
  `ETA_DEFECTO`, `EPSILON_DEFECTO`, `MAX_ITER_DEFECTO`, `SEMILLA_DEFECTO`,
  `USUARIO_DEFECTO`, `TOP_N_DEFECTO`). Sin `--rmse` (decisión 2) ni
  `--metodo`: la demo siempre corre ALS y GD, para poder compararlos.

- `_traducir_usuario(id_usuario_a_indice: dict[int, int], usuario_id: int) -> int`
  Traduce el id crudo de `--usuario` a índice de fila. Lanza
  `UsuarioNoEncontradoError` si `usuario_id` no está en el mapeo (no existe
  en MovieLens, o el filtro por `--k` lo eliminó).

- `_formatear_top_n_lado_a_lado(recomendaciones_als: list[tuple[str, float]], recomendaciones_gd: list[tuple[str, float]]) -> str`
  Arma las recomendaciones de ALS y GD como dos columnas de texto plano,
  lado a lado.

- `main(argv: list[str] | None = None) -> None`
  Reconfigura `sys.stdout` a UTF-8 al arrancar (`sys.stdout.reconfigure`,
  por consolas Windows con codepage heredado). Orquesta
  `preparar_datos_movielens`, `_traducir_usuario`, `inicializar_factores`
  (una sola vez, U0/V0 compartidos), siempre `entrenar_als` y `entrenar_gd`,
  `comparar_metodos` + `formatear_tabla_comparacion`, `graficar_convergencia`
  (a `--grafico`), `recomendar_top_n` de ambos métodos (impresas lado a
  lado con `_formatear_top_n_lado_a_lado`) y `extremos_por_factor` solo del
  V de ALS (aclarado en la salida: GD no se usa para esta parte). Si
  `entrenar_gd` lanza `DivergenciaError`, la captura y muestra solo el
  mensaje, sin traceback. Configura logging (INFO por defecto).

### `scripts/descargar_movielens.py` (spec §3)

- `descargar_movielens(directorio_destino: Path, url: str = URL_MOVIELENS_100K) -> Path`
  Descarga el zip de MovieLens 100K a `directorio_destino` y lo descomprime;
  si ya existe, lo informa y no descarga de nuevo. Si la descarga falla (sin
  red), lanza `ErrorDescargaDataset` con un mensaje que indica la URL para
  descargarlo a mano.

- `main(argv: list[str] | None = None) -> None`
  CLI mínima que llama a `descargar_movielens` con `data/` como destino.

## 3. Contenido de `src/config.py`

Solo constantes (no lógica). k, eta, epsilon, max_iter y escala de
inicialización para MovieLens quedan pendientes hasta correr la demo a mano
y elegirlos (decisión 4); no hay script de calibración aparte, el criterio
usado queda como comentario en este archivo. `TOP_N_DEFECTO` ya tiene valor
(decisión 7). `RMSE_DEFECTO` se elimina (decisión 2). `USUARIO_DEFECTO`,
`RUTA_DATOS_DEFECTO` y `RUTA_GRAFICO_DEFECTO` se agregaron con T14, para
`--usuario`, `--datos` y `--grafico` de `demo.py`.

```python
"""Valores por defecto de los hiperparámetros (algunos pendientes hasta correr la demo a mano — ver decisión 4)."""

# --- Preprocesamiento y modelo ---
K_DEFECTO: int = ...          # PENDIENTE (se completa corriendo la demo a mano) — umbral de filtro (§4) y dimensión latente (§5): mismo valor por diseño, ver decisión 5
ETA_DEFECTO: float = ...      # PENDIENTE (se completa corriendo la demo a mano)
EPSILON_DEFECTO: float = ...  # PENDIENTE (se completa corriendo la demo a mano)
MAX_ITER_DEFECTO: int = ...   # PENDIENTE (se completa corriendo la demo a mano)
SEMILLA_DEFECTO: int = ...    # PENDIENTE (se completa corriendo la demo a mano)
ESCALA_INICIALIZACION_DEFECTO: float = ...  # PENDIENTE (se completa corriendo la demo a mano)

# --- Demo ---
TOP_N_DEFECTO: int = 10       # decisión 7: lo que entra cómodo en pantalla durante la demo
USUARIO_DEFECTO: int = 1      # id crudo de MovieLens (no índice); si --k lo filtra, UsuarioNoEncontradoError
RUTA_DATOS_DEFECTO: Path = Path("data/ml-100k")   # misma carpeta que scripts/descargar_movielens.py
RUTA_GRAFICO_DEFECTO: Path = Path("convergencia.png")
```

## 4. Contenido de `src/errores.py`

```python
class ErrorFactorizacion(Exception):
    """Excepción base del paquete; nunca se instancia directamente."""


class SistemaSingularError(ErrorFactorizacion):
    """Ecuaciones normales singulares al resolver una fila en ALS.

    Atributos: fila (índice de la fila) y n_observados (cantidad de
    columnas observadas usadas para armar el sistema).
    """

    def __init__(self, fila: int, n_observados: int) -> None: ...


class ErrorDatosInsuficientes(ErrorFactorizacion):
    """El filtrado por k dejó una matriz sin filas o sin columnas."""

    def __init__(self, k: int) -> None: ...


class ErrorDescargaDataset(ErrorFactorizacion):
    """La descarga de MovieLens falló (p. ej. sin red).

    El mensaje incluye la URL para descargarlo a mano.
    """

    def __init__(self, url: str, causa: Exception | None = None) -> None: ...


class DivergenciaError(ErrorFactorizacion):
    """El descenso de gradiente divergió: f dejó de ser finito (inf o NaN).

    Atributos: iteracion (1-indexada) y eta usado. `demo.py` la captura y
    muestra solo el mensaje, sin traceback.
    """

    def __init__(self, iteracion: int, eta: float) -> None: ...


class UsuarioNoEncontradoError(ErrorFactorizacion):
    """El id crudo de `--usuario` no está en el conjunto filtrado.

    Atributo: usuario_id (el id crudo pedido).
    """

    def __init__(self, usuario_id: int) -> None: ...
```

## 5. Tests y criterios de aceptación cubiertos

`tests/conftest.py`: fixtures compartidas —
`generador_fijo` (Generator con semilla fija),
`matriz_ejemplo_informe` (la matriz Ana/Bruno/Carla/Diego de spec §11, CA-09),
`matriz_pequena_aleatoria` (para el chequeo de gradiente por diferencias
finitas, CA-07).

| Archivo | Test | CA |
|---|---|---|
| `test_datos.py` | `test_cargar_calificaciones_dimensiones_y_mascara` | CA-01 |
| `test_datos.py` | `test_filtrar_por_minimo_todas_las_filas_y_columnas_tienen_al_menos_k` | CA-03 |
| `test_modelo.py` | `test_sce_ignora_valores_fuera_de_omega` | CA-02 |
| `test_gradiente_generico.py` | `test_descenso_gradiente_converge_al_minimo_conocido` | CA-04 |
| `test_als.py` | `test_resolver_factor_transpuesto_coincide_con_calculo_manual_de_v` | CA-05 |
| `test_als.py` | `test_historial_de_sce_de_als_no_crece` | CA-06 |
| `test_als.py` | `test_als_ejemplo_4x5_no_lanza_singular_y_sce_decrece` | CA-09 |
| `test_als.py` | `test_als_ejemplo_4x5_primera_iteracion_coincide_con_informe` (`skip`, motivo: "V₀ del ejemplo de la sección 5 del informe todavía no está fijado") | CA-09 (parcial) |
| `test_gd_factorizacion.py` | `test_gradiente_sce_coincide_con_diferencias_finitas` | CA-07 |
| `test_gd_factorizacion.py` | `test_iteracion_gd_es_simultanea_contra_referencia` | CA-08 |
| `test_reproducibilidad.py` | `test_misma_semilla_misma_ejecucion_als` | CA-10 |
| `test_reproducibilidad.py` | `test_misma_semilla_misma_ejecucion_gd` | CA-10 |
| `test_recomendaciones.py` | `test_top_n_excluye_peliculas_ya_calificadas` | CA-11 |
| `test_dependencias.py` | `test_src_no_importa_librerias_prohibidas` | CA-12 |
| `test_dependencias.py` | `test_src_no_usa_inv_lstsq_pinv` | CA-12 |

`test_dependencias.py` no importa `src` para ejecutar código: recorre los
archivos `.py` de `src/` con `ast` y busca imports (`Surprise`, `implicit`,
`sklearn`) y atributos (`np.linalg.inv`, `np.linalg.lstsq`, `np.linalg.pinv`)
prohibidos, para que el test siga siendo válido aunque el resto falle en
importar.

## 6. Preguntas

Ninguna pendiente: las preguntas originales quedaron resueltas y volcadas en
la sección 0 (decisiones 1–8). Lo único que sigue abierto en la spec misma es
lo que la decisión 3 y la decisión 4 dejan explícitamente para después: V₀
del ejemplo de ALS y los valores numéricos de `config.py` (sección 12 de
`spec.md`).
