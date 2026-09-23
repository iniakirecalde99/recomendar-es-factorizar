# Tareas — Demo "Recomendar es factorizar"

Derivado de `specs/plan.md`. Cada tarea sigue CLAUDE.md: primero el test que
se lista, después el código mínimo para que pase, después `pytest -q`
completo. Una tarea no está terminada si algún test falla (incluidas las de
tareas anteriores). Ninguna tarea toca más de dos módulos de `src/`.

Orden: guarda transversal → datos y preprocesamiento → modelo → descenso de
gradiente genérico → ALS → descenso de gradiente matricial → calibración de
valores por defecto → comparación → recomendaciones → demo.

Todo test que necesite el dataset real de MovieLens (al menos el de CA-01) se
marca `@pytest.mark.movielens` y se saltea con motivo si `data/ml-100k/` no
existe (ver T00). El resto de los tests usa fixtures chicas en `tests/`, sin
red ni dataset real.

## 0. Guarda transversal

### T00 — Configuración de pytest: marker `movielens`
**Objetivo:** registrar el marker `movielens` y saltear automáticamente,
con motivo, los tests que lo usan cuando `data/ml-100k/` no existe.
**Archivos:** `pytest.ini` (registro del marker), `tests/conftest.py`
(`pytest_collection_modifyitems` que aplica el skip; no es `src/`).
**Cierra:** ninguno (setup transversal).
**Test que se escribe primero:** no hay test unitario nuevo — se verifica con
`pytest --markers` (aparece `movielens` registrado, sin warning) y con una
corrida de `pytest -q` en un checkout sin `data/ml-100k/`, donde los tests
marcados `movielens` (T03) quedan en `skipped` con el motivo, no en `error`
ni `failed`.

### T01 — Test estático anti-dependencias prohibidas
**Objetivo:** tener desde el arranque un test que falle si algún archivo de
`src/` importa Surprise/implicit/scikit-learn o usa `inv`/`lstsq`/`pinv`.
**Archivos:** ninguno en `src/` (solo `tests/test_dependencias.py`).
**Cierra:** CA-12.
**Test que se escribe primero:** es la tarea completa — `pytest` recorre
`src/` con `ast` y debe pasar vacío (sin archivos) y seguir pasando en cada
tarea siguiente. `test_src_no_importa_librerias_prohibidas`,
`test_src_no_usa_inv_lstsq_pinv`.

## 1. Datos y preprocesamiento

### T02 — Script de descarga de MovieLens
**Objetivo:** descargar y descomprimir el zip de MovieLens 100K a `data/`,
informando si ya existe y fallando con mensaje claro sin red.
**Archivos:** `scripts/descargar_movielens.py`, `src/errores.py`
(`ErrorDescargaDataset`).
**Cierra:** ninguno (soporta spec §3, no tiene CA propio).
**Test que se escribe primero:** `tests/test_descargar_movielens.py::test_descargar_movielens_informa_si_ya_existe`
y `test_descargar_movielens_falla_con_mensaje_y_url_si_no_hay_red` (con la
descarga mockeada, sin red real).

### T03 — Carga de calificaciones crudas
**Objetivo:** cargar un archivo estilo `u.data`, reindexar ids desde 0 y
armar `R` (NaN en huecos) y `M`.
**Archivos:** `src/datos.py` (`ConjuntoCalificaciones`, `cargar_calificaciones`).
**Cierra:** CA-01.
**Test que se escribe primero:**
`test_datos.py::test_cargar_calificaciones_reindexa_ids_y_arma_mascara`
(comportamiento general, con un `u.data` de prueba chico en `tests/`, sin
marker, corre siempre); y
`test_datos.py::test_cargar_calificaciones_dimensiones_y_mascara` (CA-01
propiamente dicho: R es 943×1682 y M tiene 100.000 True), marcado
`@pytest.mark.movielens`, sobre `data/ml-100k/u.data` real — se saltea con
motivo si `data/ml-100k/` no existe (ver T00).

### T04 — Filtro iterativo por mínimo k
**Objetivo:** eliminar filas/columnas con menos de k calificaciones hasta
punto fijo, reindexar y actualizar mapeos.
**Archivos:** `src/datos.py` (`filtrar_por_minimo`), `src/errores.py`
(`ErrorDatosInsuficientes`).
**Cierra:** CA-03.
**Test que se escribe primero:** `test_datos.py::test_filtrar_por_minimo_todas_las_filas_y_columnas_tienen_al_menos_k`
(y un test adicional para `ErrorDatosInsuficientes` cuando el filtro vacía la
matriz).

### T05 — Partición train/test y filtrado del conjunto de prueba
**Objetivo:** cargar `u1.base`/`u1.test` y reindexar la prueba con el mismo
mapeo del entrenamiento filtrado, descartando lo que no está en el mapeo.
**Archivos:** `src/datos.py` (`cargar_particion`, `aplicar_filtro_a_particion`).
**Cierra:** ninguno (soporta spec §4 y §9; ver decisión 7 de `plan.md`).
**Test que se escribe primero:** `test_datos.py::test_aplicar_filtro_a_particion_descarta_pares_fuera_del_mapeo`
(incluye un caso de película presente en `u1.test` pero ausente de
`u1.base`).

### T06 — Títulos
**Objetivo:** cargar `u.item` y construir el mapeo índice de columna → título.
**Archivos:** `src/datos.py` (`cargar_titulos`, `construir_indice_a_titulo`).
**Cierra:** ninguno (soporta spec §10).
**Test que se escribe primero:** `test_datos.py::test_cargar_titulos_y_construir_indice_a_titulo`
(con un `u.item` de prueba chico en latin-1).

### T07 — Orquestación de datos
**Objetivo:** encadenar carga, filtro y títulos en un único punto de entrada
para `demo.py`.
**Archivos:** `src/datos.py` (`preparar_datos_movielens`, `DatosPreparados`).
**Cierra:** ninguno.
**Test que se escribe primero:** `test_datos.py::test_preparar_datos_movielens_integra_carga_filtro_y_titulos`.

## 2. Modelo

### T08 — Predicción y SCE
**Objetivo:** calcular R̂ = U·Vᵀ y la SCE sobre Ω, sin factor 1/2.
**Archivos:** `src/modelo.py` (`predecir`, `sce`).
**Cierra:** CA-02.
**Test que se escribe primero:** `test_modelo.py::test_sce_ignora_valores_fuera_de_omega`
(incluye reemplazar un NaN fuera de Ω por un número y verificar que `sce` no
cambia).

### T09 — Inicialización compartida
**Objetivo:** generar U₀, V₀ uniformes en [0, escala) a partir de un
`Generator`, y definir `ResultadoEntrenamiento`.
**Archivos:** `src/modelo.py` (`inicializar_factores`, `ResultadoEntrenamiento`).
**Cierra:** ninguno (building block de CA-10; ver decisión 6 de `plan.md`).
**Test que se escribe primero:** `test_modelo.py::test_inicializar_factores_reproducible_con_la_misma_semilla`.

## 3. Descenso de gradiente genérico

### T10 — `descenso_gradiente` genérico
**Objetivo:** descenso de gradiente completo para funciones de ℝⁿ, con
criterio de corte por tolerancia o `max_iter`.
**Archivos:** `src/gradiente.py` (`descenso_gradiente`, `ResultadoDescensoGenerico`).
**Cierra:** CA-04.
**Test que se escribe primero:** `test_gradiente_generico.py::test_descenso_gradiente_converge_al_minimo_conocido`
(f(x,y) = (x−1)² + (y−2)² + (x+y−6)², desde (0,0), eta=0.1, epsilon=1e-10).

## 4. ALS

### T11 — `resolver_factor`
**Objetivo:** resolver por fila las ecuaciones normales de mínimos cuadrados
con `np.linalg.solve`, lanzando `SistemaSingularError` si corresponde.
**Archivos:** `src/als.py` (`resolver_factor`), `src/errores.py`
(`SistemaSingularError`).
**Cierra:** CA-05.
**Test que se escribe primero:** `test_als.py::test_resolver_factor_transpuesto_coincide_con_calculo_manual_de_v`
(y un test del caso singular, sin CA propio).

### T12 — `entrenar_als`
**Objetivo:** alternar el paso de U y V hasta convergencia o `max_iter`,
devolviendo `ResultadoEntrenamiento`.
**Archivos:** `src/als.py` (`entrenar_als`).
**Cierra:** CA-06, CA-09 (parte activa).
**Test que se escribe primero:** `test_als.py::test_historial_de_sce_de_als_no_crece`,
`test_als.py::test_als_ejemplo_4x5_no_lanza_singular_y_sce_decrece`, y
`test_als.py::test_als_ejemplo_4x5_primera_iteracion_coincide_con_informe`
marcado `skip` (motivo: V₀ del ejemplo todavía no está fijado en la spec).

## 5. Descenso de gradiente matricial

### T13 — `gradiente_sce`
**Objetivo:** calcular ∇_U f y ∇_V f sobre el mismo (U, V), sin factor 1/2.
**Archivos:** `src/gradiente.py` (`gradiente_sce`).
**Cierra:** CA-07.
**Test que se escribe primero:** `test_gd_factorizacion.py::test_gradiente_sce_coincide_con_diferencias_finitas`.

### T14 — `entrenar_gd`
**Objetivo:** descenso de gradiente completo sobre la factorización,
actualizando U y V simultáneamente con los valores de la iteración t.
**Archivos:** `src/gradiente.py` (`entrenar_gd`).
**Cierra:** CA-08.
**Test que se escribe primero:** `test_gd_factorizacion.py::test_iteracion_gd_es_simultanea_contra_referencia`.

### T15 — Reproducibilidad de ALS y GD
**Objetivo:** verificar que, con la misma semilla, ALS y GD dan exactamente
los mismos resultados.
**Archivos:** ninguno nuevo en `src/` (test que ejercita `modelo.py`,
`als.py` y `gradiente.py` ya existentes).
**Cierra:** CA-10.
**Test que se escribe primero:** `test_reproducibilidad.py::test_misma_semilla_misma_ejecucion_als`,
`test_reproducibilidad.py::test_misma_semilla_misma_ejecucion_gd`.

## 6. Calibración de valores por defecto

### T16 — Script de calibración y defaults en `config.py`
**Objetivo:** `scripts/calibrar.py` corre ALS y GD sobre MovieLens real
probando candidatos de k, eta, epsilon, max_iter, semilla y escala, y deja
los valores elegidos en `config.py` con un comentario del criterio usado.
**Archivos:** `scripts/calibrar.py`, `src/config.py`.
**Cierra:** ninguno (condición previa para T22/T23 — la demo necesita estos
defaults completos, ver decisión 4 de `plan.md`).
**Test que se escribe primero:** `test_config.py::test_defaults_de_movielens_tienen_tipos_y_rangos_validos`
— sin red ni dataset: valida que `K_DEFECTO` sea `int > 0`, `ETA_DEFECTO` y
`EPSILON_DEFECTO` sean `float > 0`, `MAX_ITER_DEFECTO` sea `int > 0`,
`SEMILLA_DEFECTO` sea `int`, y `ESCALA_INICIALIZACION_DEFECTO` sea
`float > 0`. No valida los valores exactos ni corre nada sobre el dataset.
`scripts/calibrar.py` en sí es una herramienta manual de calibración (como
`scripts/descargar_movielens.py`, corre sobre datos reales); no lleva un test
automático propio en esta tarea, solo se valida su resultado en `config.py`.

## 7. Comparación

### T17 — SCE sobre el conjunto de prueba
**Objetivo:** calcular la SCE de R̂ sobre los pares reindexados de prueba.
**Archivos:** `src/comparacion.py` (`sce_en_prueba`).
**Cierra:** ninguno (soporta spec §9).
**Test que se escribe primero:** `test_comparacion.py::test_sce_en_prueba_suma_errores_al_cuadrado_de_los_pares`.

### T18 — Tabla de comparación
**Objetivo:** armar una fila por método y formatearlas como texto plano con
f-strings, sin dependencias nuevas.
**Archivos:** `src/comparacion.py` (`comparar_metodos`, `FilaComparacion`,
`formatear_tabla_comparacion`).
**Cierra:** ninguno (soporta spec §9; ver decisión 9 de `plan.md`).
**Test que se escribe primero:** `test_comparacion.py::test_comparar_metodos_arma_una_fila_por_metodo`,
`test_comparacion.py::test_formatear_tabla_comparacion_incluye_las_columnas_esperadas`.

### T19 — Gráfico de convergencia
**Objetivo:** graficar f vs. iteración para ALS y GD en el mismo eje, escala
log en y.
**Archivos:** `src/comparacion.py` (`graficar_convergencia`).
**Cierra:** ninguno (soporta spec §9).
**Test que se escribe primero:** `test_comparacion.py::test_graficar_convergencia_guarda_archivo_en_ruta_dada`
(solo verifica que el archivo se crea, no el contenido visual).

## 8. Recomendaciones

### T20 — Top-N
**Objetivo:** recomendar las n películas no calificadas por un usuario con
mayor r̂ᵢⱼ.
**Archivos:** `src/recomendaciones.py` (`recomendar_top_n`).
**Cierra:** CA-11.
**Test que se escribe primero:** `test_recomendaciones.py::test_top_n_excluye_peliculas_ya_calificadas`.

### T21 — Extremos por factor latente
**Objetivo:** para cada columna de V, listar las películas con mayor y menor
valor, sin etiquetar el factor.
**Archivos:** `src/recomendaciones.py` (`extremos_por_factor`).
**Cierra:** ninguno (soporta spec §10).
**Test que se escribe primero:** `test_recomendaciones.py::test_extremos_por_factor_devuelve_la_cantidad_pedida_por_columna`.

## 9. Demo

### T22 — CLI
**Objetivo:** definir los argumentos de `python -m src.demo` con los
defaults de `config.py` (sin `--rmse`).
**Archivos:** `src/demo.py` (`construir_parser`).
**Cierra:** ninguno.
**Test que se escribe primero:** `test_demo.py::test_construir_parser_expone_los_argumentos_esperados_y_defaults`.

### T23 — Orquestación de la demo
**Objetivo:** correr el flujo completo (datos → inicialización → entrenar →
comparar → graficar → recomendar) e imprimir los resultados.
**Archivos:** `src/demo.py` (`main`).
**Cierra:** ninguno.
**Test que se escribe primero:** `test_demo.py::test_main_corre_extremo_a_extremo_sobre_dataset_chico_sin_lanzar_excepciones`
(con fixtures chicas en `tests/`, no la MovieLens real).
