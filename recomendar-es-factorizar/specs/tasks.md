# Tareas — Demo "Recomendar es factorizar"

Derivado de `specs/plan.md`. Cada tarea sigue CLAUDE.md: primero el test que
se lista, después el código mínimo para que pase, después `pytest -q`
completo. Una tarea no está terminada si algún test falla (incluidas las de
tareas anteriores). Ninguna tarea toca más de dos módulos de `src/`.

Orden: guarda transversal → datos y preprocesamiento → modelo → descenso de
gradiente genérico → ALS → descenso de gradiente matricial → comparación →
recomendaciones → demo.

Todo test que necesite el dataset real de MovieLens (al menos el de CA-01) se
marca `@pytest.mark.movielens` y se saltea con motivo si `data/ml-100k/` no
existe (ver T00). El resto de los tests usa fixtures chicas en `tests/`, sin
red ni dataset real.

## 0. Guarda transversal

### [x] T00 — Configuración de pytest: marker `movielens`
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

### [x] T01 — Test estático anti-dependencias prohibidas
**Objetivo:** tener desde el arranque un test que falle si algún archivo de
`src/` importa Surprise/implicit/scikit-learn o usa `inv`/`lstsq`/`pinv`.
**Archivos:** ninguno en `src/` (solo `tests/test_dependencias.py`).
**Cierra:** CA-12.
**Test que se escribe primero:** es la tarea completa — `pytest` recorre
`src/` con `ast` y debe pasar vacío (sin archivos) y seguir pasando en cada
tarea siguiente. `test_src_no_importa_librerias_prohibidas`,
`test_src_no_usa_inv_lstsq_pinv`.

## 1. Datos y preprocesamiento

### [x] T02 — Script de descarga de MovieLens
**Objetivo:** descargar y descomprimir el zip de MovieLens 100K a `data/`,
informando si ya existe y fallando con mensaje claro sin red.
**Archivos:** `scripts/descargar_movielens.py`, `src/errores.py`
(`ErrorDescargaDataset`).
**Cierra:** ninguno (soporta spec §3, no tiene CA propio).
**Test que se escribe primero:** `tests/test_descargar_movielens.py::test_descargar_movielens_informa_si_ya_existe`
y `test_descargar_movielens_falla_con_mensaje_y_url_si_no_hay_red` (con la
descarga mockeada, sin red real).

### [x] T03 — Carga de calificaciones crudas
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

### [x] T04 — Filtro iterativo por mínimo k
**Objetivo:** eliminar filas/columnas con menos de k calificaciones hasta
punto fijo, reindexar y actualizar mapeos.
**Archivos:** `src/datos.py` (`filtrar_por_minimo`), `src/errores.py`
(`ErrorDatosInsuficientes`).
**Cierra:** CA-03.
**Test que se escribe primero:** `test_datos.py::test_filtrar_por_minimo_todas_las_filas_y_columnas_tienen_al_menos_k`
(y un test adicional para `ErrorDatosInsuficientes` cuando el filtro vacía la
matriz).

### [x] T05 — Títulos
**Objetivo:** cargar `u.item` y construir el mapeo índice de columna → título.
**Archivos:** `src/datos.py` (`cargar_titulos`, `construir_indice_a_titulo`).
**Cierra:** ninguno (soporta spec §10).
**Test que se escribe primero:** `test_datos.py::test_cargar_titulos_y_construir_indice_a_titulo`
(con un `u.item` de prueba chico en latin-1).

## 2. Modelo

### [x] T06 — Predicción, SCE e inicialización compartida
**Objetivo:** calcular R̂ = U·Vᵀ, la SCE sobre Ω sin factor 1/2, e
inicializar U₀, V₀ uniformes en [0, escala) a partir de un `Generator`
compartido entre ALS y GD.
**Archivos:** `src/modelo.py` (`predecir`, `sce`, `inicializar_factores`,
`ResultadoEntrenamiento`).
**Cierra:** CA-02.
**Test que se escribe primero:** `test_modelo.py::test_sce_ignora_valores_fuera_de_omega`
(incluye reemplazar un NaN fuera de Ω por un número y verificar que `sce` no
cambia), y `test_modelo.py::test_inicializar_factores_reproducible_con_la_misma_semilla`.

## 3. Descenso de gradiente genérico

### [x] T07 — `descenso_gradiente` genérico
**Objetivo:** descenso de gradiente completo para funciones de ℝⁿ, con
criterio de corte por tolerancia o `max_iter`.
**Archivos:** `src/gradiente.py` (`descenso_gradiente`, `ResultadoDescensoGenerico`).
**Cierra:** CA-04.
**Test que se escribe primero:** `test_gradiente_generico.py::test_descenso_gradiente_converge_al_minimo_conocido`
(f(x,y) = (x−1)² + (y−2)² + (x+y−6)², desde (0,0), eta=0.1, epsilon=1e-10).

## 4. ALS

### [x] T08 — `resolver_factor`
**Objetivo:** resolver por fila las ecuaciones normales de mínimos cuadrados
con `np.linalg.solve`, lanzando `SistemaSingularError` si corresponde.
**Archivos:** `src/als.py` (`resolver_factor`), `src/errores.py`
(`SistemaSingularError`).
**Cierra:** CA-05.
**Test que se escribe primero:** `test_als.py::test_resolver_factor_transpuesto_coincide_con_calculo_manual_de_v`
(y un test del caso singular, sin CA propio).

### [x] T09 — `entrenar_als` y reproducibilidad
**Objetivo:** alternar el paso de U y V hasta convergencia o `max_iter`,
devolviendo `ResultadoEntrenamiento`, y verificar que la misma semilla
produce exactamente el mismo resultado.
**Archivos:** `src/als.py` (`entrenar_als`).
**Cierra:** CA-06, CA-09 (parte activa), CA-10 (ALS).
**Test que se escribe primero:** `test_als.py::test_historial_de_sce_de_als_no_crece`,
`test_als.py::test_als_ejemplo_4x5_no_lanza_singular_y_sce_decrece`,
`test_als.py::test_als_ejemplo_4x5_primera_iteracion_coincide_con_informe`
marcado `skip` (motivo: V₀ del ejemplo todavía no está fijado en la spec), y
`test_reproducibilidad.py::test_misma_semilla_misma_ejecucion_als`.

## 5. Descenso de gradiente matricial

### [x] T10 — `gradiente_sce`
**Objetivo:** calcular ∇_U f y ∇_V f sobre el mismo (U, V), sin factor 1/2.
**Archivos:** `src/gradiente.py` (`gradiente_sce`).
**Cierra:** CA-07.
**Test que se escribe primero:** `test_gd_factorizacion.py::test_gradiente_sce_coincide_con_diferencias_finitas`.

### [x] T11 — `entrenar_gd` y reproducibilidad
**Objetivo:** descenso de gradiente completo sobre la factorización,
actualizando U y V simultáneamente con los valores de la iteración t, y
verificar que la misma semilla produce exactamente el mismo resultado.
**Archivos:** `src/gradiente.py` (`entrenar_gd`).
**Cierra:** CA-08, CA-10 (GD).
**Test que se escribe primero:** `test_gd_factorizacion.py::test_iteracion_gd_es_simultanea_contra_referencia`,
`test_reproducibilidad.py::test_misma_semilla_misma_ejecucion_gd`.

## 6. Comparación

### [x] T12 — Tabla de comparación y gráfico de convergencia
**Objetivo:** armar una fila por método (iteraciones, motivo de corte,
tiempo, SCE final) y formatearlas como texto plano con f-strings, sin
dependencias nuevas; graficar f vs. iteración para ALS y GD en el mismo eje,
escala log en y.
**Archivos:** `src/comparacion.py` (`comparar_metodos`, `FilaComparacion`,
`formatear_tabla_comparacion`, `graficar_convergencia`).
**Cierra:** ninguno (soporta spec §9; ver decisiones 2 y 8 de `plan.md` —
sin partición entrenamiento/prueba, una sola SCE por método).
**Test que se escribe primero:** `test_comparacion.py::test_comparar_metodos_arma_una_fila_por_metodo`,
`test_comparacion.py::test_formatear_tabla_comparacion_incluye_las_columnas_esperadas`,
`test_comparacion.py::test_graficar_convergencia_guarda_archivo_en_ruta_dada`
(solo verifica que el archivo se crea, no el contenido visual).

## 7. Recomendaciones

### [x] T13 — Top-N y extremos por factor latente
**Objetivo:** recomendar las n películas no calificadas por un usuario con
mayor r̂ᵢⱼ, y para cada columna de V listar las películas con mayor y menor
valor, sin etiquetar el factor.
**Archivos:** `src/recomendaciones.py` (`recomendar_top_n`,
`extremos_por_factor`).
**Cierra:** CA-11.
**Test que se escribe primero:** `test_recomendaciones.py::test_top_n_excluye_peliculas_ya_calificadas`,
`test_recomendaciones.py::test_extremos_por_factor_devuelve_la_cantidad_pedida_por_columna`.

## 8. Demo

### [x] T14 — Demo completa
**Objetivo:** encadenar carga, filtro y títulos en `preparar_datos_movielens`;
definir los argumentos de `python -m src.demo` con los defaults de
`config.py` (sin `--rmse`); correr el flujo completo (datos → inicialización
→ entrenar → comparar → graficar → recomendar) e imprimir los resultados.
**Archivos:** `src/datos.py` (`preparar_datos_movielens`, `DatosPreparados`),
`src/demo.py` (`construir_parser`, `main`).
**Cierra:** ninguno.
**Test que se escribe primero:** `test_datos.py::test_preparar_datos_movielens_integra_carga_filtro_y_titulos`,
`test_demo.py::test_construir_parser_expone_los_argumentos_esperados_y_defaults`,
`test_demo.py::test_main_corre_extremo_a_extremo_sobre_dataset_chico_sin_lanzar_excepciones`
(con fixtures chicas en `tests/`, no la MovieLens real).

## 9. Recomendador HTML

### [x] T15 — Recomendador estático para un usuario nuevo
**Objetivo:** `src/front.py` genera `salidas/recomendador.html`, página
estática sin servidor ni dependencias externas, con la V de ALS y los
títulos incrustados como JSON. La página muestra las 30 películas con más
calificaciones para calificar de 1 a 5; con al menos k calificaciones
calcula el vector del usuario nuevo resolviendo las ecuaciones normales con
V fija (el mismo paso de U de ALS, informe 5; comentado así en el JS) y
muestra el top-10 de películas no calificadas. Si hay menos de k
calificaciones, pide más en vez de calcular. Como el JS usa la fórmula
cerrada del sistema 2×2, solo soporta k=2 (`DimensionLatenteNoSoportadaError`
si no). Además: default de `--grafico` a `salidas/convergencia.png`,
`salidas/` al `.gitignore` y se borra el `convergencia.png` suelto.
Reemplaza la idea del plano latente.
**Archivos:** `src/front.py` (`generar_html`, `peliculas_mas_calificadas`,
`construir_parser`, `main`), `src/config.py` (`RUTA_GRAFICO_DEFECTO`,
`RUTA_FRONT_DEFECTO`, `N_A_CALIFICAR_DEFECTO`), `src/errores.py`
(`DimensionLatenteNoSoportadaError`), `src/demo.py` (crea la carpeta del
gráfico), `.gitignore`, `specs/plan.md` (default del gráfico).
**Cierra:** ninguno.
**Test que se escribe primero:** `test_front.py::test_generar_html_crea_el_archivo_con_v_y_titulos_en_json`,
`test_front.py::test_vector_de_usuario_con_resolver_factor_coincide_con_formula_2x2_del_js`,
`test_front.py::test_peliculas_mas_calificadas_ordena_por_cantidad_de_calificaciones`,
`test_front.py::test_generar_html_rechaza_k_distinto_de_2`.

### [x] T16 — Ajustes al recomendador HTML y spec
**Objetivo:** la página calcula recién con al menos
`MIN_CALIFICACIONES_FRONT_DEFECTO` = 5 calificaciones (con k el sistema es
resoluble pero queda mal determinado, mismo criterio que el umbral del
filtro); el top-10 muestra solo el orden y los títulos, sin el valor
estimado. Se agrega a `spec.md` la sección 10.1 del front y el CA-13.
**Archivos:** `src/front.py`, `src/config.py`
(`MIN_CALIFICACIONES_FRONT_DEFECTO`), `specs/spec.md` (§10.1, CA-13).
**Cierra:** CA-13 (lo verifica el test de T15
`test_vector_de_usuario_con_resolver_factor_coincide_con_formula_2x2_del_js`).
**Test que se escribe primero:** `test_front.py::test_generar_html_incrusta_el_minimo_de_calificaciones_y_el_js_lo_usa`,
`test_front.py::test_top_n_de_la_pagina_muestra_solo_titulos_sin_valor_estimado`.

### [x] T17 — Validar el mínimo de calificaciones del front
**Objetivo:** `generar_html` verifica que `min_calificaciones >= k` y, si
no, lanza `MinimoCalificacionesInsuficienteError` con un mensaje claro
antes de escribir el HTML (con menos de k calificaciones el sistema del
usuario nuevo es siempre singular).
**Archivos:** `src/front.py` (`generar_html`), `src/errores.py`
(`MinimoCalificacionesInsuficienteError`).
**Cierra:** ninguno.
**Test que se escribe primero:** `test_front.py::test_generar_html_rechaza_min_calificaciones_menor_que_k_sin_escribir_el_archivo`,
`test_front.py::test_generar_html_acepta_min_calificaciones_igual_a_k`.

### [x] T18 — Géneros en el recomendador HTML
**Objetivo:** cargar los géneros de cada película (marcas 0/1 de `u.item`,
nombres de `u.genre`) y mostrarlos en español debajo de cada título, en la
lista para calificar y en el top-10. Solo para mostrar: el modelo no los usa
(regla 8 de CLAUDE.md).
**Archivos:** `src/datos.py` (`cargar_generos`), `src/front.py`
(`GENEROS_EN_ESPANOL`, `generar_html`, `main`), `specs/spec.md` (§10.1).
**Cierra:** ninguno.
**Test que se escribe primero:** `test_datos.py::test_cargar_generos_lee_las_marcas_de_u_item_con_los_nombres_de_u_genre`,
`test_front.py::test_generar_html_incrusta_los_generos_en_espanol`,
`test_front.py::test_generar_html_sin_generos_deja_listas_vacias`.

### [x] T19 — Explicar la escala de calificación en el recomendador HTML
**Objetivo:** que la página aclare que se califica de 1 a 5 y qué significa
cada valor (1 = no me gusta … 5 = me encanta), con una leyenda visible y el
significado en cada botón.
**Archivos:** `src/front.py` (`ETIQUETAS_CALIFICACION`, plantilla),
`specs/spec.md` (§10.1).
**Cierra:** ninguno.
**Test que se escribe primero:** `test_front.py::test_generar_html_explica_la_escala_de_1_a_5`.

## 10. Cambio de dataset: MovieLens latest-small

### [x] T20 — Carga de MovieLens latest-small
**Objetivo:** reemplazar MovieLens 100K por latest-small (películas hasta
2018): leer `ratings.csv` y `movies.csv` (CSV en UTF-8 con encabezado, con
el módulo `csv` porque hay títulos con coma entre comillas; calificaciones
de 0,5 a 5), con la ruta por defecto `data/ml-latest-small/`.
**Archivos:** `src/datos.py` (`cargar_calificaciones`, `cargar_titulos`,
`cargar_generos`, `preparar_datos_movielens`), `src/demo.py` y
`src/front.py` (nombres de archivo), `src/config.py`
(`RUTA_DATOS_DEFECTO`), `tests/conftest.py` y `pytest.ini` (marker
`movielens`), `specs/spec.md` (§1-3, §10.1, CA-01), `specs/plan.md`,
`CLAUDE.md`.
**Cierra:** CA-01 (nuevo: 610 × 9724, 100.836 calificaciones).
**Test que se escribe primero:** los de `test_datos.py`
(`test_cargar_calificaciones_reindexa_ids_y_arma_mascara`,
`test_cargar_calificaciones_dimensiones_y_mascara`,
`test_cargar_titulos_y_construir_indice_a_titulo`,
`test_cargar_generos_separa_la_columna_genres_de_movies_csv`,
`test_preparar_datos_movielens_integra_carga_filtro_y_titulos`) y el
dataset chico de `test_demo.py`, pasados a CSV.

### [x] T21 — Script de descarga de MovieLens latest-small
**Objetivo:** que `scripts/descargar_movielens.py` baje
`ml-latest-small.zip` de GroupLens y lo descomprima en
`data/ml-latest-small/`.
**Archivos:** `scripts/descargar_movielens.py` (`URL_MOVIELENS`,
`NOMBRE_CARPETA_DATASET`), `specs/plan.md`.
**Cierra:** ninguno.
**Test que se escribe primero:** `test_descargar_movielens.py::test_la_url_por_defecto_es_la_de_movielens_latest_small`
(y los dos tests de T02, apuntados a la carpeta nueva).

### [x] T22 — Géneros de latest-small en el recomendador HTML
**Objetivo:** traducir los géneros tal como los escribe `movies.csv`
(`Children`, `IMAX`, `(no genres listed)` → "sin género"), sacando los de
100K que ya no aparecen (`Children's`, `unknown`).
**Archivos:** `src/front.py` (`GENEROS_EN_ESPANOL`).
**Cierra:** ninguno.
**Test que se escribe primero:** `test_front.py::test_generar_html_incrusta_los_generos_en_espanol`
(con los nombres nuevos) y `test_front.py::test_todos_los_generos_de_movielens_tienen_traduccion`
(marcado `movielens`: todo género del `movies.csv` real tiene traducción).

### [x] T23 — Recalibración de defaults sobre MovieLens latest-small
**Objetivo:** repetir el experimento de calibración (mismo criterio: menor
umbral y después menor k con max|r̂| fuera de Ω < 7 para ALS y GD a la vez;
eta que converja sin que f aumente) sobre el dataset nuevo, y dejar los
valores definitivos en `config.py`: `UMBRAL_DEFECTO=40`, `K_DEFECTO=2`,
`ETA_DEFECTO=5e-4`.
**Archivos:** `src/config.py`, `specs/bitacora.md` (tablas y conclusión),
`specs/plan.md` (bloque de `config.py`).
**Cierra:** ninguno.
**Test que se escribe primero:** `test_calibracion.py::test_defaults_de_config_cumplen_el_criterio_de_calibracion_sobre_movielens`
(marcado `movielens`). No arranca en rojo: los defaults viejos (umbral=50)
también cumplen el criterio. Lo que elige el menor umbral es el experimento
de la bitácora; el test protege el criterio ante cambios futuros de los
defaults.
