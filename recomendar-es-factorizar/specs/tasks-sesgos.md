# Tareas — Experimento de sesgos (rama experimento/sesgos)

Derivado de `specs/sesgos.md` (en adelante "spec S"). Las once preguntas de
la propuesta original están resueltas y volcadas en la spec S y en CLAUDE.md
(ver "Preguntas resueltas" al final).

Reglas de trabajo (las de CLAUDE.md y las de `specs/tasks-regularizacion.md`,
aplicadas a esta rama):
- Una tarea por vez: primero el test, verlo fallar, después el código mínimo,
  después `pytest -q` completo. Una tarea no está terminada si falla algún
  test, incluidos los 55 de main, los de la rama de regularización (CA-R01
  incluido) y los de los experimentos 3 y 4 (CA-S05 se verifica en todas las
  tareas). Los tests existentes no se editan; sumar nombres a su línea de
  imports está permitido.
- `src/`: solo el módulo nuevo `src/sesgos.py` (spec S §10), que incluye el
  bucle de ALS con sesgos. `resolver_factor` no se modifica: el paso
  aumentado la llama con [V | 1] o [U | 1].
- Esta rama no usa descenso de gradiente (spec S §3).
- Los tests que necesitan el dataset real se marcan `@pytest.mark.movielens`.
- Commit al cerrar cada tarea, solo con los archivos que lista, mensaje
  "TSxx: <objetivo>".
- **Criterio de éxito congelado.** Los umbrales de la spec S §8 y
  `UMBRAL_RELEVANTE` entran a `config.py` en TS05, antes de cualquier corrida
  sobre MovieLens; desde que arranque TS08 no se tocan.

Orden: f y predicción con sesgos → ordenamientos → paso aumentado →
entrenamiento → precisión@10 → barrido y selección → reentrenamiento,
concentración y criterio → corrida y bitácora.

## 1. Núcleo en src/sesgos.py

### [x] TS01 — f con sesgos y predicción (spec S §2)
**Objetivo:** en `src/sesgos.py`: `calcular_mu(R, M)` (promedio de las
calificaciones observadas en M: Ω_ent o todo Ω, según el modelo),
`predecir_con_sesgos(U, V, b, c, mu)` = μ + b_i + c_j + U_i·V_j, y
`f_sesgos(R, M, U, V, b, c, mu, lambda_)` = SCE sobre Ω de r − r̂ +
λ (Σ‖U_i‖² + Σ‖V_j‖² + Σ b_i² + Σ c_j²). Valida λ con `validar_lambda`.
**Archivos:** `src/sesgos.py`.
**Cierra:** ninguno (soporte de CA-S02; junto con CA-S01 y CA-S02 cubre lo
que verificaba el CA-S01 original, eliminado).
**Test que se escribe primero:**
- `test_sesgos.py::test_calcular_mu_promedia_solo_omega` (los NaN y lo que
  está fuera de M no cuentan).
- `test_sesgos.py::test_predecir_con_sesgos_suma_mu_b_c_y_u_por_vt` (2 × 3 a mano).
- `test_sesgos.py::test_f_sesgos_con_sesgos_y_mu_en_cero_es_f_regularizada`
  (igualdad exacta).
- `test_sesgos.py::test_f_sesgos_calculada_a_mano` (incluye λ·(b² + c²)).
- `test_sesgos.py::test_f_sesgos_ignora_valores_fuera_de_omega`.

### [x] TS02 — Ordenamientos A y B (spec S §6)
**Objetivo:** `puntajes_ordenamiento_a(U, V, b, c, mu)` = r̂ completo y
`puntajes_ordenamiento_b(U, V)` = U·Vᵀ (sin c_j; μ y b_i no cambian el orden
de un usuario).
**Archivos:** `src/sesgos.py`.
**Cierra:** CA-S04.
**Test que se escribe primero:**
- `test_sesgos.py::test_ordenamiento_a_es_r_hat_completo` (caso chico).
- `test_sesgos.py::test_ordenamiento_b_no_depende_de_c` (cambiar c no cambia
  los puntajes B; sí cambia los A).
- `test_sesgos.py::test_ordenamiento_b_da_el_mismo_orden_por_usuario_que_r_hat_sin_c`
  (sumar μ + b_i a una fila no cambia su orden).

### [x] TS03 — Paso aumentado con `resolver_factor` (spec S §3)
**Objetivo:** `paso_usuarios(R, M, V, c, mu, lambda_)` → (U, b): llama a
`resolver_factor(R − μ − c, M, [V | 1], lambda_)` y separa la última columna
como b. `paso_peliculas(R, M, U, b, mu, lambda_)` → (V, c): la misma llamada
con (R − μ − b)ᵀ, Mᵀ y [U | 1]. Una sola función resuelve los dos pasos
(regla 4 de CLAUDE.md). Los NaN de R se mantienen (regla 2). El mismo paso
de usuarios calcula el vector (u, b) de un usuario simulado (spec S §7).
**Archivos:** `src/sesgos.py`.
**Cierra:** CA-S01.
**Test que se escribe primero:**
- `test_sesgos.py::test_paso_usuarios_coincide_con_el_sistema_de_k_mas_1_incognitas`
  (CA-S01: matriz chica, sistema armado a mano con np.linalg.solve,
  `np.allclose(rtol=1e-10)`).
- `test_sesgos.py::test_paso_peliculas_coincide_con_el_sistema_de_k_mas_1_incognitas`.
- `test_sesgos.py::test_los_dos_pasos_usan_resolver_factor` (espía: se llama
  una vez por paso, con la matriz aumentada).

### [x] TS04 — Entrenamiento ALS con sesgos
**Objetivo:** `entrenar_als_sesgos(R, M, U0, V0, mu, epsilon, max_iter,
lambda_)` en `src/sesgos.py`: b y c arrancan en 0; alterna `paso_usuarios` y
`paso_peliculas`; corta con |f(t+1) − f(t)| < ε usando `f_sesgos`; devuelve
U, V, b, c, historial, iteraciones y motivo de corte. Sin camino para fijar
sesgos (el CA-S01 original se eliminó).
**Archivos:** `src/sesgos.py`.
**Cierra:** CA-S02.
**Test que se escribe primero:**
- `test_sesgos.py::test_f_sesgos_no_crece_entre_iteraciones` (CA-S02).
- `test_sesgos.py::test_historial_guarda_f_sesgos_del_modelo_final`.
- `test_sesgos.py::test_entrenar_als_sesgos_es_reproducible`.
- `test_sesgos.py::test_entrenar_als_sesgos_aprende_sesgos_no_nulos`
  (en una matriz con un usuario que califica todo alto, su b queda > 0).

## 2. Experimento (experimentos/sesgos.py, fuera de src/)

### [x] TS05 — Precisión@10 y constantes (spec S §5, §7 y §8)
**Objetivo:** `precision_en_n(puntajes, M_ent, R, M_prueba, top_n,
umbral_relevante)`: para cada usuario con al menos una calificación ≥
`umbral_relevante` en Ω_prueba, top-N entre las películas que no tiene en
Ω_ent; precisión = aciertos / N; promedio sobre esos usuarios. Constantes
nuevas en `config.py` (no se tocan las de la regularización):
`UMBRAL_RELEVANTE` = 4, `GRILLA_K_SESGOS` = (2, 5, 10, 20),
`GRILLA_LAMBDA_SESGOS` = (1, 5, 10, 20), `MAX_FRECUENCIA_SESGOS` = 0,25,
`MIN_PELICULAS_DISTINTAS_SESGOS` = 127 (comentario: sale del experimento 3)
y `MAX_FRACCION_FUERA_DE_RANGO_SESGOS` = 0,01. Congeladas desde acá.
**Archivos:** `experimentos/sesgos.py`, `src/config.py`.
**Cierra:** CA-S03, CA-S06 (parte de constantes).
**Test que se escribe primero:**
- `test_experimento_sesgos.py::test_precision_en_n_calculada_a_mano` (CA-S03:
  3 usuarios, top-2; uno sin relevantes en prueba queda afuera del promedio).
- `test_experimento_sesgos.py::test_precision_excluye_del_top_lo_que_esta_en_entrenamiento`.
- `test_experimento_sesgos.py::test_relevante_es_mayor_o_igual_que_el_umbral`
  (4,0 cuenta; 3,5 no).
- `test_experimento_sesgos.py::test_config_tiene_grilla_y_umbrales_de_la_spec_s`.

### [x] TS06 — Barrido, selección y línea de base (spec S §5 y §7)
**Objetivo:** para cada par (k, λ) de `GRILLA_K_SESGOS` × `GRILLA_LAMBDA_SESGOS`:
`entrenar_als_sesgos` sobre Ω_ent desde `SEMILLA_INICIALIZACION`, con μ de
Ω_ent; registrar iteraciones, SCE sobre Ω_prueba (r̂ completo), fuera de
rango (spec R §8, sobre r̂ completo) y precisión@10 con A y con B. Selección:
menor SCE de prueba, sin regla de empate. Línea de base de precisión:
`entrenar_als`, k = 2, λ = 0, sobre Ω_ent, con `SEMILLA_INICIALIZACION`,
`EPSILON_DEFECTO` y `MAX_ITER_DEFECTO`, ordenamiento absoluto. CLI
`python -m experimentos.sesgos` con defaults de config.
**Archivos:** `experimentos/sesgos.py`.
**Cierra:** CA-S06.
**Test que se escribe primero:**
- `test_experimento_sesgos.py::test_barrido_devuelve_una_fila_por_par_con_precision_a_y_b`.
- `test_experimento_sesgos.py::test_barrido_usa_mu_de_omega_ent`.
- `test_experimento_sesgos.py::test_seleccion_elige_la_menor_sce_de_prueba_sin_empate`.
- `test_experimento_sesgos.py::test_linea_de_base_usa_entrenar_als_de_main_sobre_omega_ent`
  (espía: k = 2, λ = 0, máscara de entrenamiento, semilla de inicialización).
- `test_experimento_sesgos.py::test_parser_toma_los_defaults_de_config`.

### [x] TS07 — Reentrenamiento, concentración y criterio (spec S §5, §7 y §8)
**Objetivo:** con el par elegido, reentrenar sobre todo Ω con μ de todo Ω, y
medir sobre ese modelo el fuera de rango (r̂ completo) y la concentración de
los 321 usuarios reales con A y con B (reusa `medir_tops` de
`experimentos/ordenamiento_relativo.py`). Referencia fuera del criterio:
usuarios simulados con notas reales, con (u, b) del paso de usuarios
aumentado (objetivo r − μ − c, mismo λ). `evaluar_criterio`: éxito si A o B
cumple los cuatro puntos de la §8; el criterio 3 (precisión) sale del
modelo de la partición y los criterios 1, 2 y 4, del reentrenado.
**Archivos:** `experimentos/sesgos.py`.
**Cierra:** ninguno (soporte de CA-S07).
**Test que se escribe primero:**
- `test_experimento_sesgos.py::test_reentrenamiento_usa_todo_omega_y_mu_de_todo_omega`.
- `test_experimento_sesgos.py::test_usuario_simulado_usa_el_paso_de_usuarios_aumentado`.
- `test_experimento_sesgos.py::test_criterio_exitoso_si_a_o_b_cumple_todo`,
  `..._falla_si_ninguno_cumple_los_cuatro` y
  `..._no_mezcla_criterios_de_a_y_de_b` (resultados sintéticos).
- `test_experimento_sesgos.py::test_main_corre_de_punta_a_punta_sobre_dataset_chico`
  (humo, dataset sintético, sin MovieLens).

### [ ] TS08 — Corrida única y registro
**Objetivo:** correr `python -m experimentos.sesgos` una sola vez sobre
MovieLens; registrar en `specs/bitacora.md` la tabla completa del barrido, el
tamaño de la partición, el par elegido, la línea de base, las métricas de A
y de B (precisión, concentración, fuera de rango), la referencia con
simulados y el veredicto criterio por criterio. Si no cumple, se registra
como está, sin cambiar grilla, semillas, umbrales ni ordenamientos.
**Archivos:** `specs/bitacora.md`.
**Cierra:** CA-S07, CA-S05.
**Test que se escribe primero:** ninguno nuevo; `pytest -q` en verde antes de
correr.

## Preguntas resueltas (decisión y dónde quedó)

1. **CA-S01 y μ.** El CA-S01 original se elimina, sin camino de código para
   fijar sesgos; la corrección la cubren el test de TS01 (f con sesgos y μ en
   0 == `f_regularizada`), CA-S01 y CA-S02. Los CA-S se renumeraron
   (el anterior CA-S02 es el actual CA-S01, y así). Spec S §9; TS01, TS04.
2. **μ al reentrenar.** μ es el promedio de los datos de entrenamiento de
   cada modelo: Ω_ent en la partición, todo Ω en el reentrenado. Spec S §2 y
   §5; TS01, TS06, TS07.
3. **Qué modelo mide cada criterio.** Criterio 3 sobre la partición;
   criterios 1, 2 y 4 sobre el modelo reentrenado. Spec S §5; TS07.
4. **Nombres de la grilla.** `GRILLA_K_SESGOS` y `GRILLA_LAMBDA_SESGOS`; las
   existentes no se tocan. Spec S §5; TS05, TS06.
5. **ALS con sesgos fuera de `als.py`.** El bucle va en `src/sesgos.py`.
   Spec S §10; TS04.
6. **Línea de base.** `entrenar_als`, k = 2, λ = 0, sobre Ω_ent, con
   `SEMILLA_INICIALIZACION`, `EPSILON_DEFECTO` y `MAX_ITER_DEFECTO`. Spec S
   §7; TS06.
7. **Usuario simulado.** Mismo paso aumentado que un usuario real (u y b con
   [V | 1], objetivo r − μ − c, mismo λ). Spec S §7; TS03, TS07.
8. **Fuera de rango.** Siempre sobre r̂ completo. Spec S §7; TS06, TS07.
9. **Umbrales.** Propios: `MIN_PELICULAS_DISTINTAS_SESGOS` = 127 (sale del
   experimento 3), `MAX_FRECUENCIA_SESGOS` = 0,25 y
   `MAX_FRACCION_FUERA_DE_RANGO_SESGOS` = 0,01; no se reusan los de la
   regularización. Spec S §8; TS05.
10. **CLAUDE.md.** La regla 3 nombra la f de specs/sesgos.md §2 para
    experimento/sesgos. Commiteado junto con las specs.
11. **Descenso de gradiente.** La spec S §3 deja explícito que esta rama no lo
    usa. Reglas de trabajo de este archivo.
