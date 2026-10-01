# Tareas — Experimento de regularización (rama experimento/regularizacion)

Derivado de `specs/regularizacion.md` (en adelante "spec R"). Las doce
preguntas de la propuesta original están resueltas y volcadas en la spec R y
en CLAUDE.md (ver "Preguntas resueltas" al final): se puede arrancar TR01.

Reglas de trabajo (las de CLAUDE.md, aplicadas a esta rama):
- Una tarea por vez: primero el test, verlo fallar, después el código mínimo,
  después `pytest -q` completo. Una tarea no está terminada si falla algún
  test, incluidos los de main (CA-R06 se verifica en todas las tareas: los
  55 tests de main siguen pasando sin modificaciones; los tests nuevos se
  agregan, nunca se edita uno existente). CA-R06 protege lo que verifica
  cada test de main, no las líneas del archivo: sumar nombres a la línea de
  imports de un archivo de tests existente no lo viola. Los imports nuevos
  van en esa línea de arriba, nunca a mitad de archivo ni con `noqa`.
- Todo parámetro nuevo de `src/` lleva default que reproduce main (λ = 0).
- Los tests que necesitan el dataset real se marcan `@pytest.mark.movielens`.
- Commit al cerrar cada tarea, solo con los archivos que lista, mensaje
  "TRxx: <objetivo>".
- **Criterios de éxito congelados.** Los umbrales de la sección 9 de la spec R
  (130 películas, 25 %, 1 % fuera de rango, 1 % de empate) entran a
  `config.py` en TR06, antes de cualquier corrida sobre MovieLens. Se pueden
  cambiar hasta que arranque TR10; desde ese momento no se tocan.

Orden: λ en ALS → f regularizada → λ en el gradiente → λ en el entrenamiento
(con la garantía de CA-R01) → partición → métricas y criterios → barrido y
selección → reentrenamiento y verificación con GD → concentración con λ →
corrida y bitácora.

## 1. Núcleo en src/ (siempre con λ = 0 por defecto)

### [x] TR01 — λ en `resolver_factor`
**Objetivo:** agregar el parámetro `lambda_` (default 0) a `resolver_factor` y
resolver por fila (V_iᵀ V_i + λ I) U_iᵀ = V_iᵀ r_i con `np.linalg.solve`,
como en main (spec R §4). La misma función sigue resolviendo el paso de V
con Rᵀ y Mᵀ. La singularidad se evalúa sobre la A ya regularizada (no había
chequeo previo por cantidad de calificaciones: se detecta solo por el
`LinAlgError` de `np.linalg.solve`). `lambda_` < 0 lanza
`LambdaNegativoError` (hereda de `ErrorFactorizacion` y de `ValueError`)
con el valor recibido en el mensaje.
**Archivos:** `src/als.py`, `src/errores.py` (`LambdaNegativoError`).
**Cierra:** CA-R02, CA-R03.
**Test que se escribe primero:**
- `test_als.py::test_resolver_factor_con_lambda_satisface_las_ecuaciones_normales_regularizadas`
  (CA-R03: matriz chica y bien condicionada, armada a mano; verificar
  (Vᵀ V + λ I) u = Vᵀ r con `np.allclose(..., rtol=1e-10)`).
- `test_als.py::test_resolver_factor_con_lambda_positivo_no_es_singular_con_menos_de_k_calificaciones`
  (CA-R02: usuario con 1 calificación y k = 2, λ > 0 → sin `SistemaSingularError`).
- `test_als.py::test_resolver_factor_con_lambda_cero_es_identico_al_actual`
  (igualdad exacta con la llamada sin λ, también por el camino transpuesto).
- `test_als.py::test_resolver_factor_con_lambda_negativo_lanza_value_error_con_el_valor`.

### [x] TR02 — f regularizada (spec R §3) y `validar_lambda`
**Objetivo:** función aparte `f_regularizada(R, M, U, V, lambda_)` = SCE sobre
Ω + λ (Σ‖U_i‖² + Σ‖V_j‖²) en el módulo compartido (spec R §3). `sce` no se
toca: queda como la SCE pura y es la que se usa para medir sobre Ω_prueba;
`f_regularizada` es la de los criterios de corte. La validación de λ ≥ 0 se
centraliza en `validar_lambda` (en `modelo.py`), que usan `f_regularizada` y
`resolver_factor` (reemplaza el chequeo propio que tenía desde TR01).
**Archivos:** `src/modelo.py` (`validar_lambda`, `f_regularizada`),
`src/als.py` (`resolver_factor` usa `validar_lambda`).
**Cierra:** ninguno (soporte de CA-R01 y CA-R04).
**Test que se escribe primero:**
- `test_modelo.py::test_f_regularizada_con_lambda_cero_es_la_sce`
  (`f_regularizada(..., 0) == sce(...)`, igualdad exacta).
- `test_modelo.py::test_f_regularizada_suma_lambda_por_normas_al_cuadrado`
  (valor calculado a mano en una matriz de 2 × 2).
- `test_modelo.py::test_f_regularizada_ignora_valores_fuera_de_omega`
  (como CA-02, con λ > 0).
- `test_modelo.py::test_f_regularizada_con_lambda_negativo_lanza_lambda_negativo_error`.
- `test_modelo.py::test_validar_lambda_acepta_cero_y_positivos_y_rechaza_negativos`.
- Los 4 tests de TR01 siguen pasando sin cambios.

### [x] TR03 — λ en `gradiente_sce`
**Objetivo:** ∇_U f = −2 E V + 2 λ U, ∇_V f = −2 Eᵀ U + 2 λ V (spec R §5),
con `lambda_` default 0; los dos gradientes se siguen calculando sobre el
mismo (U, V). `gradiente_sce` conserva el nombre; su docstring aclara que
devuelve el gradiente de `f_regularizada` (spec R §3), que con λ = 0 es el
de la SCE. Valida λ con `validar_lambda`.
**Archivos:** `src/gradiente.py`.
**Cierra:** CA-R04.
**Test que se escribe primero:**
- `test_gd_factorizacion.py::test_gradiente_con_lambda_coincide_con_diferencias_finitas`
  (CA-R04: diferencias finitas de `f_regularizada`, λ > 0, error relativo
  < 1e-5 como CA-07).
- `test_gd_factorizacion.py::test_gradiente_con_lambda_cero_es_identico_al_actual`
  (igualdad exacta con la llamada sin λ).
- `test_gd_factorizacion.py::test_gradiente_con_lambda_negativo_lanza_lambda_negativo_error`.
- El test actual de CA-07 sigue pasando sin cambios (λ = 0).

### [x] TR04 — λ en `entrenar_als` y `entrenar_gd`, y garantía de main
**Objetivo:** pasar λ (default 0) a los dos entrenamientos (spec R §11): ALS
usa `resolver_factor(..., λ)`, GD usa `gradiente_sce(..., λ)`, y ambos cortan
con |f(t+1) − f(t)| < ε usando `f_regularizada` (que con λ = 0 es la `sce`
que usan hoy). `historial_f` guarda los valores de `f_regularizada`.
`DivergenciaError` igual que en main.
**Archivos:** `src/als.py`, `src/gradiente.py`.
**Cierra:** CA-R01.
**Test que se escribe primero:**
- `test_regularizacion.py::test_lambda_cero_sin_particion_reproduce_main`
  (CA-R01, marcado `movielens`: con los defaults de config, ALS da SCE
  22.937,3886 en 10 iteraciones y GD 23.045,8796 en 462). **Si este test no
  pasa, se frena la rama** (no se sigue con TR05).
- `test_als.py::test_historial_de_f_de_als_con_lambda_no_crece` (matriz chica,
  λ > 0: cada paso de ALS minimiza f exactamente, así que f no crece).
- `test_gd_factorizacion.py::test_iteracion_gd_con_lambda_es_simultanea_contra_referencia`
  (como CA-08, con λ > 0).
- `test_als.py::test_entrenar_als_con_lambda_cero_es_identico_al_actual`
  (U, V e historial idénticos a la llamada sin λ).

### [x] TR05 — Partición entrenamiento/prueba
**Objetivo:** `particionar(M, fraccion_prueba, generador)`: después del
filtro por umbral, sortear sin reposición round(fraccion · |Ω|) pares para
prueba con un `np.random.Generator` de semilla fija, pasar a entrenamiento
todo par de prueba cuyo usuario o película quede sin calificaciones de
entrenamiento, y loguear el tamaño real de cada conjunto (spec R §6).
Devuelve dos máscaras, M_ent y M_prueba; R no se toca. `FRACCION_PRUEBA`
(0,2) y `SEMILLA_PARTICION = SEMILLA_DEFECTO` (por referencia) van a
`config.py`. El tamaño real (después de pasar pares a entrenamiento) se
registra en la bitácora en TR10.
**Archivos:** `src/datos.py`, `src/config.py`.
**Cierra:** CA-R05, CA-R07 (parte de la partición).
**Test que se escribe primero:**
- `test_datos.py::test_particion_es_disjunta_y_su_union_es_omega`.
- `test_datos.py::test_particion_es_reproducible_con_la_semilla`.
- `test_datos.py::test_particion_todo_usuario_y_pelicula_de_prueba_tiene_calificaciones_de_entrenamiento`
  (máscara rala, fracción 0,5).
- `test_datos.py::test_particion_pasa_a_entrenamiento_los_pares_que_dejarian_huerfano_a_un_usuario_o_pelicula`
  (con fracción 1 todos los pares quedan huérfanos y vuelven a entrenamiento).

## 2. Experimento (experimentos/, fuera de src/)

### [x] TR06 — Métricas y criterios de éxito
**Objetivo:** en `experimentos/regularizacion.py`, funciones puras para:
SCE sobre Ω_prueba (se usa `sce` con M_prueba, sin función nueva),
`medir_fuera_de_rango` (fracción de estimaciones fuera de
[ESCALA_MIN − 1, ESCALA_MAX + 1] y máximo de |r̂| sobre los pares fuera de
Ω, spec R §8) y `evaluar_criterios` (devuelve un `Veredicto` con cada
criterio) contra la spec R §9. Recibe
por separado lo que sale de la partición (criterio 2: SCE de prueba del par
elegido y de (k = 2, λ = 0)) y lo que sale del modelo reentrenado sobre
todo Ω (criterios 3 y 4), según spec R §7. Constantes nuevas en
`config.py`: `ESCALA_MIN` = 0,5 y `ESCALA_MAX` = 5 (rango permitido
[−0,5; 6]) y los umbrales de §9 (congelados desde acá, ver arriba):
`MIN_PELICULAS_EXITO` = 130, `MAX_FRECUENCIA_EXITO` = 0,25,
`MAX_FRACCION_FUERA_DE_RANGO` = 0,01 y `TOLERANCIA_EMPATE` = 0,01. "Fuera
de Ω" son solo los pares no observados: ni Ω_ent ni Ω_prueba (spec R §8).
**Archivos:** `experimentos/regularizacion.py`, `src/config.py`.
**Cierra:** CA-R07 (parte de rango y umbrales).
**Test que se escribe primero:**
- `test_experimento_regularizacion.py::test_fuera_de_rango_cuenta_solo_pares_no_observados`
  (los pares de Ω_prueba con r̂ fuera de [−0,5; 6] no cuentan).
- `test_experimento_regularizacion.py::test_fuera_de_rango_usa_los_limites_menos_medio_y_seis`
  (r̂ = −0,5 y r̂ = 6 están dentro; −0,51 y 6,01, fuera).
- `test_experimento_regularizacion.py::test_evaluar_criterios_falla_si_k_es_2`,
  `..._falla_si_la_sce_de_prueba_no_mejora_a_k2_lambda0`,
  `..._falla_con_mas_de_1_por_ciento_fuera_de_rango`,
  `..._falla_con_menos_de_130_peliculas_o_mas_de_25_por_ciento`,
  `..._exitoso_si_cumple_todo` (resultados sintéticos, con los bordes "al
  menos" y "a lo sumo" incluidos).
- `test_experimento_regularizacion.py::test_config_tiene_la_escala_y_los_umbrales_de_la_seccion_9`.

### [x] TR07 — Barrido de (k, λ) y selección
**Objetivo:** para cada par de la grilla (`GRILLA_K` = {2, 3, 5, 10},
`GRILLA_LAMBDA` = {0, 1, 5, 10, 20} en `config.py`), ALS sobre Ω_ent con U₀,
V₀ generados con `SEMILLA_INICIALIZACION` (nueva en `config.py`, separada de
`SEMILLA_PARTICION`; la misma para todos los pares; se define como
`SEMILLA_INICIALIZACION = SEMILLA_DEFECTO`, por referencia y no con el
literal 42, para que (k = 2, λ = 0) arranque del mismo U₀, V₀ que main),
registrando
iteraciones, f final, SCE sobre Ω_prueba y fuera de rango (spec R §7). Un
par que falla (`SistemaSingularError`) queda en la tabla con el error, no
corta el barrido. Selección: entre los pares a menos del 1 % de la mejor
SCE de prueba gana el de menor k; si hay varios con ese k, el de menor SCE
de prueba. CLI `python -m experimentos.regularizacion` con todo como
argumentos (defaults de config).
**Archivos:** `experimentos/regularizacion.py`, `src/config.py`.
**Cierra:** CA-R07.
**Test que se escribe primero:**
- `test_experimento_regularizacion.py::test_barrido_devuelve_una_fila_por_par_de_la_grilla`
  (matriz chica, grilla chica pasada por argumento).
- `test_experimento_regularizacion.py::test_barrido_registra_el_error_de_un_par_singular_y_sigue`.
- `test_experimento_regularizacion.py::test_seleccion_elige_menor_sce_de_prueba`,
  `..._con_empate_menor_al_1_por_ciento_gana_el_menor_k` y
  `..._con_varios_pares_del_menor_k_gana_el_de_menor_sce` (filas sintéticas).
- `test_experimento_regularizacion.py::test_todos_los_lambda_de_un_k_arrancan_del_mismo_u0_v0`.
- `test_experimento_regularizacion.py::test_seleccion_ignora_los_pares_con_error`.
- `test_experimento_regularizacion.py::test_config_de_la_grilla_y_las_semillas`.
- `test_experimento_regularizacion.py::test_parser_toma_los_defaults_de_config` (CA-R07).

### [ ] TR08 — Reentrenamiento sobre todo Ω y verificación con GD
**Objetivo:** con el par elegido, (a) reentrenar con ALS sobre todo Ω, sin
partición, para evaluar los criterios 3 y 4 (spec R §7, último punto); y
(b) correr descenso de gradiente sobre la partición y registrar su SCE de
prueba, dividiendo η por 2 y arrancando de nuevo desde la misma
inicialización si f deja de ser finita, hasta `MAX_REDUCCIONES_ETA` veces
(default 3, en `config.py`); cada intento se registra y, si sigue
divergiendo, queda como falla de GD sin cambiar el veredicto (spec R §5,
último punto).
**Archivos:** `experimentos/regularizacion.py`, `src/config.py`.
**Cierra:** CA-R07 (parte de `MAX_REDUCCIONES_ETA`).
**Test que se escribe primero:**
- `test_experimento_regularizacion.py::test_reentrenar_sobre_todo_omega_usa_el_par_elegido_y_la_mascara_completa`
  (matriz chica: el modelo devuelto se entrenó con M completa, no con M_ent).
- `test_experimento_regularizacion.py::test_verificar_gd_divide_eta_por_2_al_divergir_y_registra_cada_intento`
  (`entrenar_gd` reemplazado por un doble que lanza `DivergenciaError` las
  primeras n veces: se verifica la secuencia de η y la lista de intentos).
- `test_experimento_regularizacion.py::test_verificar_gd_se_rinde_tras_max_reducciones_y_registra_la_falla`
  (siempre diverge: `MAX_REDUCCIONES_ETA` + 1 intentos y resultado "falla de GD",
  sin excepción hacia afuera).

### [ ] TR09 — Concentración con λ
**Objetivo:** que `experimentos/concentracion.py` reciba k y λ, entrene ALS
sobre todo Ω con ese λ (como el modelo reentrenado de TR08) y resuelva el
usuario simulado con `resolver_factor(..., λ)` y notas con distribución real
(spec R §8). Con λ = 0 y k = 2 tiene que seguir dando lo registrado en la
bitácora (65 películas; 44,5 % con notas reales).
**Archivos:** `experimentos/concentracion.py`.
**Cierra:** ninguno (soporte de CA-R08).
**Test que se escribe primero:**
- `test_experimento_concentracion.py::test_simular_con_lambda_cero_reproduce_la_bitacora`
  (marcado `movielens`).
- `test_experimento_concentracion.py::test_simular_pasa_lambda_a_resolver_factor`
  (matriz chica: con λ > 0 el u simulado cumple las ecuaciones regularizadas).

### [ ] TR10 — Corrida completa y registro
**Objetivo:** correr `python -m experimentos.regularizacion` sobre MovieLens
con los criterios ya congelados; con el par elegido, el reentrenamiento, la
verificación con GD (con sus intentos de η) y la concentración de TR09;
registrar en `specs/bitacora.md` la tabla completa del barrido (k, λ,
iteraciones, SCE de prueba, fuera de rango), el tamaño real de
entrenamiento y prueba (después de pasar pares a entrenamiento), el
par elegido con la aclaración de que su SCE de prueba es optimista, la
comparación con GD (y cada intento de η, o la falla), la concentración y el
veredicto contra la spec R §9 (criterio 2 sobre la partición; 3 y 4 sobre
el modelo reentrenado). Si falla algún criterio se registra igual y la rama
no se integra.
**Archivos:** `specs/bitacora.md`.
**Cierra:** CA-R08, CA-R06 (verificación final: los 55 tests de main
siguen pasando sin modificaciones y el único salteado es el de la V₀ del
informe).
**Test que se escribe primero:** ninguno nuevo (tarea de corrida y
registro); antes de correr, `pytest -q` completo en verde.

## Preguntas resueltas (decisión y dónde quedó)

1. **Solver.** main usa `np.linalg.solve`, permitido por la regla 5 (prohíbe
   inv, lstsq y pinv). Spec R §4 corregida; TR01 lo usa.
2. **Dónde vive la f de §3.** Función aparte, `f_regularizada`, en
   `src/modelo.py`; `sce` queda como la SCE pura y se usa para medir sobre
   Ω_prueba; `f_regularizada` se usa en los criterios de corte. Con λ = 0,
   `f_regularizada == sce`. Spec R §3 y §5; TR02, TR03, TR04.
3. **λ en `entrenar_als` y `entrenar_gd`.** Sí reciben λ; duplicar los
   bucles en `experimentos/` haría que CA-R01 no pruebe el código real.
   Spec R §11 corregida; TR04 lo implementa.
4. **Rango de §8.** `ESCALA_MIN` = 0,5 y `ESCALA_MAX` = 5 (escala del
   dataset); rango permitido [−0,5; 6]. Spec R §8; TR06.
5. **"Fuera de Ω" con partición.** Solo los pares no observados; los de
   prueba no cuentan. Spec R §8; TR06.
6. **Empate del 1 %.** Entre los pares a menos del 1 % de la mejor SCE de
   prueba gana el de menor k; si hay varios con ese k, el de menor SCE de
   prueba. Spec R §7; TR07.
7. **Misma inicialización con k distintos.** Misma semilla para todos los
   pares, `SEMILLA_INICIALIZACION` en `config.py`, separada de
   `SEMILLA_PARTICION`. Spec R §7; TR07.
8. **GD diverge con el par elegido.** η se divide por 2 desde la misma
   inicialización, hasta `MAX_REDUCCIONES_ETA` veces (default 3, en
   `config.py`); cada intento se registra; si sigue divergiendo, se registra
   como falla de GD y el veredicto no cambia. Spec R §5; TR08 (b).
9. **Concentración con partición.** La partición sirve solo para elegir
   (k, λ); elegido el par, se reentrena con ALS sobre todo Ω, y sobre ese
   modelo se evalúan los criterios 3 y 4 (el 2, sobre la partición). Spec R
   §7; TR08 (a), TR09.
10. **CA-R06.** "Los 55 tests de main siguen pasando sin modificaciones y el
    único test salteado sigue siendo el de la V₀ del informe". Spec R §10;
    reglas de trabajo y TR10.
11. **Tolerancia de CA-R03.** `np.allclose` con rtol = 1e-10, sobre una
    matriz chica y bien condicionada. Spec R §10; TR01.
12. **CLAUDE.md.** Regla 1 sin el guion de más; regla 3 habla de "función a
    minimizar f (la SCE en main; la SCE más el término λ en esta rama)".
    Commiteado junto con las dos specs.
