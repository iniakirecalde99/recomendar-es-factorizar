# Tareas — Experimento de regularización (rama experimento/regularizacion)

Derivado de `specs/regularizacion.md` (en adelante "spec R"). **No se empieza
TR01 hasta resolver las siete preguntas pendientes del final.** Las
preguntas 1, 3, 8, 9 y 12 ya están resueltas y volcadas en la spec R y en
CLAUDE.md (ver "Preguntas resueltas").

Reglas de trabajo (las de CLAUDE.md, aplicadas a esta rama):
- Una tarea por vez: primero el test, verlo fallar, después el código mínimo,
  después `pytest -q` completo. Una tarea no está terminada si falla algún
  test, incluidos los de main (CA-R06 se verifica en todas las tareas).
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

### [ ] TR01 — λ en `resolver_factor`
**Objetivo:** agregar el parámetro λ (default 0) a `resolver_factor` y
resolver por fila (V_iᵀ V_i + λ I) U_iᵀ = V_iᵀ r_i con `np.linalg.solve`,
como en main (spec R §4). La misma función sigue resolviendo el paso de V
con Rᵀ y Mᵀ.
**Archivos:** `src/als.py`.
**Cierra:** CA-R02, CA-R03.
**Test que se escribe primero:**
- `test_als.py::test_resolver_factor_con_lambda_satisface_las_ecuaciones_normales_regularizadas`
  (CA-R03: matriz chica a mano, verificar (Vᵀ V + λ I) u = Vᵀ r con tolerancia,
  ver pregunta 11).
- `test_als.py::test_resolver_factor_con_lambda_positivo_no_es_singular_con_menos_de_k_calificaciones`
  (CA-R02: usuario con 1 calificación y k = 2, λ > 0 → sin `SistemaSingularError`).
- `test_als.py::test_resolver_factor_con_lambda_cero_es_identico_al_actual`
  (igualdad exacta con la llamada sin λ en una matriz chica).

### [ ] TR02 — f regularizada (spec R §3)
**Objetivo:** f(U, V) = SCE sobre Ω + λ (Σ‖U_i‖² + Σ‖V_j‖²) en el módulo
compartido, con λ = 0 igual a la SCE de main. Forma (función nueva o
parámetro de `sce`): ver pregunta 2.
**Archivos:** `src/modelo.py`.
**Cierra:** ninguno (soporte de CA-R01 y CA-R04).
**Test que se escribe primero:**
- `test_modelo.py::test_f_regularizada_con_lambda_cero_es_la_sce` (igualdad exacta).
- `test_modelo.py::test_f_regularizada_suma_lambda_por_normas_al_cuadrado`
  (valor calculado a mano en una matriz de 2 × 2).
- `test_modelo.py::test_f_regularizada_ignora_valores_fuera_de_omega`
  (como CA-02, con λ > 0).

### [ ] TR03 — λ en `gradiente_sce`
**Objetivo:** ∇_U f = −2 E V + 2 λ U, ∇_V f = −2 Eᵀ U + 2 λ V (spec R §5),
con λ default 0; los dos gradientes se siguen calculando sobre el mismo (U, V).
**Archivos:** `src/gradiente.py`.
**Cierra:** CA-R04.
**Test que se escribe primero:**
- `test_gd_factorizacion.py::test_gradiente_con_lambda_coincide_con_diferencias_finitas`
  (CA-R04: con la f de TR02, λ > 0, error relativo < 1e-5 como CA-07).
- El test actual de CA-07 sigue pasando sin cambios (λ = 0).

### [ ] TR04 — λ en `entrenar_als` y `entrenar_gd`, y garantía de main
**Objetivo:** pasar λ (default 0) a los dos entrenamientos (spec R §11): ALS
usa `resolver_factor(..., λ)`, GD usa `gradiente_sce(..., λ)`, y ambos cortan
con |f(t+1) − f(t)| < ε usando la f de TR02. `DivergenciaError` igual que en
main.
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

### [ ] TR05 — Partición entrenamiento/prueba
**Objetivo:** después del filtro por umbral, mandar al azar una fracción de Ω
a prueba con un `np.random.Generator` de semilla fija, pasar a
entrenamiento todo par de prueba cuyo usuario o película quede sin
calificaciones de entrenamiento, y loguear el tamaño de cada conjunto
(spec R §6). Devuelve dos máscaras, M_ent y M_prueba; R no se toca.
`FRACCION_PRUEBA` (0,2) y `SEMILLA_PARTICION` van a `config.py`.
**Archivos:** `src/datos.py`, `src/config.py`.
**Cierra:** CA-R05, CA-R07 (parte de la partición).
**Test que se escribe primero:**
- `test_datos.py::test_particion_es_disjunta_y_su_union_es_omega`.
- `test_datos.py::test_particion_es_reproducible_con_la_semilla`.
- `test_datos.py::test_particion_pasa_a_entrenamiento_los_pares_que_dejarian_huerfano_a_un_usuario_o_pelicula`
  (matriz chica armada para que el sorteo deje huérfana a una película).

## 2. Experimento (experimentos/, fuera de src/)

### [ ] TR06 — Métricas y criterios de éxito
**Objetivo:** en `experimentos/regularizacion.py`, funciones puras para:
SCE sobre Ω_prueba (reusando `sce` con M_prueba), porcentaje de estimaciones
fuera de [ESCALA_MIN − 1, ESCALA_MAX + 1] y máximo de |r̂| sobre los pares
fuera de Ω (spec R §8), y `evaluar_criterios` contra la spec R §9. Recibe
por separado lo que sale de la partición (criterio 2: SCE de prueba del par
elegido y de (k = 2, λ = 0)) y lo que sale del modelo reentrenado sobre
todo Ω (criterios 3 y 4), según spec R §7. Constantes nuevas en
`config.py`: `ESCALA_MIN`, `ESCALA_MAX` (ver pregunta 4) y los umbrales de
§9 (congelados desde acá, ver arriba).
**Archivos:** `experimentos/regularizacion.py`, `src/config.py`.
**Cierra:** CA-R07 (parte de rango y umbrales).
**Test que se escribe primero:**
- `test_experimento_regularizacion.py::test_fuera_de_rango_cuenta_solo_pares_fuera_de_omega`.
- `test_experimento_regularizacion.py::test_evaluar_criterios_falla_si_k_es_2`,
  `..._falla_si_la_sce_de_prueba_no_mejora_a_k2_lambda0`,
  `..._falla_con_mas_de_1_por_ciento_fuera_de_rango`,
  `..._falla_con_menos_de_130_peliculas_o_mas_de_25_por_ciento`,
  `..._exitoso_si_cumple_todo` (resultados sintéticos).

### [ ] TR07 — Barrido de (k, λ) y selección
**Objetivo:** para cada par de la grilla (`GRILLA_K` = {2, 3, 5, 10},
`GRILLA_LAMBDA` = {0, 1, 5, 10, 20} en `config.py`), ALS sobre Ω_ent desde la
inicialización de la semilla de config (ver pregunta 7), registrando
iteraciones, f final, SCE sobre Ω_prueba y fuera de rango (spec R §7). Un
par que falla (`SistemaSingularError`) queda en la tabla con el error, no
corta el barrido. Selección: menor SCE de prueba, con la regla de empate
del 1 % (ver pregunta 6). CLI `python -m experimentos.regularizacion` con
todo como argumentos (defaults de config).
**Archivos:** `experimentos/regularizacion.py`, `src/config.py`.
**Cierra:** CA-R07.
**Test que se escribe primero:**
- `test_experimento_regularizacion.py::test_barrido_devuelve_una_fila_por_par_de_la_grilla`
  (matriz chica, grilla chica pasada por argumento).
- `test_experimento_regularizacion.py::test_barrido_registra_el_error_de_un_par_singular_y_sigue`.
- `test_experimento_regularizacion.py::test_seleccion_elige_menor_sce_de_prueba`
  y `..._con_empate_menor_al_1_por_ciento_gana_el_menor_k` (filas sintéticas).
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
iteraciones, SCE de prueba, fuera de rango), el tamaño de la partición, el
par elegido con la aclaración de que su SCE de prueba es optimista, la
comparación con GD (y cada intento de η, o la falla), la concentración y el
veredicto contra la spec R §9 (criterio 2 sobre la partición; 3 y 4 sobre
el modelo reentrenado). Si falla algún criterio se registra igual y la rama
no se integra.
**Archivos:** `specs/bitacora.md`.
**Cierra:** CA-R08, CA-R06 (verificación final: todos los tests de main
siguen pasando).
**Test que se escribe primero:** ninguno nuevo (tarea de corrida y
registro); antes de correr, `pytest -q` completo en verde.

## Preguntas resueltas (decisión y dónde quedó)

1. **Solver.** main usa `np.linalg.solve`, permitido por la regla 5 (prohíbe
   inv, lstsq y pinv). Spec R §4 corregida; TR01 lo usa.
3. **λ en `entrenar_als` y `entrenar_gd`.** Sí reciben λ; duplicar los
   bucles en `experimentos/` haría que CA-R01 no pruebe el código real.
   Spec R §11 corregida; TR04 lo implementa.
8. **GD diverge con el par elegido.** η se divide por 2 desde la misma
   inicialización, hasta `MAX_REDUCCIONES_ETA` veces (default 3, en
   `config.py`); cada intento se registra; si sigue divergiendo, se registra
   como falla de GD y el veredicto no cambia. Spec R §5; TR08 (b).
9. **Concentración con partición.** La partición sirve solo para elegir
   (k, λ); elegido el par, se reentrena con ALS sobre todo Ω, y sobre ese
   modelo se evalúan los criterios 3 y 4 (el 2, sobre la partición). Spec R
   §7; TR08 (a), TR09.
12. **CLAUDE.md.** Regla 1 sin el guion de más; regla 3 habla de "función a
    minimizar f (la SCE en main; la SCE más el término λ en esta rama)".
    Commiteado junto con las dos specs.

## Preguntas pendientes (necesito respuesta antes de TR01)

2. **Dónde vive la f de la sección 3.** `sce` está en `src/modelo.py`, que
   según la regla 3 es lo único compartido entre ALS y GD, junto con la
   inicialización y la predicción. ¿Se agrega λ como parámetro de `sce`
   (default 0) o se crea una función aparte, por ejemplo `f_regularizada`, y
   `sce` queda como la SCE pura que se usa para medir sobre Ω_prueba? Con
   λ > 0 ya no es una "suma de cuadrados del error", por eso propongo la
   función aparte.
4. **Rango de §8.** El dataset va de 0,5 a 5. ¿`ESCALA_MIN` es 0,5 (rango
   permitido [−0,5; 6]) o 1, la escala de la página (rango [0; 6])?
5. **"Fuera de Ω" en §8 con partición.** ¿Los pares fuera de Ω son los que no
   están ni en Ω_ent ni en Ω_prueba (no observados), o también cuentan los de
   prueba (todo lo que no es Ω_ent)? Propongo lo primero, porque los de
   prueba ya se miden con la SCE de prueba.
6. **Regla de empate del 1 % (§7).** ¿Se compara cada par contra el mejor,
   y entre todos los que quedan a menos del 1 % de la SCE de prueba mínima
   gana el de menor k? Si empatan en k, ¿gana el de menor SCE o el de menor
   λ?
7. **"Desde la misma inicialización" (§7) con k distintos.** U₀ y V₀ cambian
   de forma con k. ¿Alcanza con "la misma semilla para todos los pares", o
   tiene que ser el mismo U₀, V₀ para todos los λ de un mismo k? Con una
   misma semilla, las dos cosas coinciden para un k dado.
10. **CA-R06 "55 y 1 salteado".** Con los tests nuevos el total va a subir.
    ¿Lo interpreto como "los 55 tests de main siguen pasando y el único
    salteado sigue siendo el de la V₀", sin importar el total?
11. **Tolerancia de CA-R03.** No está fijada. Propongo `np.allclose` con
    rtol = 1e-10, porque `np.linalg.solve` resuelve un sistema de k × k bien
    condicionado. ¿Está bien?
