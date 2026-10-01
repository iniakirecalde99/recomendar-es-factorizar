# Bitácora — implementación T11 a T14

Registro de la ejecución autónoma de T11 a T14 (`specs/tasks.md`): rojo,
verde y decisiones de diseño no explícitas en `plan.md`/`spec.md` que hizo
falta tomar para escribir el código.

## T11 — `entrenar_gd` y reproducibilidad

### Rojo

```
ImportError: cannot import name 'entrenar_gd' from 'src.gradiente'
(dos errores de colección: tests/test_gd_factorizacion.py y
tests/test_reproducibilidad.py)
```

### Verde

```
....s...s...............                                                 [100%]
=========================== short test summary info ===========================
SKIPPED [1] tests\test_als.py:83: V0 del ejemplo de la sección 5 del informe todavía no está fijado (spec.md §12)
SKIPPED [1] tests\test_datos.py:45: requiere el dataset real de MovieLens en .../data/ml-100k (correr scripts/descargar_movielens.py)
22 passed, 2 skipped in 0.24s
```

### Decisiones de diseño

- `test_iteracion_gd_es_simultanea_contra_referencia`: para forzar
  exactamente una iteración (y así comparar contra la referencia calculada
  a mano en el test) se usa `max_iter=1` — con `max_iter=1` el bucle corre
  una sola vez sin importar `epsilon`, porque la condición del `while` es
  `n_iteraciones < max_iter`.
- Para el test de reproducibilidad de GD (`eta=0.01`, `epsilon=1e-8`,
  `max_iter=300`) no hace falta que converja: alcanza con que las dos
  corridas, con la misma semilla, hagan exactamente lo mismo (incluso si
  divergiera, ambas divergirían igual).

## T12 — Tabla de comparación y gráfico de convergencia

### Rojo

```
ModuleNotFoundError: No module named 'src.comparacion'
1 error in 0.25s
```

### Verde

```
....s......s...............                                              [100%]
=========================== short test summary info ===========================
SKIPPED [1] tests\test_als.py:83: V0 del ejemplo de la sección 5 del informe todavía no está fijado (spec.md §12)
SKIPPED [1] tests\test_datos.py:45: requiere el dataset real de MovieLens en .../data/ml-100k (correr scripts/descargar_movielens.py)
25 passed, 2 skipped in 3.17s
```

### Decisiones de diseño

- Backend de matplotlib fijado a `"Agg"` con `matplotlib.use("Agg")` al
  principio de `src/comparacion.py`, antes de importar `pyplot` (instrucción
  explícita del usuario): la demo solo guarda el gráfico a archivo, nunca
  llama a `plt.show()`.
- `formatear_tabla_comparacion`: ancho de columnas fijo elegido a mano
  (10/12/18/12/14 caracteres) para que el encabezado y los valores queden
  alineados; no hay una fuente en la spec para estos anchos exactos, es una
  decisión de formato sin impacto en los criterios de aceptación.
- `graficar_convergencia` cierra la figura (`plt.close(fig)`) después de
  guardarla, para no acumular figuras abiertas si la demo llama a la
  función varias veces (no está en la spec, es higiene de matplotlib).

## T13 — Top-N y extremos por factor latente

### Rojo

```
ModuleNotFoundError: No module named 'src.recomendaciones'
1 error in 0.63s
```

### Verde

```
....s......s.................                                            [100%]
=========================== short test summary info ===========================
SKIPPED [1] tests\test_als.py:83: V0 del ejemplo de la sección 5 del informe todavía no está fijado (spec.md §12)
SKIPPED [1] tests\test_datos.py:45: requiere el dataset real de MovieLens en .../data/ml-100k (correr scripts/descargar_movielens.py)
27 passed, 2 skipped in 0.75s
```

### Decisiones de diseño

- `extremos_por_factor` devuelve, por columna, primero las "mayores" y
  después las "menores" (en ese orden dentro de la tupla), siguiendo el
  orden en que plan.md las menciona ("películas con mayor y con menor
  valor"); no había un orden más explícito que ese en la spec.
- Los índices de desempate en `np.argsort` (cuando hay valores iguales) los
  resuelve numpy por posición; no hay ningún criterio de desempate en la
  spec, así que no se fuerza ninguno.

## T14 — Demo completa

### Qué se alcanzó a hacer (en verde, sin commitear)

`test_datos.py::test_preparar_datos_movielens_integra_carga_filtro_y_titulos`
y `preparar_datos_movielens`/`DatosPreparados` en `src/datos.py` (la parte
de T14 que no depende de `config.py`). Rojo:

```
ImportError: cannot import name 'preparar_datos_movielens' from 'src.datos'
1 error in 0.57s
```

Verde (suite completa, con esto ya agregado):

```
....s......s..................                                           [100%]
=========================== short test summary info ===========================
SKIPPED [1] tests\test_als.py:83: V0 del ejemplo de la sección 5 del informe todavía no está fijado (spec.md §12)
SKIPPED [1] tests\test_datos.py:46: requiere el dataset real de MovieLens en .../data/ml-100k (correr scripts/descargar_movielens.py)
28 passed, 2 skipped in 0.75s
```

### Dónde frené y por qué

El siguiente test de T14 es
`test_demo.py::test_construir_parser_expone_los_argumentos_esperados_y_defaults`.
Según `plan.md` (`src/demo.py`), `construir_parser` define `--k`, `--eta`,
`--epsilon`, `--max-iter`, `--semilla`, `--usuario`, `--top-n` "con los
defaults de `config.py`". Pero:

- `src/config.py` **no existe** en el repo (verificado con un glob).
- Según `plan.md` §3, `K_DEFECTO`, `ETA_DEFECTO`, `EPSILON_DEFECTO`,
  `MAX_ITER_DEFECTO`, `SEMILLA_DEFECTO` y `ESCALA_INICIALIZACION_DEFECTO`
  son literalmente `...` (Ellipsis): están marcados **PENDIENTE**, a
  elegir "corriendo la demo a mano" (decisión 4 de `plan.md`, y sección 12
  de `spec.md`).
- Para correr la demo a mano hace falta el dataset real de MovieLens
  (`data/ml-100k/`), que no está presente en este entorno (por eso los
  tests `movielens` se saltean).
- Ninguna tarea del backlog actual (T05–T14, reescrito en esta misma
  conversación) crea `config.py` ni hace esa calibración — la tarea de
  calibración (`scripts/calibrar.py`) se sacó explícitamente del alcance a
  pedido del usuario.

Es circular: no puedo escribir `test_construir_parser_..._y_defaults` sin
antes tener valores numéricos en `config.py`, y esos valores solo se
pueden elegir corriendo la demo — que todavía no existe y que, aunque
existiera, no tiene con qué dataset correr acá. Inventar números yo mismo
violaría CLAUDE.md ("sin valores hardcodeados... con defaults definidos en
`src/config.py`") y la decisión 4 de `plan.md`, que dice explícitamente que
esos valores no se calibran por tarea aparte sino corriendo la demo a mano.
No lo resolví por mi cuenta: freno acá.

No toqué `src/demo.py` ni `src/config.py`. `T14` queda `[ ]` en
`specs/tasks.md`.

### Qué necesito del usuario para seguir (una de estas)

1. Los valores numéricos de `K_DEFECTO`, `ETA_DEFECTO`, `EPSILON_DEFECTO`,
   `MAX_ITER_DEFECTO`, `SEMILLA_DEFECTO`, `ESCALA_INICIALIZACION_DEFECTO`
   (con el criterio para documentarlo en el comentario de `config.py`, como
   pide la decisión 4), aunque sea corriendo la demo a mano en otro lado.
2. Retomar una tarea de calibración (como la que se sacó de alcance), ahora
   que hay forma de correrla — requeriría reabrir `tasks.md`/`plan.md`.
3. Redefinir `construir_parser` para que esos argumentos sean obligatorios
   (sin default) en vez de leer `config.py` — esto cambia lo que dice
   `plan.md` sección "`src/demo.py`" y necesita aprobación explícita antes
   de tocar la spec.

### Destrabe (usuario, mensaje siguiente)

El usuario destrabó T14 dando explícitamente los defaults provisorios para
`src/config.py`:

```
K_DEFECTO = 10, ESCALA_INICIALIZACION_DEFECTO = 1.0, ETA_DEFECTO = 1e-4,
EPSILON_DEFECTO = 1e-2, MAX_ITER_DEFECTO = 1000, SEMILLA_DEFECTO = 42,
TOP_N_DEFECTO = 10
```

Se creó `src/config.py` con esos siete valores, cada uno con un comentario
que empieza con `PROVISORIO: calibrar sobre MovieLens real.` seguido del
motivo del valor (no un criterio derivado de datos reales: son válidos
solo para que la demo corra, no para las conclusiones del informe — hay
que recalibrarlos corriendo la demo sobre MovieLens real).

### Rojo (resto de T14)

```
ImportError: cannot import name 'demo' from 'src'
1 error in 0.54s
```

### Verde

```
....s......s....................                                         [100%]
=========================== short test summary info ===========================
SKIPPED [1] tests\test_als.py:83: V0 del ejemplo de la sección 5 del informe todavía no está fijado (spec.md §12)
SKIPPED [1] tests\test_datos.py:46: requiere el dataset real de MovieLens en .../data/ml-100k (correr scripts/descargar_movielens.py)
30 passed, 2 skipped in 0.80s
```

### Decisiones de diseño

- **`main(argv)` no tiene parámetro para la ruta de datos** (la firma de
  plan.md es exactamente `main(argv: list[str] | None = None) -> None`, sin
  agregarle nada). Se resolvió con un constante de módulo
  `RUTA_DATOS_DEFECTO = Path("data/ml-100k")` (mismo lugar que usa
  `scripts/descargar_movielens.py`); el test de punta a punta usa
  `monkeypatch.setattr(demo, "RUTA_DATOS_DEFECTO", tmp_path)` para apuntar
  a los archivos chicos de `tests/`, sin tocar la firma de `main`. Mismo
  patrón para `RUTA_GRAFICO_DEFECTO` (dónde se guarda `convergencia.png`),
  que tampoco tiene un `--flag` propio en la spec.
- **`--metodo` no listado en `config.py`** (spec §10 no define un
  `METODO_DEFECTO`): se usa `default="ambos"`, alineado con el objetivo de
  la demo (spec §1: "mostrar en vivo la convergencia de ambos"). `--usuario`
  tampoco tiene constante en `config.py`; se usa `default=0` (primer
  usuario del conjunto filtrado, que siempre existe si no se disparó
  `ErrorDatosInsuficientes`).
- **`comparar_metodos` exige `resultado_als` y `resultado_gd`, ambos
  obligatorios** — no soporta un solo método. Cuando `--metodo` es `als` o
  `gd` (no `ambos`), `demo.py` arma una `FilaComparacion` de una sola fila
  a mano (`_fila_individual`, función nueva y propia de `demo.py`, no de
  `comparacion.py`) y se la pasa a `formatear_tabla_comparacion` (que sí
  acepta cualquier cantidad de filas). `graficar_convergencia` tampoco
  soporta un solo método (pide ambos resultados): con `--metodo != "ambos"`
  no se genera gráfico, solo con `--metodo ambos`.
- **Recomendaciones con `--metodo` distinto de `ambos`**: se arma con el
  resultado que haya corrido (ALS si corrió, si no GD). Con `--metodo
  ambos`, se usa el de ALS (no hay un criterio en la spec para elegir uno
  de los dos si ambos corrieron; ALS se eligió por ser el primero que
  describe el informe, ver spec §6 antes que §7).
- Verificado aparte (no forma parte de ningún test): `python -m src.demo
  --help` funciona en PowerShell nativo. Falló una sola vez con
  `UnicodeEncodeError` al correrlo desde la consola Bash de esta sesión,
  por la codepage cp1252 de esa consola en particular (el caracter "≈" de
  la descripción); no es un bug del código — no se tocó nada por esto.

### Estado final

Todo T11–T14 queda commiteado (ver commits de esta sesión), junto con
`specs/tasks.md` (T11–T14 en `[x]`) y este archivo. **Los defaults de
`src/config.py` siguen siendo provisorios** (comentario `PROVISORIO:
calibrar sobre MovieLens real` en cada uno): hace falta correrlos contra el
dataset real antes de usar los números que salgan de la demo para el
informe.

## T14 — Ajustes posteriores (5 puntos, aprobado modificar spec.md §10 y plan.md)

Aprobado explícitamente por el usuario: modificar `spec.md` §10 y `plan.md`
(sección `src/demo.py`, más las dos excepciones nuevas en la sección
`src/errores.py` y las tres constantes nuevas en `src/config.py`, por
consistencia con lo que ya describían esas secciones).

### Rojo

```
7 failed
- test_construir_parser_...: assert not True (args.metodo seguía existiendo)
- test_traducir_usuario_...: AttributeError: module 'src.demo' has no attribute '_traducir_usuario' (x2)
- test_formatear_top_n_lado_a_lado_...: AttributeError: no attribute '_formatear_top_n_lado_a_lado'
- test_main_..., test_main_propaga_..., test_main_captura_...: SystemExit: 2
  (argparse: unrecognized arguments: --datos ... --grafico ...)
```

### Verde

```
....s......s..........................                                   [100%]
=========================== short test summary info ===========================
SKIPPED [1] tests\test_als.py:83: V0 del ejemplo de la sección 5 del informe todavía no está fijado (spec.md §12)
SKIPPED [1] tests\test_datos.py:46: requiere el dataset real de MovieLens en .../data/ml-100k (correr scripts/descargar_movielens.py)
36 passed, 2 skipped, 2 warnings in 0.82s
```

(los 2 warnings son `RuntimeWarning: overflow encountered in square`,
esperados: son justamente los tests que fuerzan la divergencia con un eta
enorme.)

### Cambios, punto por punto

1. **`--usuario` recibe el id crudo.** `UsuarioNoEncontradoError` nueva en
   `errores.py`. `_traducir_usuario(id_usuario_a_indice, usuario_id)` nueva
   en `demo.py`, testeada aislada (sin correr `main`). `USUARIO_DEFECTO = 1`
   en `config.py` (sin comentario `PROVISORIO`: a diferencia de los
   hiperparámetros numéricos, no necesita calibrarse contra el dataset —
   si el `--k` pedido lo filtra, el error explica qué pasó).
2. **Sin `--metodo`.** `main` corre siempre `entrenar_als` y `entrenar_gd`.
   Se borró `_fila_individual` (ya no hace falta: `comparar_metodos` se usa
   siempre, con ambos resultados). Top-N: `_formatear_top_n_lado_a_lado`
   nueva, testeada aislada, arma dos columnas de texto plano (ALS | GD).
   Extremos por factor: solo `resultado_als.V`, con una línea aclarando en
   la salida que GD no se usa para esa parte (no había forma de "mezclar"
   extremos de dos factorizaciones distintas de manera significativa).
3. **`--datos` y `--grafico` en vez de constantes de módulo.** Antes
   `RUTA_DATOS_DEFECTO`/`RUTA_GRAFICO_DEFECTO` vivían en `demo.py` y el test
   de punta a punta las pisaba con `monkeypatch.setattr`; ahora son
   constantes de `config.py` usadas como default de `--datos`/`--grafico`
   (`type=Path`), y el test pasa las rutas de `tmp_path` directo por
   `argv`, sin monkeypatch.
4. **`sys.stdout.reconfigure(encoding="utf-8")`** como primera línea de
   `main` (antes de cualquier `print`). Motivo: la corrida manual anterior
   de `python -m src.demo --help` había fallado con `UnicodeEncodeError`
   en una consola Bash con codepage cp1252 (ver bitácora de T14, entrada
   anterior) — con esto, correrlo desde una consola así ya no debería
   fallar. Confirmado que no rompe `capsys` en los tests (el objeto que
   pytest pone en `sys.stdout` también soporta `.reconfigure()`).
5. **`main` captura `DivergenciaError`** alrededor de las dos llamadas de
   entrenamiento (`entrenar_als` y `entrenar_gd`) y hace `print(f"\nError:
   {error}")` sin relanzar — se pierde el resultado de ALS si ya había
   corrido, a cambio de un mensaje limpio en vez de traceback. Test nuevo
   (`test_main_captura_divergenciaerror_y_no_propaga_traceback`) usa el
   mismo truco que el test de `entrenar_gd` (T11): `eta=1e10` sobre la
   matriz chica de la demo.

### Decisiones de diseño adicionales (no pedidas explícitamente, pero necesarias para cerrar los 5 puntos)

- `UsuarioNoEncontradoError` NO se captura en `main` (a diferencia de
  `DivergenciaError`): se deja propagar sin envolver, igual que
  `SistemaSingularError`/`ErrorDatosInsuficientes` en el resto del código
  — el usuario pidió explícitamente capturar solo `DivergenciaError`.
- El ancho de columna de `_formatear_top_n_lado_a_lado` (40 caracteres) es
  una decisión de formato sin fuente en la spec, igual que los anchos de
  `formatear_tabla_comparacion` (T12).
- Se actualizó también la sección `src/errores.py` de `plan.md` (agregando
  `DivergenciaError`, que databa de la sesión anterior y nunca se había
  documentado ahí, y `UsuarioNoEncontradoError`) y la sección
  `src/config.py` (las tres constantes nuevas), para que `plan.md` no quede
  inconsistente con el código — no estaba en la aprobación explícita del
  usuario (que mencionó "spec.md §10 y plan.md (src/demo.py)"), pero
  dejarlas sin actualizar hacía que el propio texto de la sección
  `src/demo.py` de `plan.md` mencionara excepciones y constantes no
  documentadas en ningún otro lado del mismo archivo.

## Calibración de defaults sobre MovieLens real

Dos corridas de calibración hechas con scripts temporales, fuera de `src/`
y `tests/` (sin commitear), sobre `data/ml-100k/` real. Semilla fija = 42,
escala de inicialización = 1.0 en todas, usuario 1 (id crudo) para las
recomendaciones. Tablas completas tal como se le pasaron al usuario en el
chat, copiadas acá por pedido explícito.

### Corrida "k=10" (defaults originales de la demo + exploración de k/eta)

**Resumen del filtro (k=10, que en ese momento era también el umbral):**

```
WARNING:src.datos:filtro por k=10: se eliminaron 0 usuarios, 530 películas y 2047 calificaciones
MovieLens filtrado (k=10): 943 usuarios, 1152 películas, 97953 calificaciones.
```

**ALS (epsilon=1e-2, max_iter=1000) — primeras 5 y últimas 5 iteraciones:**

```
ALS iteración 1: f=70372.7          ALS iteración 996: f=46410.7
ALS iteración 2: f=60252.4          ALS iteración 997: f=46410.6
ALS iteración 3: f=56433.4          ALS iteración 998: f=46410.5
ALS iteración 4: f=54231.9          ALS iteración 999: f=46410.4
ALS iteración 5: f=52810            ALS iteración 1000: f=46410.4
WARNING:src.als: se alcanzó max_iter=1000 sin cortar por tolerancia
```

**GD (eta=1e-4, epsilon=1e-2, max_iter=1000) — primeras 5 y últimas 5 iteraciones:**

```
GD iteración 1: f=228991            GD iteración 996: f=59871.7
GD iteración 2: f=195636            GD iteración 997: f=59858.4
GD iteración 3: f=172057            GD iteración 998: f=59845.2
GD iteración 4: f=155357            GD iteración 999: f=59831.9
GD iteración 5: f=143347            GD iteración 1000: f=59818.7
WARNING:src.gradiente: se alcanzó max_iter=1000 sin cortar por tolerancia
```

**Tabla de comparación:**

| método | iteraciones | motivo de corte | tiempo (s) | SCE final |
|---|---:|---|---:|---:|
| ALS | 1000 | max_iter | 48.2639 | 46410.3539 |
| GD | 1000 | max_iter | 31.7897 | 59818.7316 |

**Top-10 ALS y GD lado a lado, usuario 1:**

| ALS | r̂ | GD | r̂ |
|---|---:|---|---:|
| That Old Feeling (1997) | 15785.72 | Secrets & Lies (1996) | 5.60 |
| I'll Do Anything (1994) | 10546.00 | Ran (1985) | 5.48 |
| Shooting Fish (1997) | 9114.86 | Harold and Maude (1971) | 5.24 |
| Little Odessa (1994) | 5766.62 | Lawrence of Arabia (1962) | 5.23 |
| Sum of Us, The (1994) | 3914.38 | Magnificent Seven, The (1954) | 5.16 |
| Cronos (1992) | 3183.21 | 8 1/2 (1963) | 5.15 |
| For Love or Money (1993) | 2795.84 | Down by Law (1986) | 5.11 |
| Rich Man's Wife, The (1996) | 2468.23 | Duck Soup (1933) | 5.10 |
| Wild Bill (1995) | 1974.99 | Hard Eight (1996) | 5.08 |
| Addiction, The (1995) | 1965.20 | Annie Hall (1977) | 5.06 |

**Extensión: max\|r̂\| dentro/fuera de Ω, k=10, resultado de ALS:**

| max\|r̂\| dentro de Ω | max\|r̂\| fuera de Ω |
|---|---|
| 6.4463 | 170967321.8578 |

Top-10 ALS usuario 1 (misma corrida) con cantidad de calificaciones de esa película:

| película | r̂ | n_calificaciones |
|---|---:|---:|
| That Old Feeling (1997) | 15785.72 | 11 |
| I'll Do Anything (1994) | 10546.00 | 10 |
| Shooting Fish (1997) | 9114.86 | 10 |
| Little Odessa (1994) | 5766.62 | 10 |
| Sum of Us, The (1994) | 3914.38 | 11 |
| Cronos (1992) | 3183.21 | 10 |
| For Love or Money (1993) | 2795.84 | 12 |
| Rich Man's Wife, The (1996) | 2468.23 | 15 |
| Wild Bill (1995) | 1974.99 | 12 |
| Addiction, The (1995) | 1965.20 | 11 |

10 filas de V (ALS) con mayor norma vs. cantidad de calificaciones de esa película:

| película | \|\|V_fila\|\| | n_calificaciones |
|---|---:|---:|
| That Old Feeling (1997) | 73588.28 | 11 |
| Fresh (1994) | 64077.32 | 10 |
| Clean Slate (1994) | 61664.03 | 10 |
| Ponette (1996) | 44083.15 | 10 |
| Wild Things (1998) | 35788.49 | 11 |
| Sum of Us, The (1994) | 28641.27 | 11 |
| Unhook the Stars (1996) | 25024.27 | 10 |
| I'll Do Anything (1994) | 23335.42 | 10 |
| White Man's Burden (1995) | 22381.27 | 10 |
| Bloodsport 2 (1995) | 20180.87 | 10 |

correlación(norma de fila de V, n_calificaciones) = **-0.1112** (casi nula:
la magnitud de la fila de V no depende de cuántas calificaciones tiene esa
película — el problema es otro: filas/columnas cerca del mínimo de
observaciones quedan sub-determinadas).

**ALS con k=2, 3, 5 (epsilon=1, max_iter=1000, mismo umbral=k=10 original
para el filtro, solo cambiando k):**

| k | usuarios | películas | iteraciones | motivo de corte | SCE final | max\|r̂\| fuera Ω |
|---:|---:|---:|---:|---|---:|---:|
| 2 | 943 | 1541 | 27 | tolerancia | 75127.8977 | 19971.2215 |
| 3 | 943 | 1473 | 89 | tolerancia | 69754.9064 | 107047.4924 |
| 5 | 943 | 1349 | — | **SistemaSingularError** (fila 1344, 5 observaciones) | — | — |

Top-10 usuario 1, k=2:

| película | r̂ |
|---|---:|
| Country Life (1994) | 1187.64 |
| Men With Guns (1997) | 894.09 |
| Cérémonie, La (1995) | 253.76 |
| Foreign Student (1994) | 212.79 |
| S.F.W. (1994) | 186.10 |
| Men of Means (1998) | 144.29 |
| Slingshot, The (1993) | 111.07 |
| Visitors, The (Visiteurs, Les) (1993) | 90.81 |
| Love and Death on Long Island (1997) | 82.05 |
| Joy Luck Club, The (1993) | 75.69 |

Top-10 usuario 1, k=3:

| película | r̂ |
|---|---:|
| Fausto (1993) | 4655.08 |
| Designated Mourner, The (1997) | 661.60 |
| Savage Nights (Nuits fauves, Les) (1992) | 490.46 |
| It Takes Two (1995) | 311.62 |
| Prefontaine (1997) | 248.29 |
| Best Men (1997) | 210.54 |
| Rendezvous in Paris (Rendez-vous de Paris, Les) (1995) | 200.24 |
| Underneath, The (1995) | 124.46 |
| Boys Life (1995) | 82.88 |
| Grateful Dead (1995) | 60.71 |

**GD con k=2, eta=2e-4/5e-4/1e-3 (epsilon=1, max_iter=3000):**

| eta | iteraciones | motivo de corte | SCE final | iteraciones con f creciente | resultado |
|---:|---:|---|---:|---:|---|
| 2e-4 | 1002 | tolerancia | 75251.9766 | 0 | — |
| 5e-4 | 3000 | max_iter | 93681.1269 | 1491 | — |
| 1e-3 | — | — | — | — | **DivergenciaError** en la iteración 14 |

### Corrida "experimento de umbral" (umbral independiente de k)

ALS: epsilon=1, max_iter=1000. GD: eta=2e-4, epsilon=1, max_iter=3000.

| umbral | k | usuarios | películas | calificaciones | ALS iter | ALS motivo | ALS SCE | ALS max\|r̂\| fuera Ω | GD iter | GD motivo | GD SCE | GD max\|r̂\| fuera Ω |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---:|---:|
| 20 | 2 | 917 | 937 | 94443 | 13 | tolerancia | 70970.26 | 8.8834 | 956 | tolerancia | 71247.59 | 6.7561 |
| 20 | 3 | 917 | 937 | 94443 | 27 | tolerancia | 66362.70 | 10.3223 | 1261 | tolerancia | 67086.13 | 6.5461 |
| 20 | 5 | 917 | 937 | 94443 | 70 | tolerancia | 59013.85 | 203.8911 | 1789 | tolerancia | 60491.63 | 9.6413 |
| 50 | 2 | 513 | 560 | 69222 | 11 | tolerancia | 50893.56 | 5.8295 | 711 | tolerancia | 51015.37 | 5.8523 |
| 50 | 3 | 513 | 560 | 69222 | 25 | tolerancia | 47940.72 | 6.4760 | 1143 | tolerancia | 48408.32 | 6.1028 |
| 50 | 5 | 513 | 560 | 69222 | 56 | tolerancia | 43561.83 | 11.9232 | 1439 | tolerancia | 44447.87 | 6.9047 |
| 100 | 2 | — | — | — | — | **ErrorDatosInsuficientes** | — | — | — | **ErrorDatosInsuficientes** | — | — |
| 100 | 3 | — | — | — | — | **ErrorDatosInsuficientes** | — | — | — | **ErrorDatosInsuficientes** | — | — |
| 100 | 5 | — | — | — | — | **ErrorDatosInsuficientes** | — | — | — | **ErrorDatosInsuficientes** | — | — |

(umbral=100 vacía toda la matriz en la cascada del filtro: no queda ningún
usuario ni película con al menos 100 calificaciones.)

Combinación elegida (menor umbral, luego menor k, con max\|r̂\| fuera de Ω
< 7 para ALS y GD a la vez): **umbral=50, k=2**.

Top-10 ALS, usuario 1 (umbral=50, k=2):

| película | r̂ |
|---|---:|
| Close Shave, A (1995) | 4.90 |
| Casablanca (1942) | 4.86 |
| Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1963) | 4.85 |
| Rear Window (1954) | 4.77 |
| Chinatown (1974) | 4.76 |
| Secrets & Lies (1996) | 4.76 |
| Third Man, The (1949) | 4.75 |
| One Flew Over the Cuckoo's Nest (1975) | 4.71 |
| Ran (1985) | 4.71 |
| Lawrence of Arabia (1962) | 4.69 |

Top-10 GD, usuario 1 (umbral=50, k=2):

| película | r̂ |
|---|---:|
| Close Shave, A (1995) | 4.89 |
| Casablanca (1942) | 4.87 |
| Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1963) | 4.85 |
| Rear Window (1954) | 4.77 |
| Third Man, The (1949) | 4.74 |
| Chinatown (1974) | 4.72 |
| Secrets & Lies (1996) | 4.71 |
| One Flew Over the Cuckoo's Nest (1975) | 4.69 |
| Lawrence of Arabia (1962) | 4.69 |
| Manchurian Candidate, The (1962) | 4.68 |

### Conclusión

- **Tener al menos k observaciones por fila es necesario, pero no
  suficiente.** Con el diseño original (`umbral == k`), incluso con k=10
  (bien por encima del mínimo teórico) las predicciones de ALS fuera de Ω
  explotan (~1.7×10⁸ en el peor caso, top-10 del usuario 1 en el orden de
  los miles); con k=5 aparece directamente `SistemaSingularError`. La
  correlación entre la norma de una fila de V y la cantidad de
  calificaciones de esa película es prácticamente nula (-0.11): no es que
  "más calificaciones" arregle la norma, es que estar cerca del mínimo dejaba
  el sistema mal condicionado.
- **Separar `umbral` de `k` y subir el umbral bien por encima de k lo
  arregla.** Con `umbral=50` y `k=2` (bastante margen sobre el mínimo),
  max\|r̂\| fuera de Ω baja de miles/millones a ~5.8 para ALS y GD, en el
  rango real de las calificaciones (1 a 5). `umbral=20` todavía da >7 para
  algún k; `umbral=100` es demasiado (vacía la matriz).
- **`eta=2e-4` es el mejor de los tres probados para GD con k=2:** `5e-4`
  no converge en 3000 iteraciones (f sube 1491 veces) y `1e-3` diverge
  (`DivergenciaError`) en la iteración 14.
- **Valores definitivos** (`src/config.py`): `UMBRAL_DEFECTO=50`,
  `K_DEFECTO=2`, `ETA_DEFECTO=2e-4`, `EPSILON_DEFECTO=1.0`,
  `MAX_ITER_DEFECTO=3000`, `SEMILLA_DEFECTO=42`,
  `ESCALA_INICIALIZACION_DEFECTO=1.0`.

## Calibración: umbral separado de k y defaults definitivos (cambios de código)

Aprobado por el usuario modificar `spec.md` y `plan.md` para reflejar la
corrección de la decisión 5 (umbral separado de k).

### Punto 1 — `filtrar_por_minimo` recibe `umbral`; `UmbralInsuficienteError`; `--umbral` en la demo

Rojo (`tests/test_datos.py`, tres tests con la firma nueva antes de tocar
`src/datos.py`):

```
TypeError: filtrar_por_minimo() got an unexpected keyword argument 'umbral' (x3)
TypeError: preparar_datos_movielens() got an unexpected keyword argument 'umbral'
4 failed, 3 passed
```

Verde (`tests/test_datos.py`): `7 passed`.

Rojo (`tests/test_demo.py`, con `--umbral` en los argv antes de tocar
`src/demo.py`):

```
SystemExit: 2 (unrecognized arguments: --umbral 2)  — x4 tests
4 failed, 3 passed
```

Verde (`tests/test_demo.py`): `7 passed, 1 warning` (el warning es el de
siempre, del test de divergencia).

### Punto 2 — `config.py` con valores definitivos

Sin test propio (son constantes; ya las ejercitan `test_construir_parser_...`
y todos los tests que llaman a `entrenar_als`/`entrenar_gd` con estos
valores). Verificado con la suite completa.

### Punto 3 — logging: DEBUG por iteración, INFO cada 100 + resumen final

Rojo (`tests/test_als.py` y `tests/test_gd_factorizacion.py`, tests nuevos
antes de tocar `src/als.py`/`src/gradiente.py`):

```
assert len(mensajes_debug) == 250
AssertionError: assert 0 == 250  (todo seguía en INFO por iteración, sin DEBUG)
```

Verde (`tests/test_als.py tests/test_gd_factorizacion.py`): `9 passed, 1 skipped, 1 warning`.

### Suite completa (los 5 puntos ya implementados)

```
....s....................................                                [100%]
=========================== short test summary info ===========================
SKIPPED [1] tests\test_als.py:85: V0 del ejemplo de la sección 5 del informe todavía no está fijado (spec.md §12)
40 passed, 1 skipped, 2 warnings in 0.98s
```

### Decisiones de diseño

- `ErrorDatosInsuficientes` e `UsuarioNoEncontradoError` cambiaron el
  nombre de su atributo/parámetro de `k`/mención de `--k` a `umbral`/
  `--umbral`, porque conceptualmente siempre fueron sobre el umbral del
  filtro, no sobre la dimensión latente — ahora que son parámetros
  distintos, usar el nombre viejo hubiera sido confuso. No lo pidió el
  usuario explícitamente, pero mantenerlo mal habría dejado dos
  excepciones con atributos que dicen "k" refiriéndose en realidad al
  umbral.
- `UmbralInsuficienteError` valida `umbral < k` al principio de
  `filtrar_por_minimo`, antes de tocar la matriz — falla rápido con un
  mensaje claro en vez de dejar que ALS explote más adelante con un
  `SistemaSingularError` menos informativo.
- El test de logging usa `epsilon=0.0` (no un número chico) para forzar
  exactamente `max_iter` iteraciones sin depender de que el algoritmo no
  converja por casualidad: `abs(diff) < 0.0` nunca es `True` porque
  `abs(...)` nunca es negativo, así que el corte por tolerancia queda
  inhabilitado de manera determinística.
- El resumen final de `entrenar_als`/`entrenar_gd` se loguea siempre (no
  solo cuando corta por `max_iter`): así el nivel INFO deja un rastro
  completo de cada corrida sin tener que subir a DEBUG.

## Recalibración sobre MovieLens latest-small (T23)

Mismo experimento que la calibración sobre 100K, repetido tras el cambio de
dataset (T20). Scripts temporales fuera de `src/` y `tests/` (sin
commitear), sobre `data/ml-latest-small/` real. Semilla 42, escala de
inicialización 1.0. ALS: epsilon=1, max_iter=1000. GD: eta=2e-4, epsilon=1,
max_iter=3000. Criterio, igual que antes: menor umbral y después menor k con
max|r̂| fuera de Ω < 7 para ALS y GD a la vez. Como latest-small es mucho más
disperso (9.724 películas para 610 usuarios), se probaron umbrales más bajos
que en 100K, y después se afinó entre 30 y 50.

Nota: la primera corrida se cortó con `_ArrayMemoryError` (2,7 MiB) porque
otro proceso de la máquina tenía tomada casi toda la memoria virtual de
Windows; no es un problema del código. Se repitió por partes.

| umbral | k | usuarios | películas | calificaciones | ALS iter | ALS SCE | ALS max\|r̂\| fuera Ω | GD iter | GD SCE | GD max\|r̂\| fuera Ω | año mediano, % ≥ 2000 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 2 | 609 | 2269 | 81109 | 27 | 46992.48 | 37.6676 | 1416 | 47539.44 | 7.1587 | 1997, 42 % |
| 10 | 3 | 609 | 2269 | 81109 | 78 | 42382.89 | 5006.0691 | 1926 | 43642.62 | 14.2253 | 1997, 42 % |
| 10 | 5 | 609 | 2269 | 81109 | 136 | 35272.49 | 268700.7511 | 2284 | 37293.07 | 11.1774 | 1997, 42 % |
| 20 | 2 | 566 | 1286 | 67020 | 21 | 39522.42 | 32.0125 | 1491 | 39957.39 | 6.3205 | 1997, 40 % |
| 20 | 3 | 566 | 1286 | 67020 | 38 | 35974.15 | 367.9705 | 1644 | 37045.56 | 10.4327 | 1997, 40 % |
| 20 | 5 | 566 | 1286 | 67020 | 96 | 30622.31 | 989.5806 | 1836 | 32181.10 | 9.3268 | 1997, 40 % |
| 30 | 2 | 435 | 827 | 52971 | 14 | 30992.45 | 8.5325 | 1416 | 31786.46 | 8.9711 | 1997, 40 % |
| 30 | 3 | 435 | 827 | 52971 | 47 | 28467.82 | 58.2797 | 1300 | 28996.37 | 7.1169 | 1997, 40 % |
| 30 | 5 | 435 | 827 | 52971 | 52 | 24651.24 | 138.7910 | 1482 | 25764.74 | 9.2326 | 1997, 40 % |
| 35 | 2 | 382 | 693 | 47024 | 13 | 27601.18 | 7.1834 | 1027 | 27953.26 | 6.4272 | 1997, 37 % |
| **40** | **2** | **321** | **534** | **38891** | **10** | **22937.39** | **6.3881** | **922** | **23191.62** | **5.8621** | 1997, 36 % |
| 50 | 2 | 207 | 302 | 23434 | 9 | 13367.34 | 5.4628 | 717 | 13639.46 | 5.4622 | 1997, 37 % |
| 50 | 3 | 207 | 302 | 23434 | 17 | 12293.65 | 17.5185 | 746 | 12965.39 | 7.3155 | 1997, 37 % |
| 50 | 5 | 207 | 302 | 23434 | 51 | 10627.39 | 22.8481 | 931 | 11407.25 | 6.5442 | 1997, 37 % |

Todas las corridas cortaron por tolerancia. Todas las combinaciones con k=3
o k=5 fallan el criterio (ALS amplifica |r̂| fuera de Ω); con k=2, umbral=40
es el menor que cumple.

**GD con umbral=40, k=2, eta=2e-4/5e-4/1e-3 (epsilon=1, max_iter=3000):**

| eta | iteraciones | motivo de corte | SCE final | iteraciones con f creciente | max\|r̂\| fuera Ω |
|---:|---:|---|---:|---:|---:|
| 2e-4 | 922 | tolerancia | 23191.62 | 0 | 5.8621 |
| 5e-4 | 462 | tolerancia | 23045.88 | 0 | 6.1228 |
| 1e-3 | — | — | — | — | **DivergenciaError** en la iteración 35 |

**Robustez de eta=5e-4 frente a la semilla (umbral=40, k=2):**

| semilla | GD iteraciones | f creciente | GD max\|r̂\| fuera Ω | ALS max\|r̂\| fuera Ω |
|---:|---:|---:|---:|---:|
| 1 | 511 | 0 | 7.24 | 6.46 |
| 2 | 755 | 0 | 6.57 | 6.15 |
| 3 | 554 | 0 | 5.65 | 6.23 |
| 7 | 432 | 0 | 5.92 | 6.16 |
| 123 | 430 | 0 | 6.07 | 6.15 |

(Con la semilla 1, GD queda apenas por encima de 7; el criterio se evalúa
con la semilla por defecto, 42.)

**Demo con los defaults nuevos (umbral=40, k=2, eta=5e-4):**

```
MovieLens filtrado (umbral=40, k=2): 321 usuarios, 534 películas, 38891 calificaciones.
ALS                 10        tolerancia      0.1198    22937.3886
GD                 462        tolerancia      2.0637    23045.8796
```

Top-10 ALS del usuario 1: Shawshank Redemption (5.25), Casablanca, Boondock
Saints (2000), Wallace & Gromit: The Wrong Trousers, Godfather, Dark Knight
(2008), Unforgiven, Amelie (2001), Life Is Beautiful, Lord of the Rings:
The Return of the King (2003). Sin regularización ni recorte, r̂ puede pasar
de 5, que es la calificación máxima.

### Conclusión

- **Valores definitivos** (`src/config.py`): `UMBRAL_DEFECTO=40`,
  `K_DEFECTO=2`, `ETA_DEFECTO=5e-4`; sin cambios `EPSILON_DEFECTO=1.0`,
  `MAX_ITER_DEFECTO=3000`, `SEMILLA_DEFECTO=42`,
  `ESCALA_INICIALIZACION_DEFECTO=1.0`. `tests/test_calibracion.py` verifica
  el criterio con esos defaults sobre el dataset real.
- **La misma lección que con 100K:** k=2 y un umbral bien por encima de k.
  Con más factores o menos observaciones, ALS amplifica r̂ fuera de Ω.
- **Películas nuevas:** el filtro se queda con las más calificadas, que en
  este dataset son mayormente de los 90 (año mediano 1997 en todos los
  umbrales; entre 36 % y 42 % del 2000 en adelante). Igual aparecen en las
  recomendaciones películas de 2000 a 2008, que con 100K (hasta 1998) no
  existían.

## Experimento de concentración de las recomendaciones con k = 2 (centrado descartado)

### Objetivo

Medir cuánto se concentran las recomendaciones con k = 2 (con el dataset
anterior se notaba que la página recomendaba Titanic casi siempre, aunque
se cambiaran las calificaciones) y probar si centrar las calificaciones
restando el promedio global μ lo corrige.

Criterio de adopción, fijado antes de correr: adoptar el centrado solo si la
cantidad de películas que aparecen en algún top-10 sube claramente respecto
de 65 y la más frecuente baja claramente del 46 %. Si la mejora es marginal,
se deja como limitación de k = 2 en la sección 8 del informe.

### Método

- Script: `experimentos/concentracion.py` (`python -m experimentos.concentracion`;
  parámetros como argumentos, con defaults de `config.py` y constantes del
  propio script). El centrado vive solo en ese script: `src/` no centra.
- Datos y modelo: MovieLens latest-small con los defaults de T23 (umbral=40,
  k=2, epsilon=1, semilla 42): 321 usuarios × 534 películas. ALS sobre R
  (sin centrado) o sobre R − μ (con centrado, μ = promedio de las
  calificaciones observadas en Ω = 3,679).
- Usuarios simulados: 3.000 (semilla de simulación 0). Cada uno califica
  entre 5 y 10 de las 30 películas más calificadas (las mismas que ofrece la
  página). Su vector u se resuelve con V fija (`resolver_factor`, informe 5),
  con sus notas − μ cuando hay centrado; la predicción es μ + Vⱼ·u. Se cuenta
  en qué top-10 (sin las ya calificadas) aparece cada película.
- Dos distribuciones de notas: uniforme (enteros 1 a 5 equiprobables,
  promedio 3) y real (notas de `ratings.csv` redondeadas hacia arriba a
  enteros, promedio 3,65). La variante real se agregó porque con notas
  uniformes todos los usuarios simulados quedan por debajo de μ y el
  centrado los trata como "exigentes", lo que sesga la comparación.

### Resultados

| notas | centrado | SCE ALS (iter) | max\|r̂\| fuera Ω | distintas en algún top-10 | más frecuente | frec. | películas en > 20 % de los top-10 | ángulo de u p5 / p50 / p95 |
|---|---|---:|---:|---:|---|---:|---:|---|
| uniforme | no | 22.937 (10) | 6,39 | 65 de 534 | Harry Potter and the Order of the Phoenix (2007) | 45,5 % | 19 | 6° / 44° / 89° |
| uniforme | sí (μ = 3,679) | 23.573 (8) | 6,03 | 77 de 534 | Batman & Robin (1997) | 52,4 % | 18 | −167° / −96° / 56° |
| real | no | 22.937 (10) | 6,39 | 65 de 534 | Harry Potter and the Order of the Phoenix (2007) | 44,5 % | 23 | 18° / 44° / 75° |
| real | sí (μ = 3,679) | 23.573 (8) | 6,03 | 80 de 534 | Nutty Professor, The (1996) | 38,4 % | 21 | −172° / 1° / 171° |

Top-5 más frecuentes de cada corrida:

- uniforme, sin centrado: Harry Potter and the Order of the Phoenix 46 %; 10 Things I Hate About You 43 %; The Patriot 42 %; The Chronicles of Narnia: The Lion, the Witch and the Wardrobe 38 %; Chinatown 33 %.
- uniforme, con centrado: Batman & Robin 52 %; The Flintstones 51 %; Wild Wild West 50 %; Fantasia 47 %; Mortal Kombat 47 %.
- real, sin centrado: Harry Potter and the Order of the Phoenix 45 %; 10 Things I Hate About You 37 %; The Patriot 34 %; How to Train Your Dragon 34 %; The Boondock Saints 33 %.
- real, con centrado: The Nutty Professor 38 %; Coneheads 38 %; The Maltese Falcon 35 %; Beverly Hills Cop III 35 %; Chinatown 35 %.

Referencia con el dataset anterior (MovieLens 100K, umbral=50, k=2, notas
uniformes, misma simulación): 63 de 560 películas en algún top-10; Titanic
en el 45,7 % de los top-10, sin estar entre las 30 para calificar (no se la
podía excluir calificándola).

### Decisión

**El centrado no se adopta.** Con la simulación pedida (notas uniformes), la
cantidad de películas distintas sube poco (65 → 77) y la más frecuente
empeora (45,5 % → 52,4 %). Con notas reales mejora algo (65 → 80; 44,5 % →
38,4 %), pero sigue siendo marginal: 80 de 534 películas es un 15 %, y 21
películas siguen apareciendo en el top-10 de más del 20 % de los usuarios.
Además, centrar empeora el ajuste: la SCE de ALS pasa de 22.937 a 23.573.

**La limitación es k = 2.** Con dos factores, cada película es un punto Vⱼ
del plano y la predicción r̂ⱼ = Vⱼ · u es su proyección sobre la dirección
de u, así que para cualquier u ganan las películas del borde exterior de la
nube de puntos. El centrado arregla otra causa, que u quede siempre en un
abanico angosto porque todas las notas son positivas (sin centrar, de 6° a
89°; centrado, casi toda la vuelta). Pero no cambia esa geometría: las
favoritas cambian, pero siguen siendo pocas. Subir k sin regularización no
es una salida (ver la recalibración T23: k = 3 y k = 5 hacen explotar r̂
fuera de Ω). Queda como limitación de k = 2 en la sección 8 del informe.
`src/` sigue sin centrado.
