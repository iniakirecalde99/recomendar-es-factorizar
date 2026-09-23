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
