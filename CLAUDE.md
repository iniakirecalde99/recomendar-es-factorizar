# Recomendar es factorizar — demo de código

Demo del trabajo de promoción de Matemática 4 (Facultad de Informática, UNLP).
El código implementa la factorización R ≈ U·Vᵀ sobre MovieLens latest-small con dos
métodos distintos: ALS (mínimos cuadrados alternados) y descenso de gradiente.
El código es la contraparte ejecutable de un informe escrito: cada fórmula
implementada tiene que coincidir con la del informe.

## Flujo de trabajo (SDD)
- La fuente de verdad es `specs/spec.md`. El diseño está en `specs/plan.md` y
  las tareas en `specs/tasks.md`.
- No implementar nada que no esté en una tarea de `tasks.md`. Si hace falta algo
  fuera de la spec, frenar y preguntar; no improvisar.
- No modificar `spec.md` sin aprobación explícita.
- Una tarea por vez: primero el test, después el código, después correr toda la
  suite. Una tarea no está terminada si algún test falla.
- Al cerrar una tarea con todos los tests en verde, commiteá solo los
  archivos que la tarea lista, con el mensaje "Txx: <objetivo>". No
  commitees nada fuera de una tarea sin preguntar.

## Stack
- Python 3.11+, numpy, matplotlib (solo gráficos de la demo), pytest.
- Prohibido usar librerías que factoricen o recomienden (Surprise, implicit,
  scikit-learn, etc.). Todo el álgebra se escribe con numpy.
- No agregar dependencias sin preguntar.

## Reglas matemáticas (no negociables)
1. main no usa regularización. En la rama experimento/regularizacion se permite el término λ definido en specs/regularizacion.md, y nada más.
   - En la rama experimento/sesgos se permiten además los sesgos por usuario y por
     película, k > 2 y la métrica precisión@10, según specs/sesgos.md.
2. Los huecos de R se representan con `np.nan`. El error se calcula SOLO sobre Ω
   usando la máscara M (booleana, M[i,j] = True si (i,j) ∈ Ω). Nunca rellenar
   huecos con ceros ni con promedios.
3. ALS y descenso de gradiente viven en módulos separados y no comparten lógica
   de actualización. Solo comparten: carga de datos, inicialización de U y V, predicción
   U @ V.T y función a minimizar f (la SCE en main; la SCE más el término λ
   en experimento/regularizacion, specs/regularizacion.md §3; la f con sesgos
   de specs/sesgos.md §2 en experimento/sesgos).
4. ALS: el paso de V es la MISMA función que el paso de U, llamada con R.T y
   M.T (relación inversa, informe 3.1 y 5.2). No escribir una segunda función
   para V.
5. ALS resuelve cada fila con las ecuaciones normales de mínimos cuadrados
   (informe 3.5) usando `np.linalg.solve`. Prohibido `np.linalg.inv`,
   `np.linalg.lstsq` y `np.linalg.pinv` (introducen conceptos fuera del programa).
6. Descenso de gradiente: versión completa (no estocástica). Los gradientes
   respecto de U y de V se calculan ambos con los valores de la iteración t y
   se actualizan juntos. Actualizar U y después calcular el gradiente de V con la
   U nueva lo convierte en un método alternado: está prohibido.
7. Notación del informe en el código: R, M (por M_Ω), U, V, k, eta (η),
   epsilon (ε). Condición de corte: |f(t+1) − f(t)| < epsilon.
8. Las dimensiones latentes emergen del entrenamiento; nunca se etiquetan ni se
   inicializan a mano con significado.

## Calidad de código
- Sin valores hardcodeados: hiperparámetros (k, eta, epsilon, max_iter, semilla,
  escala de inicialización) como argumentos con defaults definidos en
  `src/config.py`.
- Type hints y docstrings en todas las funciones públicas. Docstrings en español.
- Nombres de funciones y variables en español, salvo la notación matemática.
- Donde el código implementa una fórmula del informe, comentar la sección:
  `# Informe 5.2: paso de U por mínimos cuadrados`.
- Errores: excepciones propias (`src/errores.py`) con mensajes que digan qué
  falló y dónde (por ejemplo, qué fila produjo un sistema singular). Nunca
  `raise Exception` ni `except Exception` genérico.
- `logging` en los módulos (INFO para progreso por iteración, DEBUG para
  detalle, WARNING para cortes por max_iter o filas filtradas). `print` solo en
  el script de la demo.
- Toda aleatoriedad pasa por un `np.random.Generator` creado a partir de una
  semilla recibida como argumento.

## Datos
- El dataset se llama MovieLens en todo el código y los mensajes.
- No commitear el dataset (la licencia de GroupLens no permite redistribuirlo).
  `data/` va en .gitignore; se descarga con el script de la spec.

## Comandos
- Tests: `pytest -q`
- Demo: `python -m src.demo --help`