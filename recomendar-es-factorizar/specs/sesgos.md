# Experimento: sesgos por usuario y por película (rama experimento/sesgos)

## 1. Contexto
Cuatro experimentos (bitácora) muestran que con k = 2 unas pocas películas
acaparan los top-10, también con usuarios reales (Casablanca en el 43,9 %).
Ni centrar, ni regularizar, ni la estimación relativa lo resolvieron. Este
experimento prueba el modelo con sesgos, que separa la popularidad de cada
película del gusto personal. main no se modifica. Nada entra al informe sin
aprobación de la cátedra.

## 2. Modelo
    r̂_ij = μ + b_i + c_j + U_i·V_j
μ: promedio de las calificaciones de entrenamiento de cada modelo, fijo (no
se entrena): Ω_ent en los modelos de la partición y todo Ω en el modelo
reentrenado.
b_i: sesgo del usuario i. c_j: sesgo de la película j.
Función a minimizar:
    f = Σ_{(i,j) ∈ Ω_ent} (r_ij − r̂_ij)²
        + λ (Σ_i ‖U_i‖² + Σ_j ‖V_j‖² + Σ_i b_i² + Σ_j c_j²)

## 3. ALS con sesgos
Paso de usuarios: incógnitas (U_i, b_i) ∈ R^(k+1). Se usa V aumentada con
una columna de unos, [V | 1], y el objetivo y_ij = r_ij − μ − c_j.
Paso de películas: incógnitas (V_j, c_j), con [U | 1] y objetivo
r_ij − μ − b_i.
Ambos pasos se resuelven con resolver_factor sin modificarla (spec R §4): la
misma función para los dos pasos, con la matriz aumentada y el objetivo
ajustado. Criterio de corte: el de main, con la f de la §2.
Esta rama no usa descenso de gradiente: todos los modelos con sesgos se
entrenan con ALS. La única referencia que no tiene sesgos es la línea de base
de la §7, que también es ALS (entrenar_als de main).

## 4. Datos y partición
La misma partición de la spec R §6 (SEMILLA_PARTICION, FRACCION_PRUEBA).

## 5. Barrido y selección
Grilla en config.py como GRILLA_K_SESGOS y GRILLA_LAMBDA_SESGOS (las
GRILLA_K y GRILLA_LAMBDA de la regularización no se tocan):
k ∈ {2, 5, 10, 20}, λ ∈ {1, 5, 10, 20}.
Para cada par: ALS sobre Ω_ent desde SEMILLA_INICIALIZACION (sesgos
iniciales en 0). Se registran iteraciones, SCE sobre Ω_prueba, fuera de rango
y precisión@10 con los dos ordenamientos de la §6.
Se elige el par con menor SCE sobre Ω_prueba. Sin regla de empate.
Con el par elegido se vuelve a entrenar sobre todo Ω (con μ de todo Ω), y
sobre ese modelo se mide la concentración y el fuera de rango. Los criterios
1, 2 y 4 de la §8 se evalúan sobre el modelo reentrenado; el criterio 3
(precisión@10), sobre el modelo de la partición, porque necesita Ω_prueba.

## 6. Ordenamientos
A: por r̂_ij completo.
B: por la parte personal, U_i·V_j (μ y b_i no cambian el orden de un
   usuario; se saca c_j).
Los dos se evalúan siempre. Ninguno se elige después de ver los resultados.

## 7. Métricas
- SCE sobre Ω_prueba (como en la spec R §8).
- Fuera de rango: como en la spec R §8, siempre sobre r̂ completo
  (μ + b_i + c_j + U_i·V_j), en los dos ordenamientos.
- Precisión@10: para cada usuario con al menos una calificación >= 4 en
  Ω_prueba, top-10 entre las películas que no tiene en Ω_ent; precisión =
  (películas del top-10 calificadas >= 4 en Ω_prueba) / 10. Se promedia sobre
  esos usuarios. Umbral 4 en config.py como UMBRAL_RELEVANTE.
- Línea de base de precisión: modelo de main (entrenar_als, k = 2, λ = 0,
  sin sesgos, ordenamiento absoluto) entrenado sobre la misma Ω_ent, con
  SEMILLA_INICIALIZACION, EPSILON_DEFECTO y MAX_ITER_DEFECTO.
- Concentración con los 321 usuarios reales, como el experimento 3: top-10
  entre las películas no calificadas, películas distintas, la más frecuente y
  su %, películas en más del 20 % de los top-10. Como referencia, fuera del
  criterio: usuarios simulados con notas reales. El vector de un usuario
  simulado se calcula con el mismo paso aumentado que un usuario real: u y b
  con [V | 1], objetivo r − μ − c, mismo λ.

## 8. Criterio de éxito (fijado antes de correr)
Umbrales en config.py, propios de esta rama (no se reusan los de la
regularización): MAX_FRECUENCIA_SESGOS = 0,25,
MIN_PELICULAS_DISTINTAS_SESGOS = 127 (sale del experimento 3) y
MAX_FRACCION_FUERA_DE_RANGO_SESGOS = 0,01.
Éxito si al menos uno de los ordenamientos A o B, con el par elegido, cumple
todo:
1. La más frecuente aparece en a lo sumo el 25 % de los top-10 de usuarios
   reales.
2. Al menos 127 películas distintas en algún top-10 (no empeora main).
3. Precisión@10 mayor o igual que la línea de base.
4. A lo sumo 1 % de estimaciones fuera de rango.
Una sola corrida. Si no cumple, se registra como está, sin cambiar grilla,
semillas, umbrales ni ordenamientos.

## 9. Criterios de aceptación
(El CA-S01 original, "con sesgos fijos en 0 el entrenamiento reproduce el
ALS regularizado", se eliminó: no hay camino de código para fijar sesgos. La
corrección la cubren el test de f con sesgos y μ en 0 == f_regularizada,
CA-S01 y CA-S02. Los demás se renumeraron.)
CA-S01 El paso aumentado da la misma solución que resolver directamente el
       sistema de k+1 incógnitas en una matriz chica (np.allclose, rtol 1e-10).
CA-S02 La f de la §2 no crece entre iteraciones de ALS.
CA-S03 Precisión@10 da el valor esperado en un caso chico calculado a mano.
CA-S04 Los dos ordenamientos coinciden con su definición en un caso chico
       (B no depende de c_j).
CA-S05 Los 55 tests de main y los de la rama anterior siguen pasando sin
       modificaciones.
CA-S06 Grilla, umbrales y UMBRAL_RELEVANTE en config.py. Nada hardcodeado.
CA-S07 La bitácora registra la tabla del barrido completa, el par elegido,
       la línea de base, las métricas de A y B, y el veredicto criterio por
       criterio.

## 10. Implementación
- Cambios en src/: los sesgos (μ, b, c) en un módulo nuevo, src/sesgos.py,
  que usa resolver_factor sin modificarla. El bucle de ALS con sesgos
  (alternar los dos pasos, f de la §2, corte por ε) también vive en
  src/sesgos.py. Nada más en src/.
- Barrido en experimentos/sesgos.py, con python -m experimentos.sesgos.
- Comentarios citan la sección de esta spec.