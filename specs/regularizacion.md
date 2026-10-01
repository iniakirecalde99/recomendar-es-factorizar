# Experimento: regularización (rama experimento/regularizacion)

## 1. Contexto
Con k = 2 las recomendaciones se concentran en pocas películas (bitácora,
experimento de concentración, commit 27348a0). Subir k sin regularización
produce estimaciones fuera de Ω de magnitud absurda. Este experimento prueba
si la regularización permite usar k > 2 y si eso reduce la concentración.
main no se modifica. Nada de esta rama entra al informe sin aprobación de la
cátedra.

## 2. Alcance
Incluye: partición entrenamiento/prueba, término λ en ALS y en descenso de
gradiente, barrido de (k, λ), medición de estimaciones fuera de rango y de
concentración.
No incluye: cambios en recomendador.html (sigue con k = 2), cambios en main,
dependencias nuevas, métricas distintas de la SCE.

## 3. Función a minimizar
f(U, V) = Σ_{(i,j) ∈ Ω_ent} (r_ij − U_i·V_j)² + λ (Σ_i ‖U_i‖² + Σ_j ‖V_j‖²)

Ω_ent: pares observados de entrenamiento. ‖U_i‖²: suma de los cuadrados de
la fila i de U (ídem V_j). λ ≥ 0. Con λ = 0, f es la SCE de main.

f se implementa en src/modelo.py como una función aparte, f_regularizada;
sce queda como la SCE pura (sin término λ) y es la que se usa para medir
sobre Ω_prueba. f_regularizada es la que usan los criterios de corte de
ALS y de descenso de gradiente. Con λ = 0, f_regularizada == sce.

## 4. ALS
Paso de U, para cada usuario i, se resuelve el sistema de k × k:
    (V_iᵀ V_i + λ I) U_iᵀ = V_iᵀ r_i
donde V_i son las filas de V de las películas que i calificó en Ω_ent, r_i
sus calificaciones e I la identidad de k × k.
- Se implementa agregando el parámetro λ a resolver_factor, con default 0.
  La misma función sigue resolviendo el paso de V con Rᵀ y M_Ωᵀ.
- Se resuelve con np.linalg.solve, como en main (sin inv, lstsq ni pinv).
- Con λ > 0 el sistema no es singular aunque el usuario tenga menos de k
  calificaciones.
- Criterio de corte: el mismo que en main, usando la f de la sección 3.

## 5. Descenso de gradiente
Con E = M ∘ (R − U Vᵀ), donde M es la matriz de Ω_ent y ∘ el producto
entrada a entrada (los huecos no aportan error):
    ∇_U f = −2 E V + 2 λ U
    ∇_V f = −2 Eᵀ U + 2 λ V
- Los dos gradientes se calculan antes de actualizar (actualización
  simultánea, como en main).
- Corte: |f(t+1) − f(t)| < ε, con f = f_regularizada (sección 3).
- η, ε y MAX_ITER salen de config.py. Si f deja de ser finita, se lanza
  DivergenciaError como en main.
- Si con el par elegido f deja de ser finita, η se divide por 2 y se vuelve
  a empezar desde la misma inicialización, hasta MAX_REDUCCIONES_ETA veces
  (default 3, en config.py). Cada intento se registra. Si sigue divergiendo,
  se registra como falla del descenso de gradiente; el veredicto de la
  sección 9 no cambia, porque se evalúa con ALS.

## 6. Partición
- Se aplica después del filtro por umbral.
- FRACCION_PRUEBA de Ω (default 0,2) va al azar a prueba y el resto a
  entrenamiento, con SEMILLA_PARTICION fija en config.py.
- Un par de prueba cuyo usuario o película queda sin calificaciones en
  entrenamiento se pasa a entrenamiento.
- Todas las corridas usan la misma partición. Se registra el tamaño de cada
  conjunto.

## 7. Barrido y selección
- Grilla en config.py: k ∈ {2, 3, 5, 10}, λ ∈ {0, 1, 5, 10, 20}.
- Para cada par (k, λ): ALS sobre Ω_ent desde la misma inicialización: U₀
  y V₀ se generan con SEMILLA_INICIALIZACION (en config.py, separada de
  SEMILLA_PARTICION), la misma para todos los pares; así todos los λ de un
  mismo k arrancan del mismo U₀, V₀. Se registran iteraciones, f final, SCE
  sobre Ω_prueba y estimaciones fuera de rango.
- Se elige el par con menor SCE sobre Ω_prueba, con esta regla de empate:
  entre todos los pares cuya SCE de prueba está a menos del 1 % de la mejor,
  gana el de menor k; si hay varios con ese k, gana el de menor SCE de
  prueba.
- La selección se hace sobre el mismo conjunto de prueba, así que la SCE
  del par elegido es optimista: se registra esa aclaración.
- Con el par elegido se corre descenso de gradiente sobre la misma
  partición y se registra su SCE sobre Ω_prueba, para comparar con ALS.
- Elegido el par, se vuelve a entrenar con ALS sobre todo Ω, sin partición.
  Los criterios 3 y 4 de la sección 9 se evalúan sobre ese modelo; el
  criterio 2, sobre la partición.

## 8. Métricas
- SCE sobre Ω_prueba: Σ_{(i,j) ∈ Ω_prueba} (r_ij − r̂_ij)².
- Estimaciones fuera de rango: sobre todos los pares fuera de Ω, porcentaje
  fuera de [ESCALA_MIN − 1, ESCALA_MAX + 1] y máximo de |r̂_ij|.
  ESCALA_MIN = 0,5 y ESCALA_MAX = 5 (escala del dataset, en config.py), así
  que el rango permitido es [−0,5; 6]. "Fuera de Ω" son solo los pares no
  observados (ni en Ω_ent ni en Ω_prueba); los de prueba no cuentan, porque
  ya se miden con la SCE de prueba.
- Concentración: experimentos/concentracion.py con el modelo elegido y
  notas con la distribución real. Se reportan las películas en algún top-10
  y el porcentaje de la más frecuente. El vector del usuario simulado se
  calcula con resolver_factor y el mismo λ.

## 9. Criterios de éxito (fijados antes de correr)
El experimento es exitoso si el par elegido cumple todo:
1. k > 2.
2. Su SCE sobre Ω_prueba es menor que la de (k = 2, λ = 0).
3. A lo sumo 1 % de estimaciones fuera de rango.
4. Al menos 130 películas en algún top-10 (el doble de las 65 de main) y la
   más frecuente en el top-10 de a lo sumo 25 % de los usuarios.
Si falla alguno, el resultado se registra igual y la rama no se integra.

## 10. Criterios de aceptación
CA-R01 Con λ = 0 y sin partición, ALS y descenso de gradiente reproducen
       main: SCE 22.937,3886 en 10 iteraciones y 23.045,8796 en 462.
CA-R02 resolver_factor con λ > 0 resuelve sin SistemaSingularError un
       usuario con menos de k calificaciones.
CA-R03 En una matriz chica y bien condicionada, la salida de
       resolver_factor satisface (Vᵀ V + λ I) u = Vᵀ r según np.allclose
       con rtol = 1e-10.
CA-R04 En una matriz chica, el gradiente de la sección 5 coincide con
       diferencias finitas de la f de la sección 3.
CA-R05 La partición es disjunta, su unión es Ω, es reproducible con la
       semilla, y todo usuario y película de prueba tiene calificaciones en
       entrenamiento.
CA-R06 Los 55 tests de main siguen pasando sin modificaciones y el único
       test salteado sigue siendo el de la V₀ del informe.
CA-R07 Grilla, semilla, fracción de prueba y rango están en config.py o
       como argumentos. Ningún valor hardcodeado.
CA-R08 specs/bitacora.md registra la tabla completa del barrido (k, λ,
       iteraciones, SCE de prueba, fuera de rango), el par elegido, la
       verificación con descenso de gradiente, la concentración y el
       veredicto contra la sección 9.

## 11. Implementación
- Cambios en src/: el parámetro λ, con default 0, en resolver_factor, en f,
  en el gradiente y en entrenar_als y entrenar_gd; y la partición en
  src/datos.py. Nada más.
- El barrido vive en experimentos/regularizacion.py y se corre con
  python -m experimentos.regularizacion.
- Los comentarios citan la sección de esta spec que implementan.