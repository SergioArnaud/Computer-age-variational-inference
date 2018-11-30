## Motivación

Consideremos el problema típico de inferencia bayesiana:
$$
p(\theta  |  \mathbf{x}) \propto p(\theta |  \alpha) p(\mathbf{x}  | \theta)
$$
En muchos modelos bayesianos, la dificultad o imposibilidad para calcular $p(\mathbf{x})$ (que es una integral en dimensiones potencialmente altas) vuelve problemlático obtener una forma analítica de la distribución posterior. La técnica más común para aproximarla son los algoritmos de MCMC, que construyen una cadena de Markov sobre $\theta$  con distribución estacionaria $p(\theta  | \mathbf{x})$ para obtener, simulando, una muestra.

Pese a que los métodos de MCMC son una herramienta poderosa en el cómputo estadístico, hay ciertos problemas y contextos en los que fallan. En particular, es difícil determinar la convergencia, y en problemas muy complicados o de escala masiva pueden ser demasiado lentos para ser de utilidad práctica. Una alternativa más eficiente es la *inferencia variacional*, que replantea el problema como uno de optimización determinista al proponer una familia de densidades $\mathscr{Q}$ sobre $\theta$ y aproximar la distribución posterior $p$
$$
q^* = \underset{q \in \mathscr{Q}}{\arg\min} \{ D_{KL}(q  || p)\}
$$
Pese a la eficiencia en términos computacionales de la inferencia variacional, su uso no se ha extendido dentro de la comunidad estadística. Esto se debe en mayor medida a que utilizar inferencia variacional requiere de un diseño cuidadoso de la rutina de optimización: encontrar una familia variacional adecuada al modelo, obtener explícitamente la función objetivo, su gradiente y realizar un procedimiento de optimización. 

En este trabajo presentamos dos algoritmos que automatizan el proceso de optimización y una prueba cuantitativa de la convergencia del algoritmo. 

