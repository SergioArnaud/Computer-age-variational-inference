## Motivación

Consideremos el modelo de inferencia bayesiana:
$$
p(\theta \ | \ \ \textbf{x},\alpha) = \frac{p(\theta \ | \  \alpha) \ p(\textbf{x} \ | \ \ \theta, \alpha)}{p(\textbf{x} \ | \ \alpha)} \propto p(\theta \ | \  \alpha) \ p(\textbf{x} \ | \ \ \theta, \alpha)
$$
Donde:

- **x** es un vector de observaciones.
- $\theta$ es el vector de parámetros del modelo.
- $\alpha $ es el vector de hiperparámetros del modelo.

A lo largo del trabajo se omitiran notacionalmente los hiperparámetros de forma que obtenemos el modelo:
$$
p(\theta \ | \ \textbf{x}) = \frac{p(\theta) \ p(\textbf{x} \ | \ \theta)}{p(\textbf{x})} \propto p(\theta) \ p(\textbf{x} \ | \ \theta)
$$
Donde:

- $p(\theta) $ es la _distribución a priori_ del parámetro.
- $p(\mathbf{x}  \ | \ \theta)$ es la _distribución muestral_ de las observaciones condicionadas a los parámetros.
- $p(\mathbf{x})  = \int p(\mathbf{x} \ | \ \theta)p(\theta) $ Es la _distribución marginal_ de las observaciones (también llamada _evidencia_).
- $p(\theta \ | \ x)$ la _distribución posterior_ del parámetro.

Durante el proceso de inferencia bayesiana es de interés, dado un conjunto de observaciones, calcular la distribución posterior del parámetro. Sin embargo, en muchos modelos bayesianos la dificultar para calcular $p(x)$ imposibilita la obtención explícita  de la distribución posterior por lo que surge la idea de obtener una aproximación a dicha distribución.

En éste contexto, la técnica más común está basada en algoritmos de MCMC que consisten en construir una cadena de Markov sobre $\theta$  con distribución estacionaria $p(\theta \ | \ x)$. Al simular dicha cadena, se obtiene una muestra de $p(\theta \ | \ x)$ tomando un subconjunto de los valores tomados por la cadena.

Pese a que los métodos de MCMC son una de las herramientas más poderosa del cómputo estadístico, hay ciertos problemas y contextos en los que fallan. En general, esto ocurre cuando las simulaciones son computacionalmente intensivas, ya sea porque la cantidad de datos es masiva o porque los modelos son sumamente complejos, la _Inferencia variacional_ surge como una alternativa en dichos casos.  

La inferencia variacional replantea el problema como uno de optimización al proponer una familia de densidades $\mathscr{F}$ sobre $\theta$ y aproximar la distribución posterior de $\theta$ con $q^*(\theta)$, donde $q^*(\theta)$ es la distribución en $\mathscr{F}$ más _cercana_ a $p(\theta \ | \ x)$ en el sentido de la divergencia de Kullback-Leibler, es decir:
$$
q^*(\theta) = \underset{q(\theta) \in \mathscr{F}}{argmin} \{ KL(q(\theta) \ || \ p(\theta \ | \ \mathbf{x}) )  \}
$$
Pese a la eficiencia en términos computacionales de la inferencia variacional, su uso no se ha extendido dentro de la comunidad estadística, esto se debe en mayor medida a que utilizar inferencia variacional requiere de un diseño cuidadoso de la rutina de optimización: encontrar una familia variacional adecuada al modelo, obtener explícitamente la función objetivo, su gradiente y realizar un procedimiento de optimización *ad hoc* al problema.

La Inferencia variacional por diferenciación automática resuelve este problema automáticamente 