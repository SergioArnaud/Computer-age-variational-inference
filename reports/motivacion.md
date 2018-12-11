## Motivación

Consideremos el problema típico de inferencia bayesiana:
$$
p(\theta | \textbf{x}) = \frac{p(\theta) \ p(\textbf{x} | \theta)}{p(\textbf{x})} \propto p(\theta) \ p(\textbf{x} | \theta)
$$
En muchos modelos bayesianos, la dificultad o imposibilidad para calcular $p(\mathbf{x})$ (que es una integral en dimensiones potencialmente altas) vuelve problemático obtener una forma analítica de la distribución posterior, pese a que la técnica más común para aproximarla son los métodos de MCMC, hay ciertos contextos en los que fallan. En particular, es difícil determinar la convergencia, y en problemas muy complicados o de escala masiva pueden ser demasiado lentos para ser de utilidad práctica. 

A continuación mostramos tres de modelos que, dada su complejidad intrínseca, presentan problemas de escalabilidad al utilizar métodos de MCMC.

###### Latent Dirichlet allocation (LDA).

LDA es un modelo probabilístico generativo que forma parte de los modelos de mezclas Gaussianas, es ampliamente utilizado en procesamiento de lenguaje natural para resolver problemas de clasificación de texto y modelado de temas, la idea básica del procedimiento para generar un documento es la siguiente: 

Supongamos que se tiene un vocabulario de palabras y un conjunto de temas, entonces, para cada tema podemos obtener una distribución de las palabras que aparecen en él. De esta forma, para generar un documento, se elige una distribución de los temas que se van a tratar en el documento y con base en ella se obtiene el conjunto de palabras correspondiente. El modelo probabilístico es el siguiente:

1. Elige el número de palabras de el documento, $N \sim \mathrm{Poisson}(\xi)$.

2. Elige un parámetro para la distrubución de los temas del documento,  $\theta \sim \mathrm{Dir}(\alpha)$.

3. Para cada una de las N palabras $w_n$:

   a. Elige un tema $z_n \sim \mathrm{Multinomial}(\theta)$.

   b. Elige una palabra $w_n$ de $p(w_n|z_n,\beta)$, una distribución multinomial condicionada en el tema $z_n$.

Una vez especificado el proceso generativo, supongamos que se tiene un conjunto de documentos y se ha fijado $K$, el número de temas que se quieren obtener en los documentos. Entonces el problema de inferencia consiste en intentar *trabajar hacia atras* el proceso generativo, es decir, se requiere que el modelo aprenda la representación de los K temas en cada documento y la distribución de palabras en los tema de forma que se pueda identificar 

###### Componenetes principales probabilísticos con detección automática de relevancia.

Inicialmente, se considera el modelo de componenetes principales probabilísticos (PPCA), supongamos que se tiene un conjunto de datos $\bold{x} = x_{1:n}$ donde cada $x_i \in \mathbb{R}^n$ . Sea $M<D$ la dimensión del subespacio buscado, definimos:
$$
\begin{align*}
p(\bold{x} | \bold{w},\bold{z},\sigma) &= \prod_{n=1}^N \mathscr{N}(x_n; \bold{w}z_n,\sigma\mathbb{I}) \\
p(\bold{z}) &= \prod_{n=1}^N \mathscr{N}(z_n;0,\mathbb{I}) \\
p(\bold{w}) &= \prod_{n=1}^D \mathscr{N}(w_n;0,\mathbb{I}) \\
p(\sigma) &= \mathrm{Lognormal}(\sigma;0,1) \\
\end{align*}
$$
Donde $\bold{w} = w_{1:D}$ es  un conjunto de componentes principales

La detección automática de relevancia consiste en extender PPCA al añadir un vector M-dimensional $\bold{\alpha}$ que elige las componentes principales que serán retenidas al añadir:
$$
\begin{align*}
p(\bold{w} | \bold{\alpha}) &= \prod_{n=1}^D \mathscr{N}(w_d;0,\sigma \mathrm{diag}(\alpha)) \\
p(\bold{\alpha}) &= \prod_{n=1}^M \mathrm{Invgamma}(\alpha_m;1,1)
\end{align*}
$$

#### Inferencia variacional

Una alternativa más eficiente es la *inferencia variacional*, que replantea el problema como uno de optimización determinista al proponer una familia de densidades $\mathscr{Q}$ sobre $\theta$ y aproximar la distribución posterior $\bar{p}$
$$
q^* = \underset{q \in \mathscr{Q}}{\arg\min} \{ D_{KL}(q  || p)\}
$$
Pese a la eficiencia en términos computacionales de la inferencia variacional, su uso como método general presenta ciertas complicaciones puesto que requiere de un diseño cuidadoso de la rutina de optimización: encontrar una familia variacional adecuada al modelo, obtener explícitamente la función objetivo y su gradiente (esto puede llegar a ser sumamente complicado) y realizar un procedimiento de optimización apropiado. 

Han surgido múltiples propuestas que pretenden automatizar computacionalmente el proceso de optimización, en este proyecto se discutirán dos algoritmos publicados en el 2016 que resuelven el problema por optimización estocástica ADVI y SVGD. El primero, reparametriza de manera automática para optimizar de una familia $\mathcal{Q}$ fija de antemano. El segundo, construye una especie de descenso en gradiente en un reproducing kernel hilbert space (RKHS) apropiado de manera que se minimice la divergencia de Kullback-Leibler

