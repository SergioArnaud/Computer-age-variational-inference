# Apéndice 1: La divergencia de Kullback-Leibler

Como vimos en el texto principal, la *divergencia de Kullback Leibler de P desde Q*  juega un rol importante en la inferencia variacional, pues el problema se reduce a resolver (o al menos aproximar)
$$
q^* = \arg\min_\mathcal{Q}\{D_{KL}(q||p(\cdot|\mathbf{X}))\}
$$
En este documento se explora un poco más este importante concepto de manera más intuitiva que formal, pero sobre todo a la luz de diversas disciplinas. 

## Teoría de la información

Desde los años treinta, y sobre todo con el advenimiento de la segunda guerra mundial, hubo mucho interés en estudiar criptografía probabilísticamente. En este contexto, la teoría de la información comenzó en 1948 con la publicación de un paper de [Claude Shannon](https://en.wikipedia.org/wiki/Information_theory) titulado *Una teoría matemática de la comunicación*.$^{[1]}$ 

![Claude Shannon](img/shannon.jpg)



Desde el paper original, Shannon alude a la teoría desarrollada medio siglo antes por Ludwig Boltzmann, de donde toma el término *entropía*. En el paper original, define la entropía $H$ de una variable aleatoria discreta con soporte finito y masa $p$ 
$$
H(X) = -\sum_{x\in\mathcal{X}}p(x_i)\log p(x_i)
$$
Shannon buscaba en $H$ una manera de medir qué tan seguros estamos de los valores de una muestra, conociendo su distribución. La idea es aprovechar la estructura $\sigma$-aditiva del espacio de probabilidad subyacente para dar información antítona a la probabilidad: los eventos más raros deberían dar más información. Puede probarse que la función que resuelve este problema (es antitética a la probabilidad en el sentido de que eventos más raros dan más información) es
$$
I(x) = \log \left( \frac{1}{p(x)}\right)
$$


En este caso, la entropía de Shannon es
$$
H(X) = \mathbb{E}[I(X)]
$$
Algunas observaciones pertinentes:

1. $H$ es no negativa
2. $H$ es cero si y sólo si $X$ toma un valor con probabilidad 1.
3. $H$ es máxima cuando todos los eventos son equiprobables, pues la información inicial es mínima.
4. Si $X_1$ y $X_2$ son muestras independientes, su entropía total es la suma de ambas entropías.

Para entender mejor esta función, y conectarla con la divergencia de Kullback-Leibler, seguimos el camino de [2].

## Teoría de códigos

Sea $\mathcal{C}$ un *corpus* de palabras (una cantidad finita de ellas) que quiere encriptarse con un código binario de manera que no haya ambiguedad en los textos. Es decir, `0` y `01` no pueden ser palabras distintas, porque en `00110`  no se sabría cuál es la primera palabra. Una manera de lograr esto es usar *códigos prefijo*, en los cuales ninguna palabra puede ser prefijo de otra. Sin embargo,  hay un precio a pagar: tener palabras cortas nos obliga a tener palabras largas: si usamos una palabra de longitud $\ell$, no podemos usar $ c(\ell)=\frac{1}{2^\ell}$ palabras de longitud $\ell - 1$ porque causarían ambiguedad [(Kraft-McMillan)](https://en.wikipedia.org/wiki/Kraft–McMillan_inequality). 

Sea $L(x)$ la longitud de la palabra $x \in \mathcal{C}$ en nuestra representación binaria. La longitud de un mensaje $M$ sería $\sum_{x \in M} n_x L(x)$ , donde $n_x$ es el número de veces que $x$ aparece en el mensaje. Si supiéramos que  la palabra $x_i$ ocurre con probabilidad $p_i$ en los textos que nos interesan, podemos asignar $L(x_i)$ inteligentemente para minimizar.

Notemos que podemos invertir el costo, de manera que la longitud de una palabra con un costo dado $r$ es $c^{-1}(r) = \log_2(\frac{1}{r})$.

 

>  **Teorema**
>
> La regla óptima en el sentido de minimización es asignar $L(x_i) = \log_2{\frac{1}{p(x_i)}}= -\log_2p_i$  $_\blacksquare$.  



Partiendo del teorema y un mensaje $M$, si tuviéramos el código óptimo para expresar eventos con masa $\mathbf{p}$, hay un límite a qué tan pequeñas podemos hacer, en promedio, la longitud de $M$: la entropía.
$$
H(\mathbf{p})= -\sum_{i=1}^np_i\log_2p_i
$$
En el mismo contexto, supongamos que recibimos también mensajes de otro evento en el mismo corpus, esta vez con masa $\mathbf{q}$. Si se utilizara el mismo código para enviar mensajes de este evento que del primero, no estaríamos optimizando necesariamente (pues por definición, las las palabras están elegidas para minimizar con respecto a eventos de $\mathbf{p}$. 

Usando la construcción anterior de $H$, es natural extenderla a la *entropía cruzada de q con respecto a p*
$$
H_\mathbf{p}(\mathbf{q})=-\sum_{i=1}^nq_i\log p_i
$$
la longitud promedio de un mensaje que comunica eventos de $\mathbf{q}$ usando un código optimizando para $\mathbf{p}$. 



> **Observación**
>
> $$H_\mathbf{p}(\mathbf{q}) \neq H_\mathbf{q}(\mathbf{p})$$. Un evento común en una y raro en la otra arruina la optimalidad de los códigos elegidos. 



Finalmente, podemos definir la *divergencia de Kullback-Leibler de* $\mathbf{q}$  *a* $\mathbf{p}$ como
$$
D_\textrm{KL}(\mathbf{p}||\mathbf{q}) = H_\mathbf{q}(\mathbf{p})-H(\mathbf{p}) = -\sum_{i=1}^np_i\log_2\left(\frac{q_i}{p_i}\right)
$$
y para interpretarlo, notemos que el término con logaritmo es la diferencia de bits que cada representación usaría para representar a la palabra $x_i$, la longitud extra por usar un código subóptimo



> **Observaciones**
>
> 1. Aunque aquí usamos distribuciones discretas, pueden hacerse los cambios intuitivos para trabajar con variables aleatorias continuas. Formalmente, con variables continuas el término es *entropía diferencial*, y cambia por las tecnicalidades de usar una densidad de probabilidad y no una masa.
> 2. Hay toda una teoría general de divergencias, pero aquí nos limitamos a estudiar la de Kullback-Leibler.

## Máxima verosimilitud – un modelo combinatorio

Sea $X$ una variable nominal con masa de probabilidad $p$. Si obtenemos una muestra de $X$ y hacemos un histograma de sus valores. Si proponemos un modelo $q$ para $p$, podemos hacer una 

## Kullback-Leibler en inferencia bayesiana

Si interpretamos $Q$ como la distribución *a priori* de $\theta$, $D_{KL}(P||Q)$ es la información ganada por usar la posterior $P$ en vez de $Q$. 



> **Observación**
>
> La divergencia de Kullback-Leibler *no* satisface la desigualdad del triangulo. La información que ganamos de $\theta$ dados los valores de $X$ $\{x_1, x_2$,} puede ser mayor, menor o igual que la que ganamos dado $x_1$ solamente. 



## Información de discriminación

![Sollomon Kullback](img/kullback.jpg)

La divergencia de Kullback-Leibler también tiene una interpretación en contrastes de hipótesis, y de hecho es la que Kullback prefería$^{[3]}$: *Información de discriminación*. Supongamos que quieren contrastarse las hipótesis $H_ 1vs. H_2$. Donde $H_i$ es $X\sim f_i$. 

Cuando medimos que $X=x$,  del teorema de Bayes,
$$
\log\frac{f_1(x)}{f_2(x)} = \log \frac{\mathbb{P}\{H_1|X=x\}}{\mathbb{P}\{H_2|X=x\}} - \log \frac{\mathbb{P}\{H_1\}}{\mathbb{P}\{H_2\}}[\lambda]
$$ { }
El lado derecho de la expresión de arriba es una medida de la diferencia en información antes y después de considerar $X=x$, y al izquierdo, logaritmo del cociente de verosimilitudes lo nombramos *la información para discriminar en favor de $H_1$*. 

La información para discriminar en favor de $H_1$ dado que $X=x$ y suponiendo $H_1$ es



 

## Máxima verosimilitud



## Geometría de la información

La interpretación geométrico-diferencial de la esatdística se popularizó con el trabajo de Sun Ichi Amari. Aquí nos limitamos a dar una idea muy superficial, pues de otra manera habría que entrar en detalles no-triviales de geometría y álgebra multilineal:

Una *variedad topológica* es un espacio localmente plano. (Formalmente, es un espacio Hausdorff paracompacto localmente homeomorfo a $\mathbb{R}^m$, y $m$ es su *dimensión*). La noción más inmediata del concepto es el planeta tierra: aunque vivimos en una esfera, localmente parece plano. Sólo cuando vemos al horzionte se nota la curvatura.

Una variedad es *suave* si la transición entre sus mapas es infinitamente diferenciable. Una variedad suave es *Riemanniana* si en el espacio tangente a cada punto (intuitivamente piensen en la derivada, es algo plano que aproxima localmente) hay un producto interno definido y la transición entre ellos (de punto a punto) es diferenciable. Esta estructura permite definir una métrica en las variedades riemannianas: se mide el largo de una curva suave que conecta dos puntos.

La geometría de la información considera variedades riemannianas donde cada punto es una medida de probabilidad. Su métrica riemanniana correspondiente es la matriz de información de Fischer, que bajo algunas condiciones de regularidad tiene entradas
$$
\mathcal{I}(\theta)_{ij} = - \mathbb{E}\left[\frac{\partial^2}{\partial\theta_i\partial\theta_j}f(x;\theta) | \theta\right]
$$


## 

## Referencias

[1] Shannon, C. E. (1948). *A Mathematical Theory of Communication*. Bell System Technical Journal, 27(3), 379–423. doi:10.1002/j.1538-7305.1948.tb01338.x

[2] http://colah.github.io/posts/2015-09-Visual-Information/

[3] (1987) Letters to the Editor, The American Statistician, 41:4, 338-341, DOI: 10.1080/00031305.1987.10475510