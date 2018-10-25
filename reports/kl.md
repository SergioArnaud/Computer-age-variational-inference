# Apéndice 1: La divergencia de Kullback-Leibler

Como vimos en el texto principal, la *divergencia de Kullback Leibler de P desde Q*  juega un rol importante en la inferencia variacional, pues el problema se reduce a resolver (o al menos aproximar)
$$
q^* = \arg\min_\mathcal{Q}\{D_{KL}(q||p(\cdot|\mathbf{X}))\}
$$
En este documento se explora un poco más este importante concepto de manera más intuitiva que formal, pero sobre todo a la luz de diversas disciplinas. 

## Teoría de la información

Desde los años treinta, y sobre todo con el advenimiento de la segunda guerra mundial, hubo mucho interés en estudiar criptografía probabilísticamente. En este contexto, la teoría de la información comenzó en 1948 con la publicación de un paper de [Claude Shannon](https://en.wikipedia.org/wiki/Information_theory) titulado *Una teoría matemática de la comunicación*.$^{[1]}$ 

![Claude Shannon](img/shannon.jpg)



Desde el paper original, Shannon alude a la teoría desarrollada medio siglo antes por Ludwig Boltzmann, de donde toma el término *entropía*. En el paper original, define la entropía $H$ de una variable aleatoria discreta con soporte finito y masa $$\mathbf{p}$$ 
$$
H = -K\sum_{i=1}^np_i\log_b p_i
$$
Shannon buscaba en $H$ una manera de medir qué tan seguros estamos de los valores de una muestra, conociendo su distribución. En particular, es la función tal que

1. Es no negativa

2. Es cero si y sólo si $X$ toma un valor con probabilidad 1.

3. Es máxima cuando todos los eventos son equiprobables, pues la información inicial es mínima.

4. Si $X_1$ y $X_2$ son muestras independientes, su entropía total es la suma de ambas.

5. Le da más importancia a los eventos de probabilidad pequeña. 



   Para entender mejor la necesidad de esta medición, seguimos un ejemplo de [2] que conecta la entropía de Shannon con la divergencia de Kullback-Leibler.



## Teoría de códigos

Sea $\mathcal{C}$ un *corpus* de palabras (una cantidad finita de ellas) que quiere encriptarse con un código binario de manera que no haya ambiguedad en los textos. Es decir, `0` y `01` no pueden ser palabras distintas, porque en `00110`  no se sabría cuál es la primera palabra. Una manera de lograr esto es usar *códigos prefijo*, en los cuales ninguna palabra puede ser prefijo de otra. Sin embargo,  hay un precio a pagar: tener palabras cortas nos obliga a tener palabras largas: si usamos una palabra de longitud $\ell$, no podemos usar $ c(\ell)=\frac{1}{2^\ell}$ palabras de longitud $\ell - 1$ porque causarían ambiguedad [(Kraft-McMillan)](https://en.wikipedia.org/wiki/Kraft–McMillan_inequality). 

Sea $L(x)$ la longitud de la palabra $x \in \mathcal{C}$ en nuestra representación binaria. La longitud de un mensaje $M$ sería $\sum_{x \in M} n_x L(x)$ , donde $n_x$ es el número de veces que $x$ aparece en el mensaje. Si supiéramos que  la palabra $x_i$ ocurre con probabilidad $p_i$ en los textos que nos interesan, podemos asignar $L(x_i)$ inteligentemente para minimizar.

Notemos que podemos invertir el costo, de manera que la longitud de una palabra con un costo dado $r$ es $c^{-1}(r) = \log_2(\frac{1}{r})$.

 

>  **Teorema**
>
> La regla óptima en el sentido de minimización es asignar $L(x_i) = \log_2{\frac{1}{p(x_i)}}= -\log_2p_i$  $_\blacksquare$.  



Partiendo del teorema y un mensaje $M$, si tuviéramos el código óptimo para expresar eventos con masa $\mathbf{p}$, hay un límite a qué tan pequeñas podemos hacer, en promedio, la longitud de $M$: la entropía.
$$
H(\mathbf{p})= -\sum_{i=1}^np_i\log_2p_i = \mathbb{E}[\textrm{Longitud de $M$}]
$$
En el mismo contexto, supongamos que recibimos también mensajes de otro evento en el mismo corpus, esta vez con masa $\mathbf{q}$. Si se utilizara el mismo código para enviar mensajes de este evento que del primero, no estaríamos optimizando necesariamente (pues por definición, las las palabras están elegidas para minimizar con respecto a eventos de $\mathbf{p}$. 

Usando la construcción anterior de $H$, es natural extenderla a la *entropía cruzada de q con respecto a p*
$$
H_\mathbf{p}(\mathbf{q})=-\sum_{i=1}^nq_i\log p_i
$$
la longitud promedio de un mensaje que comunica eventos de $\mathbf{q}$ usando un código optimizando para $\mathbf{p}$. 



> **Observación**
>
> $$H_\mathbf{p}(\mathbf{q}) \neq H_\mathbf{q}(\mathbf{p})$$. Un evento común en una y raro en la otra arruina la optimali de los códigos elegidos. 



Finalmente, podemos definir la *divergencia de Kullback-Leibler de* $\mathbf{q}$  *a* $\mathbf{p}$ como
$$
D_\textrm{KL}(\mathbf{p}||\mathbf{q}) = H_\mathbf{q}(\mathbf{p})-H(\mathbf{p}) = -\sum_{i=1}^np_i\log_2\left(\frac{q_i}{p_i}\right)
$$
y para interpretarlo, notemos que el término con logaritmo es la diferencia de bits que cada representación usaría para representar a la palabra $x_i$, la longitud extra por usar un código subóptimo



> **Observación**
>
> Aunque aquí usamos distribuciones discretas, pueden hacerse los cambios usuales para trabajar con variables aleatorias continuas.



## Inferencia bayesiana

Si interpretamos $Q$ como la distribución *a priori* de $\theta$, $D_{KL}(P||Q)$ es la información ganada por usar la posterior $P$ en vez de $Q$. 



## Información de discriminación

La divergencia de Kullback-Leibler también tiene una interpretación frecuentista, y de hecho es la que Kullback prefería$^{[3]}$: *Información de discriminación*. 



## Máxima verosimilitud



## Geometría de la información



## 

## Referencias

[1] Shannon, C. E. (1948). *A Mathematical Theory of Communication*. Bell System Technical Journal, 27(3), 379–423. doi:10.1002/j.1538-7305.1948.tb01338.x

[2] http://colah.github.io/posts/2015-09-Visual-Information/

[3] (1987) Letters to the Editor, The American Statistician, 41:4, 338-341, DOI: 10.1080/00031305.1987.10475510