## Evaluación

La inferencia variacional reduce el tiempo de cómputo para muestrear posteriores complicadas, pero da sólo un aproximado. ¿Cómo podemos verificar qué tan bueno es el aproximado? Primero, es fácil diagnosticar la convergencia del algoritmo de optimización monitoreando el cambio en $\mathrm{ELBO}$, y esto siempre debería hacerse. Sin embargo, puede ser que incluso habiendo convergido el algoritmo a un máximo global de $\mathrm{ELBO}$, la familia  $\mathscr{Q}$ haya sido elegido tan desafortunadamente que aún el óptimo es malo o que por la poca penalización que $D_{KL}(q\ ||\ p)$ le pone a las colas ligeras tenagmos una densidad muy distinta a la verdadera.

Yao *et al.* (2018) proponen dos diagnósticos cuantitativos, uno para la calidad de la posterior variacional y otro para evaluar la calidad de los estimadores puntuales que podemos derivar de ella. Referimos al lector al paper original para el segundo y trataremos aquí sólo el primero.

Como vimos durante el curso, puede estimarse para cualquier función integrable $h$, $\phi = \mathbb{E}_{\theta\sim \bar{p}}[h(\theta)]$ con métodos de Monte Carlo, y la ventaja del *importance sampling* en situaciones como esta, cuando es difícil muestrear de $\bar{p}$.

Consideremos el estimador, para una muestra $\{\theta_i:i=1,\cdots,n\}$ de la posterior variacional $q$,
$$
\mathbb{E}_{\theta\sim\bar{p}}[h(\theta)]\approx\frac{\sum_{i=1}^nh(\theta_i)w_i}{\sum_{i=1}^nw_i}
$$
Cuando $w_i\equiv1$, estamos confiando completamente en la aproximación, pues la expresión se simplifica al estimador de monte carlo crudo para $\mathbb{E}_{\theta\sim q}[h(\theta)]$.

###### Proposición.

Si $w_i = p(\theta_i, x)/q(\theta_i)=(\bar{p}(\theta_i)p(x)/q(\theta_i))$, el estimador de (1) es importance sampling usando $q$.

> *Demostración*.
> $$
> \begin{align}
> \frac{\sum_{i=1}^n\frac{h(\theta_i)p(\theta_i,x)}{q(\theta_i)}}
> 	{\sum_{i=1}^n\frac{p(\theta_i,x)}{q(\theta_i)}}&\approx
> 	\frac{n\mathbb{E}_{\theta\sim q}\left[\frac{h(\theta)p(\theta,x)}{q(\theta)}\right]}{n\mathbb{E}_{\theta\sim q}\left[\frac{p(\theta,x)}{q(\theta)}\right]} \\
>
> &=\frac{\mathbb{E}_{\theta\sim q}\left[\frac{h(\theta)\bar{p}(\theta)p(x)}{q(\theta)}\right]}{\int_\Theta p(x,\theta)d\theta} \\
>
> &=\frac{p(x)\mathbb{E}_{\theta\sim\bar{p}}[h(\theta)]}{p(x)} \\
>
> &= \mathbb{E}_{\theta\sim\bar{p}}\left[h(\theta)\right]
>
> \end{align}
> $$
>

¿Existirá una manera intermedia, un mejor *bias-variance tradeoff* que las dos opciones anteriores?

#### Pareto smoothed importance sampling

Notemos que en $(2)$, como $q$ fue obtenida por inferencia variacional y puede tener colas mucho más ligeras que la posterior $\bar{p}$, los sumandos pueden hacerse grandes. Pareto smoothed importance sampling (PSIS) es una técnica de reducción de varianza introducida por Vehtari *et al.* (2017), que ajusta una distribución pareto generalizada a los $M = \min\{n/5, 3\sqrt n\}$ valores más grandes de $w_i$ y luego los reemplaza por su valor esperado bajo esta distribución.

###### Definición.

Una variable aleatoria $X$ tiene distribución *pareto generalizada* con parámetro $\theta = (k, \mu, \sigma)^\top$ si su densidad es
$$
p(x|\theta)=
\begin{cases}
\frac{1}{\sigma}\left(1+k\left(\frac{x-\mu}{\sigma}\right)\right)^{-1/k-1},\  k\ne0 \\
\frac{1}{\sigma}\exp\left(\frac{y-\mu}{\sigma}\right),\ \ \ \  k=0
\end{cases}
$$
La siguiente propiedad del parámetro $k$, que enunciamos sin demostración, nos permite usarlo como medida de discrepancia entre la verdadera posterior y la posterior variacional.

###### Teorema.

Si $X$ es una pareto generalizada con parámetro $\theta$,
$$
k = \underset{k'>0}{\inf} \left\{ \mathbb{E}_{\theta\sim q}\left[\frac{\bar{p}(\theta)}{q(\theta)}\right]^\frac{1}{k'} < \infty \right\} =
\underset{k'>0}{\inf}\left\{\mathbb{E}_{\theta\sim q}\left[\frac{p(x,\theta)}{q(\theta)}\right]^\frac{1}{k'}\right\}
= \underset{k'>0}{\inf}\left\{D_\frac{1}{k'}(\bar{p}\ ||\ q)<\infty\right\}
$$

######

Donde $D_\alpha$ es la *divergencia de Rényi de orden* $\alpha$
$$
D_\alpha(\bar{p}\ ||\ q)=\frac{1}{\alpha-1}\int_\Theta p(\theta)^\alpha
q(\theta)^{1-\alpha}d\theta
$$

###### Corolario.

Si $k>1/2$, la divergencia $\chi^2$ es infinita, y si $k >1$, $D_{KL}(\bar{p} || q) = \infty$.

¿Por qué aproximar $k$ es una buena prueba? Aunque sabemos que estamos en un mínimo (por lo menos local) de $D_{KL}(q\ ||\ \bar{p})$ para $q \in \mathscr{Q}$ , es preocupante que la divergencia en el sentido opuesto sea infinita. En la práctica, los autores sugieren descartar la inferencia desde $\hat{k} \geq 0.7$.

#### Algoritmo

```pseudocode
diagnostico_psis(conjunta=p(x, theta), x=datos, n, tol):
"""Diagnóstico de inferencia variacional.

	Parámetros:
	-----------
	conjunta (modelo de probabilidad):
		La distribución conjunta.
	n (entero):
		El número de muestras de la posterior variacional.

	Regresa:
	--------
	bandera (entero):
		Indicador que vale 0 si la aproximación es muy buena, 1 si es
        buena y -1 si es mala, según el criterio de arriba.
"""

q <- obtener por inferencia variacional
theta <- muestrear n puntos de q
w <- p(theta, x)/q(theta)
k_gorro <- ajustar una pareto generalizada a los M mayores valores de theta y
			reportar el valor del parámetro de forma
if k < 0.5
	return 0
else if k < 0.7
	return 1
else
	return -1
```

Cuando $\Theta$ tiene dimensión alta, los métodos variacionales pierden poder predictivo. En este caso, los autores recomiendan limitarse a evaluar los estimadores puntuales con su segundo método.

#### Otra forma de mejorar el ajuste

Recientemente se ha intentado corregir el problema de la subestimación de colas en la inferencia variacional cambiando la función objetivo de $D_{KL}$ a una clase más general.

###### Definición.

Las *divergencias f* son la familia de funciones
$$
D_f(p\ ||\ q)=\mathbb{E}_{x\sim q}\left[f\left(\frac{p(x)}{q(x)}\right)-f(1)\right]
$$
donde $f: \mathbb{R}^+ \to \mathbb{R}​$ es convexa.  Una clase particular es la de *divergencias* $\alpha​$, que usan $f(t) = t^\alpha/(\alpha(\alpha-1))​$ y tienen como caso particular a ambos casos de Kullback-Leibler, tomando $\alpha \rightarrow 0​$ y $\alpha\rightarrow1​$ respectivamente. Mientras más grande es $\alpha​$ (aunque esté acotado por 1), se le da más importancia a cubrir las partes de $p​$ con probabilidad positiva. Es el parámetro en el *bias-variance tradeoff* entre usar Kullback-Leibler hacia adelante y hacia atrás. Ajustando $\alpha​$ pueden mejorarse las estimaciones de modelos multimodales, y los métodos más recientes (Wang, Liu y Liu; 2018) usan divergencias $f​$ con una sucesión de funciones $f​$ que se ajustan para garantizar momentos finitos y evitar ceros.
