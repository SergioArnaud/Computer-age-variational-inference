## Inferencia variacional

La inferencia variacional transforma el problema de muestreo a un problema de optimización determinista. En particular, formula el prolema
$$
q^* = \underset{q \in \mathscr{Q}}{\arg\min} \{ D_{KL}(q \ || \ \bar{p})  \}
$$
donde $D_{KL}$ es la *divergencia de Kullback-Leibler*
$$
D_{KL}(q\ ||\ \bar{p}) = \mathbb{E}_{\theta \sim q}\left[\log\frac{q(\theta)}{\bar{p}(\theta)}\right]
$$
 y  $\mathscr{Q}$ una familia de funciones de probabilidad definida de antemano. En el apéndice A se explica con detalle la construcción y significado de la divergencia de Kullback-Leibler, pero por ahora basta saber que mide la cantidad de información perdida por utilizar $q$ en vez de $\bar{p}$. 

Lamentablemente, es imposible optimizar Kullback-Leibler de manera directa, pues
$$
\begin{align}
D_{KL}(q \ || \ \bar{p}) &= \mathbb{E}_{\theta\sim q}[\log q(\theta)] - 
	\mathbb{E}_{\theta\sim q}[\log \bar{p}(\theta)]\\ 
&= \mathbb{E}_{\theta\sim q}[\log q(\theta)] - 
	\mathbb{E}_{\theta\sim q}[\log p(\theta , x)] + \log p(x) \\
\end{align}
$$
Como la divergencia depende de la marginal de $x$, cuya dificultad de cálculo es la razón por la cuál usamos métodos aproximados de inicio, optimizamos en su lugar la siguiente función.

###### Definición.

La *cota inferior a la evidencia* es la función 
$$
\mathrm{ELBO}(q) = \mathbb{E}_{\theta\sim q}\left[\log \frac{p(\theta , x)}{q(\theta)}\right]
$$
###### Observación.

$$
\begin{align}
\mathrm{ELBO}(q) &= \mathbb{E}_{\theta\sim q}[\log p(\theta , x)] - 
	\mathbb{E}_{\theta\sim q}[\log q(\theta)] \\
&= \log p(x) - D_{KL}(q \ || \ \bar{p})
\end{align}
$$
Y como $\log p(x)$ es un valor fijo, maximizar $\mathrm{ELBO}(q)$  es equivalente a minimizar la divergencia de Kullback-Leibler. Recordando que evidencia es el término usual para la marginal de $x$ en el modelo, el nombre de la función $\mathrm{ELBO}$ se justifica con lo siguiente.

###### Proposición.

$$
ELBO(q) \leq \log p(x)
$$
> *Demostración*.
>
> Basta recordar que por ser una divergencia, $D_{KL}(q\ ||\ \bar{p}) \geq 0. \ \ _\square$ 



