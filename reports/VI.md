## Inferencia variacional

Como mencionamos previamente, la inferencia variacional consiste en minimizar:
$$
q^*(\theta) = \underset{q(\theta) \in \mathscr{F}}{argmin} \{ KL(q(\theta) \ || \ p(\theta \ | \ \mathbf{x}) )  \}
$$
Notemos que:
$$
\begin{eqnarray}
KL(q(\theta) \ || \ p(\theta \ | \ \mathbf{x})) &=& \mathbb{E}[\log q(\theta)]_{q(\theta)} - E[\log p(\theta \ | x)]_{q(\theta)} \\ 
&=& \mathbb{E}[\log q(\theta)]_{q(\theta)} - E[\log p(\theta , x)]_{q(\theta)} + \log p(x) \\
\end{eqnarray}
$$
Dada dicha formulación, optimizar $ KL(q(\theta) \ || \ p(\theta \ | \ \mathbf{x})) $ directamente no es posible, puesto que depende explícitamente de la evidencia ($p(x)$) y la dificultad para obtenerla es, precisamente, lo que nos llevó a buscar una aproximación de la posterior.

En vez de minimizar la divergencia de Kullback-Leibler, trabajaremos con la siguiente función objetivo que tiene una forma similar pero no involucra a la evidencia.
$$
\begin{eqnarray}
ELBO(q) &=& E\left[\log \frac{p(\theta , x)}{q(\theta)}\right]_{q(\theta)} \\
&=& E[\log p(\theta , x)]_{q(\theta)} - \mathbb{E}[\log q(\theta)]_{q(\theta)}
\end{eqnarray}
$$
Dicha expresión no contiene términos desconocidos por lo que es posible optimizarla. Asimismo, presenta la siguiente propiedad:
$$
\begin{eqnarray}
ELBO(q) &=& E[\log p(\theta , x)]_{q(\theta)} - \mathbb{E}[\log q(\theta)]_{q(\theta)} \\
&=& - KL(q(\theta) \ || \ p(\theta \ | \ \mathbf{x}) + \log p(x) 
\end{eqnarray}
$$
Puesto que $\log p(x)$ es un valor fijo, maximizar ELBO(q) es equivalente a minimizar la divergencia de Kullback-Leibler. Otra propiedad de ésta función se obtiene de la anterior expresión al recordar que $KL(q(\theta) \ || \ p(\theta \ | \ \mathbf{x}) \geq 0 $ :
$$
ELBO(q) \leq \log p(x)
$$
Esta última desigualdad otorga sentido al nombre de la función, *evidence lower bound* (ELBO). Finalmente observemos que:
$$
\begin{eqnarray}
ELBO(q) &=& - KL(q(\theta) \ || \ p(\theta \ | \ \mathbf{x}) + \log p(x) \\
&=& E[\log p(\theta , x)]_{q(\theta)} + \mathbb{E}\left[\log \frac{1}{q(\theta)}\right]_{q(\theta)} \\
&=& E[\log p(\theta , x)]_{q(\theta)} + H(q) 
\end{eqnarray}
$$
El primer término es la esperanza de la densidad conjunta bajo la aproximación y el segundo es la entropía de Shanon de q, la densidad variacional. 

 









