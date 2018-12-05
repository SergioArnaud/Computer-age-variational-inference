## ADVI: inferencia variacional con diferenciación automática



ADVI es un algoritmo introducido, entre otros, por Andrew Gelman y David Blei en 2016. En resumen, dada la especificación de un modelo p(x, \theta), ADVI lo transforma a p(x, \zeta) (ajustando por el mapa \theta \mapsto \zeta en la nueva conjunta) donde \zeta \in \mathbb{R}^p. Así elimina las restricciones sobre el parámetro \theta, y es posible fijar \mathscr{Q} de antemano y utilizarla para cualquier modelo. Después, ADVI transforma la función objetivo a un valor esperado sobre la función propuesta, de manera que pueda aproximarse con métodos de Monte Carlo, y resuelve la optimización con una variante de gradient descent.

Comencemos por definir los modelos que ADVI puede resolver: los modelos diferenciables.

###### Definición.

Un modelo de probabilidad $p(x, \theta)$ es *diferenciable* si $\theta$ es continua y existe $\nabla_\theta\log p(x, \theta)$ en el soporte   $\Theta \subseteq \mathbb{R}^p$  de $\theta$.

Esta clase es más extensa de lo que puede parecer a primera vista, pues considerando que se puede marginalizarse sobre las variables discretas incluye no sólo a los modelos lineales generalizados y procesos gaussianos, sino también de modelos de mezclas, latent Dirichlet allocation y hidden Markov models.

Estudiemos con más detalle el funcionamiento de ADVI.

#### Transformación automática de variables

El primer paso del algoritmo es transformar los parámetros del modelo, de $\theta \in \Theta \subseteq \mathbb{R}^p$ a $T(\theta) = \zeta \in \mathbb{R}^p$. Habiendo elegido $T$, la nueva conjunta es
$$
p(x,\zeta) = p(x, T^{-1}(\zeta))|\det \mathbb{J}_{T^{-1}}(\zeta)|
$$
donde $\mathbb{J}_f$ es la derivada de $f$. Los resultados no son invariantes ante la elección de $T$, pero calcular la transformación óptima es un problema equivalentemente difícil, pues depende de la distribución de $\theta|X$. Hay trabajo en curso para mejorar la elección, pero por ahora basta saber que las implementaciones computacionales (Stan y PyMC3) se encargan de esto.

#### El problema variacional en $\mathbb{R}^p$

ADVI implementa dos de las familias $\mathscr{Q}$ más prominentes en la literatura para $\Theta=\mathbb{R}^p$:

###### Definición.

La familia *mean-field gaussian* es el conjunto de distribuciones de la forma
$$
q(\zeta;\phi) = \mathscr{N}(\zeta;\mathbf{\mu},\mathrm{diag}(\exp(\omega)^2))
= \prod_{i=1}^p\mathscr{N}(\zeta_i;\mu_i,(e^{\omega_i})^2)
$$
donde $\omega = \log(\sigma)$ es una reparametrización que permite tener un espacio parametral libre, pues $\phi = (\mu_1, \cdots, \mu_p, \omega_1, \cdots, \omega_p)^\top \in \mathbb{R}^{2p}$ .

###### Definción.

La familia *full rank gaussian* es el conjunto de distribuciones de la forma
$$
q(\zeta;\phi) = \mathscr{N}(\zeta;\mu, LL^\top)
$$
donde $LL^\top$ es la factorización de Cholesky de $\Sigma = \mathrm{Cov}(\zeta)$. En este caso, la matriz tiene $p(p+1)/2$ parámetros reales y $\phi \in \mathbb{R}^{p+p(p+1)/2}$.

Analizando de manera rápida el *bias-variance tradeoff* de la elección de modelo, usando full rank el espacio de parámetros sube de $\mathcal{O}(p)$ a $\mathcal{O}(p^2)$, por lo que la estimación se complica, pero sí nos permite expresar correlación entre las entradas de $\zeta$, una estructura mucho más rica.

Notemos también que aunque $q$ es normal en el espacio de coordenadas de $\zeta$,  no lo es en el espacio de $\theta$; pues implícitamente definimos la variacional en el espacio original al definir $T$.
$$
q^*(\theta)= q(T(\theta);\phi)|\det\mathbb{J}_T(\theta)|
$$
que no necesariamente es normal.

Aquí la familia $\mathscr{Q}$ ya está parametrizada sobre $\phi$, por lo que denotaremos $q_\phi(\zeta)=q(T(\theta);\phi)$.

Una vez transformado el parámetro, el objetivo es minimizar
$$
\mathrm{ELBO}(q_\phi) =
\mathbb{E}_{\zeta\sim q_\phi}[\log p(x,T^{-1}(\zeta))
	+\log|\det\mathbb{J}_{T^{-1}}(\zeta)|]
	+\mathbb{H}(q_\phi)
$$
donde $\mathbb{H}$ es la entropía de Shannon (más en el apéndice A).  Como el parámetro transformado toma valores reales, el problema de inferencia variacional es
$$
\phi^*=\underset\phi{\arg\max}\ \mathrm{ELBO}(\phi)\
$$
Como no hay restricciones, podemos resolverlo por ascenso en gradiente. Sin embargo, no podemos usar diferenciación automática directamente en $\mathrm{ELBO}$. ADVI implementa una transformación más para poder "meter" el gradiente a la esperanza, para después utilizar integración de Monte Carlo. Específicamente, buscamos una transformación $S_\phi$ que absorba los parámetros variacionales para convertir la aproximación variacional a una normal estándar.

###### Proposición.

En el caso *mean-field*, la adecuada es $S_\phi(\zeta) = \mathrm{diag}(\exp(\omega))^{-1}(\zeta-\omega)$; y en el caso *full-rank* $S_\phi(\zeta)=L^{-1}(\zeta-\mu)$

> *Demostración*
>
> Basta notar que el primer caso es un caso particular del segundo, y el segundo se probó en clase cuando simulamos normales desde normales estándar. _\square

Así pues, tras aplicar la transformación, la distribución variacional es
$$
q(\xi)=\mathcal{N}(\xi;0,\mathbb{I}) = \prod_{i=1}^p(\xi_i; 0, \mathbb{I})
$$

###### Definición.

La transformación S_\phi se llama *estandarización elíptica*.

Esta transformación convierte el problema de optimización en
$$
\phi^*=\underset{\phi}{\arg\max} \
\mathbb{E}_{\xi\sim\mathcal{N}(0,\mathbb{I})}\left[\log p\left(x, T^{-1}(S_\phi^{-1}(\xi)\right)
+ \log \left|\mathbb{J}_{T^{-1}}(S^{-1}_\phi(\xi))\right|\right]
+ \mathbb{H}(q_\phi)
$$

###### Teorema.

$$
\begin{eqnarray}
\nabla_\mu\mathrm{ELBO}(q) &=&
	\mathbb{E}_{\xi\sim\mathcal{N}(0,\mathbb{I})}\left[
	\nabla_\theta\log p(x,\theta)\nabla_\zeta T^{-1}(\zeta) +
	\nabla_\zeta\log|\mathrm{det}\mathbb{J}_{T^{-1}}(\zeta)|
	\right] \\ \nonumber \\

\nabla_\omega\mathrm{ELBO}(q) &=& \mathbb{E}_{\xi\sim\mathcal{N}(0,\mathbb{I})}\left[
	(\nabla_\theta\log p(x,\theta)\nabla_\zeta T^{-1}(\zeta) +
	\nabla_\zeta\log|\mathrm{det}\mathbb{J}_{T^{-1}}(\zeta)|)
	\xi^\top\mathrm{diag}(\exp(\omega))
	\right] +1 \\ \nonumber \\

\nabla_L\mathrm{ELBO}(q) &=& \mathbb{E}_{\xi\sim\mathcal{N}(0,\mathbb{I})}\left[
	(\nabla_\theta\log p(x,\theta)\nabla_\zeta T^{-1}(\zeta) +
	\nabla_\zeta\log|\mathrm{det}\mathbb{J}_{T^{-1}}(\zeta)|)\xi^\top
	\right] + (L^{-1})^\top
\end{eqnarray}
$$

> *Demostración*
>
> Apéndice C. $_\square$

Es por las ecuaciones (9)-(11) que ADVI trabaja con la clase de modelos diferenciables. Nótese que aunque no podíamos calcular el gradiente de la esperanza en (5), sí podemos calcular expresiones complicadas como (9)-(11). Esto se debe a la diferenciación automática (la otra mitad en el nombre de ADVI), que por ser más una genialidad computacional que estadística evitamos aquí entrar en detalles y los desarrollamos en el apéndice B. Basta saber que los gradientes en (9)-(11) son fáciles de evaluar, por lo que podemos usar descenso en gradiente.

#### Una rutina de optimización

Con una forma del gradiente conveniente para los métodos de Monte Carlo, basta elegir un algoritmo de optimización. En los modelos de altas dimensiones, un algoritmo debería adaptarse a la curvatura del espacio (siguiendo el trabajo de Sun Ichi Amari sobre la geometría de la información) y al mismo tiempo  dar pasos que decrezcan en magnitud suficientemente rápido. 

Los autores de ADVI proponen el siguiente esquema de ascenso en gradiente:

En la iteración i, sean \rho^{(i)} el tamaño del paso y g^{(i)} el gradiente.  Definimos
$$
\rho^{(i)}_k=\eta\times i^{-1/2+\epsilon}\times\left(\tau+\sqrt{s_k^{(i)}}\right)^{-1}
$$
donde aplicamos la actualización recursiva
$$
s_k^{(i)}=\alpha\left(g^{(i)}_k\right)^2+(1-\alpha)s_k^{(i-1)} \\
s_k^{(1)} = \left(g_k^{(1)}\right)^2
$$
y damos así el paso en el espacio $\mathrm{ELBO}$ 
$$
\theta^{(i)}=\theta^{(i-1)}+\rho^{(i)}g^{(i)}
$$
Antes de entrar en más detalle, la ecuación $(14$) muestra la intuición del algoritmo: estando en un punto  $\theta^{(i-1)}$, damos un paso tamaño $\rho^{(i)}$ en la dirección de mayor aumento de $\mathrm{ELBO}$. Este es el algoritmo de *ascenso en gradiente*. Aunque pueda parecer ingenuo, su popularidad se debe a que en gradientes en el formato de $(9)-(11)$, $\nabla_\theta f(\theta) = \mathbb{E}_x[h(x,\theta)]$, podemos aproximar la esperanza con una muestra pequeña (es un estimador de Monte Carlo) y así eficientar el proceso para grandes volúmenes de datos.

En $(12)$, el término $ \eta > 0$ determina la escala de la sucesión de pasos (o tasas de aprendizaje en jerga de aprendizaje). Los autores recomiendan elegir $\eta \in \{0.01, 0.1, 1, 10, 100\}$ usando todos los valores en un subconjunto pequeño de los datos y elegir el de convergencia más rápida. El término $i^{1/2+\epsilon}$ decae con la iteración para dar pasos cada vez más pequeños como exigen las condiciones de Robbins y Monro para garantizar convergencia. 

 El último factor se adapta a la curvatura del espacio \textrm{ELBO}, que por la reparametrización es distinto al original. Los valores de s guardan la información sobre los gradientes anteriores, y el factor \alpha determina qué tan importante es el valor histórico en comparación con el valor actual. La sucesión \left(s^{(n)}\right)_{n\in\mathbb{N}} converge a un valor no nulo.

Los valores de $\epsilon$ y $\tau$ son para estabilidad numérica, y los autores lo fijan en $10^{-16}$ y $1$ respectivamente. Finalmente podemos presentar el algoritmo.

#### Algoritmo

```pseudocode
advi(data=x, model=p(x,theta), mean_field=TRUE):
""" ADVI: Automatic differentiation variational inference.
	
	Parámetros:
	-----------
	data (data frame):
		Las observaciones.
	model (modelo de probabilidad):
		Especificación del modelo. Usualmente en algún software de programación 
		probabilística como Stan o PyMC3
	mean_field (boolean):
		TRUE si se quiere mean-field. FALSE si se quiere full-rank.
	
	Regresa:
	--------
	(arreglo):
		La media de la normal variacional
	(arreglo):
		El vector necesario para explicar la covarianza. Tiene longitud si
		se usa mean-field o p+p(p+1)/2 si se usa full-rank
	
"""
	eta <- Determinar por grid search con un subconjunto de datos
	mu[1] <- 0
	if mean_field:
		omega[1] <- 0
	else
		L[1] <- matriz_identidad
		
	while cambio en ELBO > tol: 
		eta <- muestrear M de normal multivariada estándar
		grad_mu(ELBO) <- aproximar (9) con integración de MC 
		if mean_field:
			grad_omega(ELBO) <- aproximar (10) con integración de MC
		else:
			grad_L(ELBO) <- aproximar (11) con integración de MC
		rho <- calcular con las ecuaciones (12) y (14)
		mu <- mu + rho*grad_mu(ELBO)
		if mean_field:
			omega <- omega + rho*grad_omega(ELBO)
		else
			L <- L + diag(rho)*grad_L(ELBO)
	return([mu, omega if mean_field else L])
```

Si se usa descenso en gradiente, ADVI tiene complejidad \mathcal{O}(nMp) por iteración, donde n es la cantidad de datos. En la variante estocástica con minibatch, pueden usarse b\ll n  puntos y baja a \mathcal{O}(bMp) para escalar a datos masivos.

#### Última nota

¿Cómo elegir entre full-rank y mean-field? Los autores recomiendan utilizar full-rank sólo cuando interesan las covarianzas (y varianzas) posteriores, pues cuando sólo interesa la media posterior, mean-field es suficientemente bueno y sustancialmente más rápido.



