## ADVI: inferencia variacional con diferenciación automática



ADVI es un algoritmo introducido, entre otros, por Andrew Gelman y David Blei en 2016. En resumen, dada la especificación de un modelo $p(x, \theta)$, ADVI lo transforma a $p(x, \zeta)$ (ajustando por el mapa $\theta \mapsto \zeta$ en la nueva conjunta) donde $\zeta \in \mathbb{R}$. Así elimina las restricciones sobre el parámetro $\theta$, y es posible fijar $\mathscr{Q}$ de antemano y utilizarla para cualquier modelo. Después, ADVI transforma la función objetivo a un valor esperado sobre la función propuesta, de manera que pueda aproximarse con métodos de Monte Carlo, y resuelve la optimización con una variante de gradient descent. 

Comencemos por definir los modelos que ADVI puede resolver: los modelos diferenciables.

###### Definición.

Un modelo de probabilidad $p(x, \theta)$ es *diferenciable* si $\theta$ es continua y existe $\nabla_\theta\log p(x, \theta)$ en el soporte $\Theta \subseteq \mathbb{R}^p$  de $\theta$. 

Esta clase es más extensa de lo que puede parecer a primera vista, pues considerando que se puede marginalizarse sobre las variables discretas incluye no sólo a los modelos lineales generalizados y procesos gaussianos, sino también de modelos de mezclas, latent Dirichlet allocation y hidden Markov models.

Estudiemos con más detalle el funcionamiento de ADVI.

#### Transformación automática de variables

El primer paso del algoritmo es transformar los parámetros del modelo, de $\theta \in \Theta \subseteq \mathbb{R}^p$ a $T(\theta) = \zeta \in \mathbb{R}^p$. Habiendo elegido $T$, la nueva conjunta es 
$$
p(x,\zeta) = p(x, T^{-1}(\zeta))|\det \mathbb{J}_{T^{-1}}(\zeta)|
$$
donde $\mathbb{J}_f$ es la derivada de $f$. Los resultados no son invariantes ante la elección de $T$, pero calcular la transformación óptima es un problema equivalentemente difícil, pues depende de la distribución de $\theta|X$. Hay trabajo en curso para mejorar la elección, pero por ahora basa saber que las implementaciones computacionales (Stan y PyMC3) se encargan de esto.

#### El problema variacional en $\mathbb{R}^p$ 

ADVI implementa dos de las familias $\mathscr{Q}$ más prominentes en la literatura para $\Theta=\mathbb{R}^p$: 

###### Definición.

La familia *mean-field gaussian* es el conunto de distribuciones de la forma
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

Analizando de manera rápida el *bias-variance tradeoff* de la elección de modelo, usando full rank el espacio de parámetros sube de $O(p)$ a $O(p^2)$, por lo que la estimación se complica, pero sí nos permite expresar correlación entre las entradas de $\zeta$, una estructura mucho más rica.

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
> Basta notar que el primer caso es un caso particular del segundo, y el segundo se probó en clase cuando simulamos normales desde normales estándar. $_\square$

Así pues, tras aplicar la transformación, la distribución variacional es
$$
q(\eta)=\mathcal{N}(\eta;0,\mathbb{I}) = \prod_{i=1}^p(\eta_i; 0, \mathbb{I})
$$

###### Definición.

La transformación $S_\phi$ se llama *estandarización elíptica*.

Esta transformación convierte el problema de optimización en 
$$
\phi^*=\underset{\phi}{\arg\max} \ 
\mathbb{E}_{\eta\sim\mathcal{N}(0,\mathbb{I})}\left[\log p\left(x, T^{-1}(S_\phi^{-1}(\eta)\right)
+ \log \left|\mathbb{J}_{T^{-1}}(S^{-1}_\phi(\eta))\right|\right]  
+ \mathbb{H}(q_\phi)
$$

###### Teorema

$$
\nabla_\mu\mathrm{ELBO}(q)=
	\mathbb{E}_{\eta\sim\mathcal{N}(0,\mathbb{I})}\left[
	\nabla_\theta\log p(x,\theta)\nabla_\zeta T^{-1}(\zeta) +
	\nabla_\zeta\log|\mathrm{det}\mathbb{J}_{T^{-1}}(\zeta)|
	\right] \\
\nabla_\omega\mathrm{ELBO}(q) = \mathbb{E}_{\eta\sim\mathcal{N}(0,\mathbb{I})}\left[
	(\nabla_\theta\log p(x,\theta)\nabla_\zeta T^{-1}(\zeta) +
	\nabla_\zeta\log|\mathrm{det}\mathbb{J}_{T^{-1}}(\zeta)|)
	\eta^\top\mathrm{diag}(\exp(\omega))
	\right] +1 \\
\nabla_L\mathrm{ELBO}(q) = \mathbb{E}_{\eta\sim\mathcal{N}(0,\mathbb{I})}\left[
	(\nabla_\theta\log p(x,\theta)\nabla_\zeta T^{-1}(\zeta) +
	\nabla_\zeta\log|\mathrm{det}\mathbb{J}_{T^{-1}}(\zeta)|)\eta^\top
	\right] + (L^{-1})^\top
$$

> *Demostración*
>
> Apéndice C. $_\square$

#### Una rutina de optimización

Con una forma del gradiente conveniente para los métodos de Monte Carlo, basta elegir un algoritmo de optimización. Los autores de ADVI proponen el siguiente:

En la iteración $i$, sean $\rho^{(i)}$ el tamaño del paso y $g^{(i)}$ el gradiente.  Definimos
$$
\rho^{(i)}_k=\xi+i^{-1/2+\epsilon}\times\left(\tau+\sqrt{s_k^{(i)}}\right)^{-1}
$$
donde aplicamos la actualización recursiva
$$
s_k^{(i)}=\alpha g^{(i)}_k+(1-\alpha)s_k^{(i-1)} \\
s_k^{(1)} = g_k^{(i)}
$$
