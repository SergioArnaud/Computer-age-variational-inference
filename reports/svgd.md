## SVGD: Stein Variational Gradient Descent

SVGD es un algoritmo determinístico de propósito general para realizar inferencia variacional introducido por Qiang Liu y Dilin Wang en el 2016. En terminos generales, el algoritmo utiliza un resultado teórico que conecta la divergencia de Kullback-Leibler con la discrepancia de Stein para transportar iterativamente un conjunto de partículas hacia la distribución objetivo, esto se lleva a cabo al realizar un proceso de descenso por gradiente funcional en un RKHS.

####Discrepancia de Stein y divergencia de Kullback–Leibler

Un desarrollo detallado de la discrepancia kernelizada de Stein se encuentra en el apéndice D, se recomienda su lectura antes de continuar (queda advertido) puesto que es de suma importancia para los resultados presentados a continuación.

Comencemos por recordar la discrepancia kernelizada de Stein.

###### Teorema.

Sea $\mathcal{H}$ el RKHS definido por un kernel positivo $\mathrm{K}(x,x')$ en la *clase de Stein de $p$* y consideremos $\phi(x') := \mathbb{E}_{x \sim q} [ \mathcal{A}_p K_{x}(x')]$ donde $\mathcal{A}_p$ es el *operador de Stein*, entonces
$$
\mathbb{S}(p,q) = || \phi ||_{\mathcal{H}}
$$
Donde $\mathbb{S}(p,q)$ es la discrepancia kernelizada de Stein. Más aún, $\langle f, \phi\rangle_{\mathcal{H}} = \mathbb{E}_{x\sim q}[\text{traza}(\mathcal{A}_pf)]$ de forma que 
$$
\mathbb{S}(p,q) = \max_{f\in\mathcal{H}}\{\mathbb{E}_{x \sim q}[\text{traza}(\mathcal{A}_pf)] \quad \text{donde} \quad \Vert f \Vert_\mathcal{H} \leq 1 \}
$$
Y el máximo se obtiene cuando  $ f = \frac{\phi}{\Vert \phi \Vert_\mathcal{H}}$ 

El siguiente teorema enuncia una impresionante conexión entre la divergencia de Kullback–Leibler y la discrepancia de Stein kernelizada, su resultado es la base para el desarrollo de SVGD.

###### Teorema.

Sea $T(\theta) = \theta + \epsilon \phi(\theta)​$ y $q_T(z)​$ la densidad de $z = T(x)​$ cuando $x \sim q​$. Entonces
$$
\nabla_{\epsilon} \left.D_{KL}(q_T || p) \right |_{\epsilon = 0} = - \mathbb{E}_{x\sim q}[\text{traza}(\mathcal{A_p}\phi(x))]
$$
Donde $\mathcal{A_p}$ es el operador de Stein. $_\square$

Con base en el teorema anterior y la discrepancia de Stein es posible encontrar de forma explícita la dirección de la perturbación que ocasiona el mayor descenso en la divergencia de Kullback-Leibler.

###### Corolario

Consideremos todas las direcciones de la perturbación $\phi$ en la bola $\mathcal{B} = \{\phi \in \mathcal{H}: \Vert \phi \Vert_{\mathcal{H}}\leq \mathbb{S}(p,q) \}$, la dirección de mayor descenso de $D_{KL}(q_T || p)$ es
$$
\phi_{q,p}^*(\cdot) = \mathbb{E_{x \sim q}}[K_x \nabla_x \log p(x) + \nabla_x K_x]
$$

> *Demostración*.
>
> Basta recordar que 
> $$
> \underset{\phi\in\mathcal{H}}{\arg\max}\{\mathbb{E}_{x \sim q}[\text{traza}(\mathcal{A}_p\phi)], \  \Vert \phi \Vert_\mathcal{H} \leq 1 \} = \frac{\mathbb{E}_{x \sim q} [ \mathcal{A}_p K_{x}(\cdot)]}{\Vert \mathbb{E}_{x \sim q} [ \mathcal{A}_p K_{x}] \Vert^2_{\mathcal{H}}} \nonumber
> $$
> Asimismo
> $$
> \mathbb{S}(p,q) = \Vert \mathbb{E}_{x \sim q} [ \mathcal{A}_p K_{x}] \Vert_{\mathcal{H}} \nonumber
> $$
> Luego
> $$
> \underset{\phi\in\mathcal{H}}{\arg\max}\{\mathbb{E}_{x \sim q}[\text{traza}(\mathcal{A}_p\phi)], \  \Vert \phi \Vert_\mathcal{H} \leq \mathbb{S}(p,q) \} = \mathbb{E}_{x \sim q} [ \mathcal{A}_p K_{x}] \nonumber
> $$
> Considerando el resultado $(3)$ concluye la demostración $_\square$ 

###### Observación. 

El resultado recién obtenido sugiere un proceso iterativo que permite transformar una distribición inicial $q_0$ a la distribución objetivo $p$. 

Comenzamos por aplicar la transformación $T_0^*(x) = x + \epsilon_0 \phi_{q_0, p}^*(x)$ sobre $q_0$ que disminuye la divergencia de Kullback–Leibler en $\epsilon_0 \mathbb{S}(q_0,p)$, esto produce una nueva distribución $q_1(x) = q_{0T_0}(x)$. Procedemos inductivamente para definir una sucesión de distribuciones $\{q_k\}_{k\in\mathbb{N}}$ de la siguiente forma
$$
\begin{align}
q_{k+1} = q_{kT_k}  && T_k(x) = x + \epsilon_k \phi_{q_k,p}^*(x).
\end{align}
$$

Dicha sucesión eventualmente converge a la distribución objetivo $p$.

#### Interpretación funcional

Podemos reinterpretar $(4)$ como un gradiente funcional en un RKHS donde la transformación $T^*(x) = x + \epsilon \phi_{q,p}^*(x)$ es equivalente a dar un paso de *descenso por gradiente funcional* en el RKHS correspondiente. 

###### Teorema

Sea $T(x) = x + f(x)$, $f \in \mathcal{H}$ y $q_T$ la densidad de $z = T(x)$ con $x \sim q$.
$$
\begin{align}
\nabla_f \left.D_{KL}(q_T \Vert p )\right|_{f=0} = - \phi_{q,p}^*(x), && \Vert \phi_{q,p}^* \Vert_{\mathcal{H}}^2 = \mathbb{S}(q,p).
\end{align}
$$

#### Implementación computacional

Para implementar el proceso iterativo definido en $(5)$ es necesario obtener $\phi_{q_k,p}^*$ en cada iteración, sin embargo, su cálculo explícito presenta complicaciones prácticas.

Por esta razón, se propone tomar una muestra $\{x_1^0, ..., x_n^0\}$ de la distribución inicial $q_0$ de forma que en la k-ésima iteración del proceso se aproxime  $\phi_{q_{k+1},p}^*(\cdot)  = \mathbb{E_{x \sim q_{k+1}}}[K_x \nabla_x \log p(x) + \nabla_x K_x]$ con la media muestral de $\{x_1^k, ..., x_n^k\}$ de la siguiente forma
$$
\hat{\phi_{k}^*}(x) = \frac{1}{n}\sum_{j=1}^n [K(x_j^k,x)\nabla_{x_j^k} \log p(x_j^k) + \nabla_{x_j^k}K(x_j^k,x)]
$$
Los términos expresión $(7)$ presentan comportamientos opuestos, por un lado $K(x_j^k,x)\nabla_{x_j^k} \log p(x_j^k) $ transporta a las partículas hacia las regiones de mayor probabilidad, por otro lado $\nabla_{x_j^k}K(x_j^k,x)$ actúa como una fuerza repulsiva que previene a las partículas de quedar estancadas en las regiones de mayor concentración para generar cierta dispersión.

Observemos finalmente que el procedimiento iterativo no depende de la distribución inicial $q_0$ de forma que la muestra inicial  $\{x_1^0, ..., x_n^0\}$ puede ser obtenida por medio de un proceso determinista o como una muestra de cualquier distribución de probabilidad.

De esta manera, podemos presentar el algoritmo.

```pseudocode
svgd(model=p(x), kernel=RBF(), n_particulas=100, n_iter=10000, q_0=Normal(0,1)):
""" SVGD: Stein Variational Gradient Descent

    Parámetros:
    -----------
    model (modelo de probabilidad):
        Especificación del modelo. Usualmente en algún software de programación 
        probabilística como Stan o PyMC3.
    kernel (función, f(histograma) -> (k(x,.), \nabla_x k(x,.)):
    	Kernel del proceso, por default es el kernel RBF.
	n_iter (int):
		Número de iteraciones
	n_particulas (int)
		Número de partículas
	
	Regresa:
	--------
	(arreglo):
		Un conjunto de partículas que aproxima la distribución objetivo
"""
	x = obtener una muestra de tamaño n_particulas de la distribución inicial
    	especificada q_0
	for iter in n_iter:
		phi(x) = Calcular la ecuación (5)
		x = x + eps*phi(x)
	return x
```

#### Implementación óptima

El mayor problema en términos computaciones consiste en calcular el gradiente $\nabla_x \log p(x_i)$ para $i\in\{1,...,n\}$, si se tiene $p(x) \propto p_0(x) \prod_{k=1}^N p(D_k \vert x)$ una posible optimización es aproximar $\nabla_x \log p(x_i)$ de la siguiente forma
$$
\nabla_x \log p(x_i) \approx \log p_0(x) + \frac{N}{\vert \Omega \vert}\sum_{k \in \Omega} \log p(D_k \vert x)
$$
Asimismo se puede paralelizar la evaluación de los gradientes y aproximar la suma en $(5)$ evaluando únicamente una muestra delas partículas.

#### Relación con el máximo a posteriori

¿Qué ocurre si consideramos una muestra inicial de tamaño 1, digamos $\{x_0^0\}$? 

Retomando $(7)$ obtenemos 
$$
\begin{align*}
\hat{\phi_{k}^*}(x) &= \frac{1}{n}\sum_{j=1}^n [K(x_j^k,x)\nabla_{x_j^k} \log p(x_j^k) + \nabla_{x_j^k}K(x_j^k,x)] \\ \\
&= K(x_0^k,x) \nabla_{x_0^k} \log p(x_0^k) + \nabla_{x_0^k}K(x_0^k,x)
\end{align*}
$$
De forma que la función de actualización está data por
$$
\begin{align*}
x_0^{k+1} &= x_0^k + \epsilon_k \phi_{q_k,p}^*(x_0^k)\\ \\
&= x_0^k + \epsilon_k\left[K\left(x_0^k,x_0^k\right) \nabla_{x_0^k} \log p(x_0^k) + \nabla_{x_0^k}K(x_0^k,x_0^k)\right] \\ \\
&= x_0^k + \epsilon_k\left[\Vert x_0^k \Vert_\mathcal{H}^2 \nabla_{x_0^k} \log p(x_0^k) + \nabla_{x_0^k}K(x_0^k,x_0^k)\right]
\end{align*}
$$
Más aún, si suponemos $\nabla_{x}K(x,x) = 0$  -propiedad que cumple el kernel RBF y la mayoría de los kernels positivos definidos- obtenemos la siguiente expresión
$$
x_0^{k+1} = x_0^k + \epsilon_k\left[\Vert x_0^k \Vert_\mathcal{H}^2 \nabla_{x_0^k} \log p(x_0^k) \right]
$$
Asi pues, el algoritmo se reduce al procedimiento típico de ascenso por gradiente para obtener el *máximo a posteriori* (MAP por sus siglas en ingles) de la distribución objetivo. Éste método destaca puesto que utiliza una sola partícula en contraste con los métodos de Monte Carlo que necesitan muestras de mayor tamaño.

#### Convergencia

Como vimos previamente, la sucesión de densidades construida en $(5)$ efectivamente converge a la distribución objetivo $p$, no obstante, no es claro si la aproximación propuesta en $(7)$ también lo hace. 

Sea $\hat{q_{0}}$ la distribución empírica de $\{x_1^0, ..., x_n^0\}$, si $n\to \infty$ entonces $\hat{q_{0}}$ converge a $q_0$ con probabilidad 1 ¿Qué se puede decir sobre la convergencia de la distribición empírica obtenida en la k-ésima iteración, $\hat{q}_ {k+1}$ a $q_ {k+1}$?  

Sea $\Phi_k$ el mapeo tal que $q_ {k+1} = \Phi_k(q_k)$, entonces la densidad empírica en la $k$-ésima iteración se puede escribir como $\hat{q}_ {k+1} = \Phi_k(\hat{q}_k)$.  Asimismo, $\hat{q}_0 \to q_0$ por lo que bastaría que $\Phi_k$ presenta propiedades *deseables* para que $\hat{q}_{k+1} \to q_{k+1}$.

Formalmente, si se tiene que la distribución inicial $q_0$ tiene densidades sueves y divergencia de Kullback Leibler finita con $p$ se puede probar que $\hat{q}_k$ converge débilmente a $q_k$, asimismo la sucesión $\left\{D_{KL}(\hat{q}_i \Vert p )\right\}_{i=1}^\infty$ decrece monótonamente, propiedad que nos permite establecer la convergencia.

Finalmente, establecer la tasa explícita de convergencia de SVGD es un problema abierto y un área activa de investigación.




