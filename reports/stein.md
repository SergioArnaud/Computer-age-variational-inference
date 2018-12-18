# Apéndice D: Discrepancia de Stein

El *método de Stein* es un resultado general ampliamente utilizado en la teoría de probabilidad para obtener cotas sobre distancias entre distribuciones. Su uso durante el siglo XX fue exclusívamente teórico, pero en los últimos años se han adaptado las ideas principales del método para introducir sus resultados en el campo de *probabilistic programming*. A continuación se presenta una derivación de los principales resultados de la discrepancia de Stein y la discrepancia kernelizada de Stein.

###### Definición.

Sea  $p(x)$ una densidad continua y diferenciable (suave) con soporte $\mathcal{X}\subseteq \mathbb{R}^n $, definimos la *función score de Stein* de $p(x)$ como
$$
\mathrm{s}_p(x) = \frac{\nabla_xp(x)}{p(x)}=\nabla_x\log p(x)
$$

###### Definición.

Decimos que una función $f: \mathcal{X} \to \mathbb{R^d}$ , $f(x) = [f_1(x),...,f_d(x)]$ se encuentra en la *clase de Stein de $p$* si es suave y satisface
$$
\begin{align}
\int_{x \in \mathcal{X}} \nabla_x(f_i(x)p(x))dx = 0 && \forall i \in \{1,...,d\}
\end{align}
$$

###### Observación.

Podemos reescribir $(2)$ de la siguiente forma
$$
\int_{x \in \mathcal{X}} \nabla_x(f_i(x)p(x))dx = \mathbb{E}_{x\sim p}\left[\frac {\nabla_x(f_i(x)p(x)) }{p(x)}\right]
$$
Notemos que
$$
\begin{align}
\frac {\nabla_x(f_i(x)p(x)) }{p(x)}  &= \frac{\nabla_x f_i(x)p(x) + f_i(x) \nabla_x p(x)}{p(x)}\nonumber \\ \nonumber \\
&=  \nabla_x f_i(x) + f_i(x) \mathrm{s}_p
\end{align}
$$
De esta forma, si $f$ está en la clase de Stein de $p$
$$
\begin{align}
\mathbb{E}_{x \sim p} [\nabla f_i(x) + f_i(x) \mathrm{s}_p] = 0 && \forall i \in \{1,...,d\}
\end{align}
$$

###### Definición.

Con base en $(4)$ definimos el *operador de Stein* $\mathcal{A}_p$ 
$$
\mathcal{A}_pf(x) = \mathrm{s}_p(x)f(x)^\top + \nabla_xf(x)
$$
Y nombramos a $(5)$  la *identidad de Stein*
$$
\mathbb{E}_{x \sim p} [\mathcal{A}_pf(x)] = 0 \nonumber
$$
Consideremos $q(x)$, una densidad suave con soporte en $\mathcal{X}$ y tomemos la esperanza de el operador de Stein pero ahora bajo $x \sim q$. A diferencia de lo ocurrido con la identidad de Stein, $\mathbb{E}_{x \sim q} [\mathcal{A}_pf(x)]$ es positiva en el caso general y, como mostraremos a continuación, su valor induce una noción de cercanía entre las densidades $q$ y $p$.

###### Lema.

$$
\mathbb{E}_{x \sim q} [\mathcal{A}_pf(x)] = \mathbb{E}_{x\sim q}\left[(\mathrm{s}_p(x)-\mathrm{s}_q(x))f(x)^\top\right]
$$

> *Demostración*
>
> Basta notar que $\mathbb{E}_{x \sim q} [\mathcal{A}_qf(x)] = 0$ y proceder de la siguiente manera
> $$
> \begin{align*}
> \mathbb{E}_{x \sim q} [\mathcal{A}_pf(x)] &= \mathbb{E}_{x \sim q} [\mathcal{A}_pf(x) - \mathcal{A}_qf(x)] \\ \\
> &= \mathbb{E}_{x\sim q}[  \mathrm{s}_p(x)f(x)^\top + \nabla_xf(x) -\mathrm{s}_q(x)f(x)^\top - \nabla_xf(x) ] \\ \\ 
> &= \mathbb{E}_{x \sim q}[(\mathrm{s}_p-\mathrm{s}_q)f(x)^\top] \ _\square
> \end{align*}
> $$
>

El resultado del lema anterior da pie a introducir la discrepancia de Stein.

###### Definicion.

Sea $\mathcal{F}$ una familia de funciones en la clase de Stein de $p$, definimos la *discrepancia de Stein* como
$$
\mathbb{\hat{S}}(q,p) = \sup_{f\in\mathcal{F}}\ \mathbb{E}_{x \sim q} [\mathcal{A}_pf(x)]
$$

###### Observación.

Es importante elegir a la familia $\mathcal{F}$ de forma que se garantice $\mathbb{\hat{S}}(q,p)$ > 0 si $p \not = q$.

La discrepancia de Stein presenta un problema de optimización variacional sumamente complicado, que usualmente es imposible de resolver explícitamente. Sin embargo, si la solución al problema fuera lineal, digamos, $f = \sum_iw_if_i$ con una base conocida $\{f_i\}$, bastaría resolver el problema para los coeficientes, pues
$$
\mathbb{E}_{x\sim q}[\mathcal{A}_pf(x)] = 
\mathbb{E}_{x\sim q}\left[\mathcal{A}_p\sum_iw_if_i(x)\right] = 
\sum_iw_i\mathbb{E}_{x\sim q}[\mathcal{A}_pf_i(x)] =
\sum_iw_i\beta_i
\
$$
y en este caso el problema es $\{\max w^\top \beta \ \ s.a. \|w\|\leq1\}$. En consecuencia, queremos una $\mathcal{F}$ suficientemente rica para poder amar estas combinaciones lineales. Una buena solución es elegir la bola unitaria del RKHS asociado a un kernel positivo $\mathrm{K}$. Para los resultados presentados a continuación se recomienda la lectura del Apéndice C que presenta una breve introducción a los kérneles positivos definidos y los RKHS, pero basta aquí saber que el espacio $\mathcal{H}$ incluye todas las funciones de la forma $g(x) = \sum_iw_i\mathrm{K}(x,x_i)$ para conjuntos finitos $\{x_i\}\subseteq \mathcal{X}$ con norma $\|g\|_\mathcal{H}=\sum_{i,j}w_iw_j\mathrm{K}(x_i, x_j)$. 

###### Definición.

La *discrepancia de Stein kernelizada* entre las distribuciones $p$ y $q$ está dada por
$$
\mathbb{S}(q,p) = \mathbb{E}_{x,x' \sim q}\left[\delta_{p,q}(x)^\top\mathrm{K}(x,x')\delta_{p,q}(x')\right]
$$
Donde $\delta_{p,q}(x) := \mathrm{s}_p(x) - \mathrm{s}_q(x)$

###### Observación.

Si $p$ es una distribución de colas pesadas, $\mathbb{S}$ puede presentar comportamientos indeseables para una medida de discrepancia. Por ejemplo:  $\mathbb{S}(p,q) = 0 $, con  $p\not = q$.

###### Definición.

Decimos que un kernel $\mathrm{K}(x,x')$ se encuentra en la *clase de Stein de p* si sus segundas derivadas parciales son continuas y tanto $\mathrm{K}(x,\cdot)  $ como $\mathrm{K}(\cdot,x)$ se encuentran en la clase de Stein de $p$ para cualquier $x$ fijo.

######Observación

Usualmente se utiliza el kernel *RBF* 
$$
\mathrm{K}(x,y) = \exp\left(-\frac{\Vert x - y\Vert^2}{2\sigma^2}\right)
$$
Que está en la clase de Stein de $p$ si es una densidad suave con soporte $\mathcal{X} = \mathbb{R}^n$

###### Teorema.

Sea $\mathcal{H}$ el RKHS definido por un kernel positivo $\mathrm{K}$ en la clase de Stein de $p$ y consideremos $\beta(x') := \mathbb{E}_{x \sim q} [ \mathcal{A}_p K_{x}(x')]$ entonces:
$$
\mathbb{S}(q,p) = || \beta ||_{\mathcal{H}}^2
$$
Más aún, $\langle f, \beta \rangle_{\mathcal{H}} = \mathbb{E}_{x\sim q}[\text{traza}(\mathcal{A}_pf)]$ con 
$$
\mathbb{S}(q,p) = \max_{\Vert f \Vert_\mathcal{H} \leq 1}\{\mathbb{E}_{x \sim q}[\text{traza}(\mathcal{A}_pf)]\}
$$
Donde el máximo se obtiene cuando $ f = \frac{\beta}{\Vert \beta \Vert_\mathcal{H}}$

> Demostración.
>
> Para demostrar $(12)$ procedemos de la siguiente manera
> $$
> \begin{align*}
> \mathbb{S}(q,p) &= \mathbb{E}_{x,x' \sim q}\left[ (\mathrm{s}_p(x) - \mathrm{s}_q(x))^\top\mathrm{K}(x,x')(\mathrm{s}_p(x') - \mathrm{s}_q(x'))\right] \\ \\
> &= \mathbb{E}_{x,x' \sim q}\left[ (\mathrm{s}_p(x) - \mathrm{s}_q(x))^\top\langle K_x, K_{x'} \rangle_\mathcal{H}(\mathrm{s}_p(x') - \mathrm{s}_q(x'))\right] & \text{Propiedad de reproducibilidad} \\ \\
> &= \left \langle\mathbb{E}_{x \sim q}\left[ (\mathrm{s}_p(x) - \mathrm{s}_q(x))^\top K_x] \ , \ \mathbb{E}_{x' \sim q} [K_{x'} (\mathrm{s}_p(x') - \mathrm{s}_q(x'))\right] \right \rangle_\mathcal{H}  \\ \\
> &=  \left\Vert \mathbb{E}_{x \sim q}[ (\mathrm{s}_p(x) - \mathrm{s}_q(x))^\top K_{x}] \right\Vert_{\mathcal{H}}^2  \\ \\
> &= \left\Vert \mathbb{E}_{x \sim q}[ \mathcal{A_p}K_{x}] \right\Vert_{\mathcal{H}}^2
> \end{align*}
> $$
> Para $(13)$ es suficiente ver que $\langle f, \beta \rangle_{\mathcal{H}} = \mathbb{E}_{x\sim p}[\text{traza}(\mathcal{A}_qf)]$. Basta utilizar las propiedades del producto interno y recordar la propiedad de reproducibilidad de los RKHS, es decir, $\langle f, \mathrm{K}_x \rangle_{\mathcal{H}} = f(x)$ para concretar la demostración. $_\square$

