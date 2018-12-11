# Apéndice D: Discrepancia de Stein

El *método de Stein* es un resultado general ampliamente utilizado en la teoría de probabilidad para obtener cotas sobre distancias entre distribuciones. Su uso durante el siglo XX fue exclusívamente teórico, sin embargo, durante los últimos años se han adaptado las ideas principales del método para introducir sus resultados en el campo de *probabilistic programming*. A continuación se presenta una derivación de los principales resultados de la discrepancia de Stein y la discrepancia kernelizada de Stein.

###### Definición.

Sea  $p(x)$ una densidad continua y diferenciable (suave) con soporte $\mathcal{X}\subseteq \mathbb{R}^n $, definimos la *función score de Stein* de $p(x)$ como
$$
\mathrm{s}_p = \frac{\nabla_xp(x)}{p(x)}
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

Con base en $(4)$ definimos el *operador de Stein*, $\mathcal{A}_p$ de la siguiente forma
$$
\mathcal{A}_pf(x) = \mathrm{s}_p(x)f(x)^\top + \nabla_xf(x)
$$
Asimismo, la expresión obtenida en $(5)$ es conocida como la *identidad de Stein*
$$
\mathbb{E}_{x \sim p} [\mathcal{A}_pf(x)] = 0 \nonumber
$$
Consideremos $q(x)$, una densidad suave con soporte en $\mathcal{X}$ y tomemos la esperanza de el operador de Stein pero ahora bajo $x \sim q$. A diferencia de lo ocurrido con la identidad de Stein, $\mathbb{E}_{x \sim q} [\mathcal{A}_pf(x)]$ es positiva en el caso general y, como mostraremos a continuación, su valor induce una noción de cercanía entre las densidades $q$ y $p$.

###### Lema.

$$
\mathbb{E}_{x \sim q} [\mathcal{A}_pf(x)] = \mathbb{E}_q[(\mathrm{s}_p-\mathrm{s}_q)f(x)^\top]
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
\mathbb{\hat{S}}(p,q) = \max_{f\in\mathcal{F}}(\mathbb{E}_{x \sim q} [\mathcal{A}_pf(x)])
$$

###### Observación.

Es importante elegir a la familia $\mathcal{F}$ de forma que se garantice $\mathbb{\hat{S}}(p,q)$ > 0 si $p \not = q$.

La discrepancia de Stein presenta un problema de optimización variacional sumamente complicado que, usualmente, es imposible de resolver explícitamente. En consecuencia, definimos la *discrepancia de Stein kernelizada*, que nos permitirá obtener de una manera más sencilla la discrepancia de Stein al tomar $\mathcal{F}$ como la bola en el RKHS asociado con un kernel positivo definido suave. Para los resultados presentados a continuación se recomienda la lectura del Apéndice C que presenta una breve introducción a los kernels positivos definidos y los RKHS.

###### Definición.

La *discrepancia de Stein kernelizada* entre las distribuciones $p$ y $q$ está dada por
$$
\mathbb{S}(p,q) = \mathbb{E}_{x,x' \sim p}\left[\delta_{q,p}(x)^T\mathrm{K}(x,x')\delta_{q,p}(x')\right]
$$
Donde $\delta_{q,p}(x) := \mathrm{s}_q(x) - \mathrm{s}_p(x)$

###### Observación.

Si $p$ es una distribución de colas pesadas, $\mathbb{S}(p,q)$ puede presentar comportamientos que no son deseables para una medida de discrepancia, por ejemplo:  $\mathbb{S}(p,q) = 0 , \  p\not = q$.

###### Definición.

Decimos que un kernel $\mathrm{K}(x,x')$ se encuentra en la *clase de Stein de p* si sus segundas derivadas parciales son continuas y tanto $\mathrm{K}(x,\cdot)  $ como $\mathrm{K}(\cdot,x)$ se encuentran en la clase de Stein de $p$ para cualquier x fijo.

######Observación

Usualmente se utiliza el kernel *RBF* 
$$
\mathrm{K}(x,y) = \exp\left(-\frac{\Vert x - y\Vert^2}{2\sigma^2}\right)
$$
Que está en la clase de Stein de $p$ si es una densidad suave con soporte $\mathcal{X} = \mathbb{R}^n$

###### Teorema.

Sea $\mathcal{H}$ el RKHS definido por un kernel positivo $\mathrm{K}(x,x')$ en la clase de Stein de $p$ y consideremos $\beta(x') := \mathbb{E}_{x \sim p} [ \mathcal{A}_q K_{x}(x')]$ entonces:
$$
\mathbb{S}(p,q) = || \beta ||_{\mathcal{H}}^2
$$
Más aún, $\langle f, \beta \rangle_{\mathcal{H}} = \mathbb{E}_{x\sim p}[\text{traza}(\mathcal{A}_qf)]$ con 
$$
\sqrt{\mathbb{S}(p,q)} = \max_{f\in\mathcal{H}}\{\mathbb{E}_{x \sim p}[\text{traza}(\mathcal{A}_qf)] \quad \text{donde} \quad \Vert f \Vert_\mathcal{H} \leq 1 \}
$$
Donde el máximo se obtiene cuando $ f = \frac{\beta}{\Vert \beta \Vert_\mathcal{H}}$

> Demostración.
>
> Para demostrar $(11)$ procedemos de la siguiente manera
> $$
> \begin{align*}
> \mathbb{S}(p,q) &= \mathbb{E}_{x,x' \sim p}\left[ (\mathrm{s}_q(x) - \mathrm{s}_p(x))^\top\mathrm{K}(x,x')(\mathrm{s}_q(x') - \mathrm{s}_p(x'))\right] \\ \\
> &= \mathbb{E}_{x,x' \sim p}\left[ (\mathrm{s}_q(x) - \mathrm{s}_p(x))^\top\langle K_x, K_{x'} \rangle_\mathcal{H}(\mathrm{s}_q(x') - \mathrm{s}_p(x'))\right] & \text{Propiedad de reproducibilidad} \\ \\
> &= \left \langle\mathbb{E}_{x \sim p}\left[ (\mathrm{s}_q(x) - \mathrm{s}_p(x))^\top K_x] \ , \ \mathbb{E}_{x' \sim p} [K_{x'} (\mathrm{s}_q(x') - \mathrm{s}_p(x'))\right] \right \rangle_\mathcal{H}  \\ \\
> &=  \left\Vert \mathbb{E}_{x \sim p}[ (\mathrm{s}_q(x) - \mathrm{s}_p(x))^\top K_{x}] \right\Vert_{\mathcal{H}}^2  \\ \\
> &= \left\Vert \mathbb{E}_{x \sim p}[ \mathcal{A_q}K_{x}] \right\Vert_{\mathcal{H}}^2
> \end{align*}
> $$
> Para $(12)$ es suficiente ver que $\langle f, \beta \rangle_{\mathcal{H}} = \mathbb{E}_{x\sim p}[\text{traza}(\mathcal{A}_qf)]$. Basta utilizar las propiedades del producto interno y recordar la propiedad de reproducibilidad de los RKHS, es decir, $\langle f, \mathrm{K}_x \rangle_{\mathcal{H}} = f(x)$ para concretar la demostración. $_\square$