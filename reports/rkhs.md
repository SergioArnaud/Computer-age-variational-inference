#Apéndice C: Espacios de Hilbert con kernel reproductor 

#### Kernels positivos definidos

###### Definición.

Un *kernel positivo definido* en $\mathcal{X}​$ es una función simétrica $\mathrm{K}: \mathcal{X} \times \mathcal{X} \to \mathbb{R}​$ tal que para todo $n \in \mathbb{N}​$, $\{x_i,\dots,x_n\} \subseteq \mathcal{X}​$, y $\{c_i ... c_n \} \subseteq \mathbb{R}​$ se cumple
$$
\sum_{i,j=1}^n c_ic_i K(x_i,x_j) \geq 0
$$

###### Observación.

La matriz $\mathbb{G}$ con $ (\mathbb{G})_{ij} = \mathrm{K}(x_i,x_j) $ también conocida como la *matriz de Gram* es positiva semidefinida.

###### Proposición.

Si $\mathrm{K}_1, \mathrm{K}_2 : \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ son kernels positivos definidos, entonces los siguientes también lo son

- $a\mathrm{K}_1 + b\mathrm{K}_2$ con $ a,b \geq 0$
- $\mathrm{K}_1\mathrm{K}_2$ 
- $\lim\limits_{i\to\infty}  {K}_i$

###### Proposición.

Se $\mathrm{K}$ es un kernel positivo definido y $f:\mathcal{X} \to \mathbb{R}$ una función arbitraria, entonces $f(x) \mathrm{K}(x,y)f(y)$ y    $f(x)f(y)$ son positivos definidos.

###### Proposición.

Sea $(\mathrm{V} , \langle\cdot,\cdot\rangle)$ un espacio vectorial con producto interior, entonces para todo mapeo $\Phi : \mathcal{X} \to \mathrm{V}$, $\langle\Phi(x),\Phi(y)\rangle$ es un kernel definido positivo. 

> *Demostración*
>
> Sean $\{x_i,\dots,x_n\} \subseteq \mathcal{X}$, y $\{c_i ... c_n \} \subseteq \mathbb{R}$
> $$
> \begin{align*}
> \sum_{i,j=1}^n c_ic_k\mathrm{K}(x_i,x_j) &= \left\langle\sum_{i=1}^nc_i\Phi(x_i),\sum_{i=1}^nc_k\Phi(x_j) \right\rangle \\
> &= \left\Vert \sum_{i=1}^n c_i\Phi(x_i)\right\Vert \\
> &\geq 0 \ _\square
> \end{align*}
> $$
>

###### Ejemplos.

$$
\begin{align*}
\text{Kernel lineal} && \mathrm{K}(x,y) &= x^Ty  \\
\text{Kernel Gaussiano} && \mathrm{K}(x,y) &= exp\left(-\frac{1}{2\sigma^2}\Vert x-y \Vert^2\right)  \\
\text{Kernel Laplaciano} && \mathrm{K}(x,y) &= exp\left(-\alpha \sum\limits_{i=1}^n|x_i-y_i|\right) \\
\end{align*}
$$

#### Espacios de Hilbert con Kernel reproductor

###### Definición.

Un *Espacio de hilbert con kernel reproductor (RKHS por sus siglas en inglés)* sobre $\mathcal{X}$ es un espacio de Hilbert $\mathcal{H}$   conformado por funciones en $\mathcal{X}$, tal que el mapeo evaluacion
$$
\begin{align*}
e_x : \mathcal{H} \to \mathbb{K}, && e_x(f) = f(x) 
\end{align*}
$$
es un funcional lineal continuo para toda $x \in \mathcal{X}$.

###### Observación.

Por el teorema de representación de Riesz, $\forall x \in \mathcal{X}$, podemos escribir el funcional lineal $e_x$ de la siguiente forma
$$
\begin{align}
e_x(f) := f(x) = \langle f, K_x\rangle && \forall f \in \mathcal{H}
\end{align}
$$
En particular, $K_x$ es una función en $\mathcal{H}$ de forma que si $y\in\mathcal{X}$, por el teorema de representación obtenemos
$$
K_x(y) = \langle K_x, K_y \rangle
$$
Con base en la expresión obtenida en $(3)$ surge la siguiente definición.

###### Definición.

El *kernel reproductor* de un RKHS es la función
$$
\begin{align*}
\mathrm{K}: \mathcal{X} \times \mathcal{X} \to \mathbb{R} && \mathrm{K}(x,y) := K_x(y)
\end{align*}
$$

###### Proposicición.

Sea $\mathcal{H}$ un espacio de Hilbert sobre $\mathcal{X}$, entonces $\forall{x} \in \mathcal{X}$ existe una función $K_x \in \mathcal{H}$ tal que
$$
\begin{align}
\langle f, K_x\rangle = f(x) && \forall f \in \mathcal{H} && (\textit{Propiedad de reproducibilidad})
\end{align}
$$

###### Demostración.

> Como se mostró en $(2)$ es una consecuencia directa del teorema de representación de Riesz.

Hemos visto como un RKHS define un *kernel reproductor* - el lector deberá convencerse de que es simétrico y positivo - En consecuencia, todo RKHS define un kernel positivo definido. ¿Será cierta la afirmación conversa? El siguiente teorema resuelve la cuestión afirmativamente, todo kernel positivo definido define de manera única un RKHS.

###### Teorema (Moore–Aronszajn, 1950)

Sea $\mathrm{K}: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ un kernel positivo definido, entonces existe un único RKHS $\mathcal{H}_\mathrm{K}$ tal que:

- $\mathrm{K}(\cdot,x) \in \mathcal{H}_\mathrm{K} \quad \forall x \in \mathcal{X}$
- $\mathrm{Span}\{\mathrm{K}(\cdot,x) | x \in \mathcal{X}\}$ es denso en $\mathcal{H}_\mathrm{K}$
- $\mathrm{K}$ es un kernel reproductor de $\mathcal{H}_\mathrm{K}$

###### Demostración.

> Ver la prueba en [x] $_\square$
>
> http://members.cbio.mines-paristech.fr/~jvert/svn/kernelcourse/notes/aronszajn.pdf

###### Definición.

Sea F un Funcional $F : \mathcal{H} \to \mathbb{K}$ el *gradiente funcional* de $F$ denotado por $\nabla_f F(f)$ es una función en $\mathcal{H}$ tal que $F(f + \epsilon g(x)) = F(f) + \epsilon\langle \nabla_f F(f),g\rangle_{\mathcal{H}} + O(\epsilon^2)$ para cualquier $g \in \mathcal{H}$ y $\epsilon \in \mathbb{R}$