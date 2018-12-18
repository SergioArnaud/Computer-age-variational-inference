
## Apéndice E.


```python
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pymc3 as pm
from pymc3.math import logsumexp
from pymc3 import Normal, Metropolis, sample, MvNormal, Dirichlet, \
    DensityDist, find_MAP, NUTS, Slice
from pymc3.variational.callbacks import CheckParametersConvergence

import theano
import theano.tensor as tt
from theano.tensor.nlinalg import det
%env THEANO_FLAGS=device=cpu,floatX=float32
```

#### Ejemplo ADVI

Sea $X \in \mathcal{M}_{n \times p}$ una matriz de datos donde cada obervación $x_i$ se distribuye como una mezcla de K normales p variadas, es decir
$$
x_i \sim \sum_{j=1}^K \pi_j \mathcal{N}_p( \mu_j, \mathbb{I}) \quad \sum_j \alpha_j = 1
$$
Donde los vectores de medias $\mu_j$ y las proporciones $\pi_j$ son previamente definidos.

El primer ejemplo consiste en una primera instancia, dados $n$, $p$ y $K$, obtener una muestra de dicha distribución (generando las $\mu$ y las $\pi$ se generan aleatoriamente). 

Definir el siguiente modelo
$$
\begin{align*}
 x | \ \pi, \mu &\sim \sum_{j=1}^K \pi_j \mathcal{N}_p( \mu_j, \mathbb{I}) \\ 
\end{align*}
$$
Con priors
$$
\begin{align*}
\pi_i &\sim \mathrm{Dirichlet(100)} \\
\mu_i &\sim \mathcal{N}(\mathbf{0},100\mathbb{I}) \\
\end{align*}
$$

Y obtener una muestra de la distribución posterior dada la muestra generada.


```python
# log probability of individual samples
def logp_normal(mu, tau, value):
    k = tau.shape[0]
    delta = lambda mu: value - mu
    return (-1 / 2.) * (k * tt.log(2 * np.pi) + tt.log(1./det(tau)) +
                         (delta(mu).dot(tau) * delta(mu)).sum(axis=1))

# Log likelihood of Gaussian mixture distribution
def logp_gmix(mus, pi, tau, n_samples):
    def logp_(value):
        logps = [tt.log(pi[i]) + logp_normal(mu, tau, value) for i, mu in enumerate(mus)]
        return tt.sum(logsumexp(tt.stacklists(logps)[:, :n_samples], axis=0))
    return logp_
    
    
def create_model(K=2, p=2, n_samples=100, seed = 111, minibatch=False, minibatch_size=100):
    
    # Setting seed
    rng = np.random.RandomState(seed)
    
    # Generating random data as a normal mixture
    mu_ = rng.normal(loc=0, scale = 2, size = (K,p))
    pi_ = rng.dirichlet(np.repeat(1,K))
    choices = np.array([rng.multinomial(1, pi_) for _ in range(n_samples)]).T
    
    xs = [z[:, np.newaxis] * rng.multivariate_normal(m, np.eye(p), size=n_samples)
          for z, m in zip(choices, mu_)]
    data = np.sum(np.dstack(xs), axis=2)
    

    # Defining the model
    if minibatch:
        mb = pm.Minibatch(data, minibatch_size)
    
    with pm.Model() as model:
        mus = [MvNormal('mu_{}'.format(i),
                        mu = pm.floatX(np.zeros(p)),
                            tau = pm.floatX(.001*np.eye(p)),
                        shape=(p,)) for i in range(K)]
        
        pi = Dirichlet('pi', a=pm.floatX(100 * np.ones(K)), shape=(K,))
        
        if minibatch:
            x = DensityDist('x', logp_gmix(mus, pi, np.eye(p), n_samples), observed=mb, total_size = len(data))
        else:
            x = DensityDist('x', logp_gmix(mus, pi, np.eye(p), n_samples), observed=data)
    return data,model,mu_,pi_
```


```python
# Some functions to plot data
def plot_2d_means(data,trace):
    plt.figure(figsize=(5, 5))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.5, c='g')
    mu_0, mu_1 = trace['mu_0'], trace['mu_1']
    plt.scatter(mu_0[:, 0], mu_0[:, 1], c="r", s=3, alpha=.5)
    plt.scatter(mu_1[:, 0], mu_1[:, 1], c="b", s=3, alpha=.5)
    
def plot_elbo(inference):
    fig = plt.figure(figsize=(12,5))
    hist_ax = fig.add_subplot(111)
    hist_ax.plot(-inference.hist)
    hist_ax.set_title('Negative ELBO track');
    hist_ax.set_xlabel('Iteración')
    hist_ax.set_ylabel('ELBO')
```

Poniendo semilla, creando datos y especificando modelo. En este caso los datos son generados por una mezcla de 2 normales bivariadas 


```python
seed = 100
data,model,mu,pi= create_model()
```


```python
# Obteniendo medias de las normales
mu
```


```python
# Obteniendo proporciones de la mezcla
pi
```

###### Metrópolis Hastings

A continuación realizamos una muestra con Metrópolis Hastings


```python
with model:
    metropolis_trace = sample(1000, pm.Metropolis(), random_seed=seed)
pm.traceplot(metropolis_trace)
```

Realizamos una visualización en la cual se grafican de color verde la muestra original, de color azul aquellos generados por una distribución normal con la media $\hat{\mu}_0$ posterior obtenida tras el muestreo y de color rojo aquellos generados por una distribución normal con la media $\hat{\mu}_1$correspondiente.


```python
plot_2d_means(data,metropolis_trace)
```

###### ADVI

Ahora realizamos el cálculo con advi


```python
with model:
    inference = pm.fit(method = pm.ADVI(),obj_optimizer=pm.adagrad(learning_rate=1e-1),
                               callbacks = [CheckParametersConvergence()],
                               random_seed = seed
                              )
    advi_trace = inference.sample(500)
    
pm.traceplot(advi_trace);
```

Volvemos a graficar 


```python
plot_2d_means(data,advi_trace)
```

Finalmente graficamos el negativo de la función ELBO para corroborar la convergencia.


```python
plot_elbo(inference)
```

###### Escalabilidad

A continuación mostramos la ejecución de el algoritmo con una especificación de modelo de dimesniones mucho mayores 


```python
data,model,mu,theta = create_model(p=120,K=5,n_samples=150000,minibatch=True,minibatch_size=350)
```

Unicamente se realiza con ADVI porque con MCMC python falla. Nótese que el tiempo de ejecución es aproximadamente de minuto y medio


```python
with model:
    inference = pm.fit(method = pm.ADVI(),obj_optimizer=pm.adagrad(learning_rate=1e-1),
                               callbacks = [CheckParametersConvergence()],
                               random_seed = seed
                              )
    advi_trace = inference.sample(500)
pm.traceplot(advi_trace);
```

## Modelos de mezclas de normales

Un ejemplo sencillo para mostrar el funcionamiento de SVGD en pymc3

$ X \sim .3Y_1 + .5Y_2 + .2Y_3$

$Y1 \sim N(-.1,.2)$

$Y2 \sim N(.5,.1)$

$Yy \sim N(.1,.1)$

Especificación del modelo


```python
w = pm.floatX([.3, .5, .2 ])
mu = pm.floatX([-.1, .5, 1])
sd = pm.floatX([.2, .1, .1])
seed = 100

with pm.Model() as model:
    x = pm.NormalMixture('x', w=w, mu=mu, sd=sd, dtype=theano.config.floatX)
```

#### Metrópolis Hastings


```python
with model:
    step = pm.Metropolis()
    trace_metropolis = pm.sample(5000, step, random_seed=seed)

pm.plot_posterior(trace_metropolis, color='LightSeaGreen',figsize=(10,5))
pm.traceplot(trace_metropolis, figsize=(10,3))
```

#### SVGD


```python
with model:
    svdg = pm.SVGD(n_particles=100,random_seed=seed)
    inference_svdg = svdg.fit(callbacks=[CheckParametersConvergence()])
    trace_SVDG = inference_svdg.sample(5000)
pm.plot_posterior(trace_SVDG, color='LightSeaGreen',figsize=(10,5))
pm.traceplot(trace_SVDG, figsize=(12,4))
```

Comparación


```python
sns.set(rc={'figure.figsize':(8,5)})
ax = sns.kdeplot(trace_metropolis['x'], label='MCMC');
sns.kdeplot(trace_SVDG['x'], label='SVDG');
```


```python

```
