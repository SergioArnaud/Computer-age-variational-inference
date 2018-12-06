

```python
import pandas as pd
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pymc3 as pm
from pymc3.variational.callbacks import CheckParametersConvergence
from pymc3.math import logsumexp
from pymc3 import Normal, Metropolis, sample, MvNormal, Dirichlet, \
                  DensityDist, find_MAP, NUTS, Slice
import theano
import theano.tensor as tt
from theano.tensor.nlinalg import det
%env THEANO_FLAGS=device=cpu,floatX=float32

import warnings
warnings.filterwarnings('ignore')
```

## Modelos de mezclas de normales

Un ejemplo toy pero mejor checa el siguiente ejemplo a ver que opinas porque está muy raro (y al final es una generalización de este)

$ X \sim .2Y_1 + .6Y_2 + .2Y_3$

$Y1 \sim N(-.1,.2)$

$Y2 \sim N(.5,.1)$

$Yy \sim N(.1,.1)$

Especificación del modelo


```python
w = pm.floatX([.2, .6, .2 ])
mu = pm.floatX([-.1, .5, 1])
sd = pm.floatX([.2, .1, .1])

with pm.Model() as model:
    x = pm.NormalMixture('x', w=w, mu=mu, sd=sd, dtype=theano.config.floatX)
```


```python
def plot_elbo(inference):
    '''
    Jalo documentar las funciones ya para la entrega D: <3 
    '''
    fig = plt.figure(figsize=(12,5))
    hist_ax = fig.add_subplot(111)

    hist_ax.plot(-inference.hist)
    hist_ax.set_title('Negative ELBO track');
    hist_ax.set_xlabel('Iteración')
    hist_ax.set_ylabel('ELBO')
```

#### Metrópolis Hastings


```python
with model:
    step = pm.Metropolis()
    trace_metropolis = pm.sample(5000, step)

pm.plot_posterior(trace_metropolis, color='LightSeaGreen',figsize=(10,5))
pm.traceplot(trace_metropolis, figsize=(10,3))
```

#### ADVI


```python
with model:
    advi = pm.ADVI()
    inference_advi = advi.fit(callbacks=[CheckParametersConvergence()])
    trace_advi = inference_advi.sample(5000)
    
pm.plot_posterior(trace_advi, color='LightSeaGreen',figsize=(12,5))
```


```python
plot_elbo(inference_advi)
```


```python
pm.traceplot(trace_ADVI);
```

#### Full rank ADVI


```python
with model:
    full_advi = pm.FullRankADVI()
    inference_full_advi = fradvi.fit(callbacks=[CheckParametersConvergence()])
    trace_full_advi = inference_fradvi.sample(50000)
    
pm.plot_posterior(trace_full_advi, color='LightSeaGreen',figsize=(12,5))
```


```python
plot_elbo(inference_full_advi)
```

#### SVGD


```python
with model:
    svdg = pm.SVGD(n_particles=200)
    inference_svdg = svdg.fit(callbacks=[CheckParametersConvergence()])
    trace_SVDG = inference_svdg.sample(5000)
pm.plot_posterior(trace_SVDG, color='LightSeaGreen',figsize=(12,5))
```

Comparación


```python
ax = sns.kdeplot(trace_MCMC['x'], label='NUTS');
sns.kdeplot(trace_ADVI['x'], label='ADVI');
sns.kdeplot(trace_FRADVI['x'], label='Full Rank ADVI');
sns.kdeplot(trace_SVDG['x'], label='SVDG', color = 'black');
```

## Ejemplo 2

Supongamos que tenemos $\alpha \sim Dir(1)$ y una matriz de observaciones $Y$ donde cada $y_{i:}$es una observación de una distribución Normal D-variada con vector de medias $\begin{pmatrix} \mu_{i0} \\ \vdots \\ \mu_{iD} \end{pmatrix}$ con probabilidad $\alpha_i$ y varianzas la identidad


```python
def create_model(K=2, D=2, n_samples=100, seed = 1249, minibatch=False, minibatch_size=100):
    
    # Setting seed
    rng = np.random.RandomState(seed)
    
    # Generating random data
    mu_ = pm.Normal.dist(mu=0,sd=2,shape=(K,D)).random()
    
    theta_ = pm.Dirichlet.dist(np.repeat(1,K)).random()
    choices = np.array([rng.multinomial(1, theta_) for _ in range(n_samples)]).T
    
    xs = [z[:, np.newaxis] * rng.multivariate_normal(m, np.eye(D), size=n_samples)
          for z, m in zip(choices, mu_)]
    data = np.sum(np.dstack(xs), axis=2)
    
    
    # THE MODEL
    
    # log probability of individual samples
    def logp_normal(mu, tau, value):
        k = tau.shape[0]
        delta = lambda mu: value - mu
        return (-1 / 2.) * (k * tt.log(2 * np.pi) + tt.log(1./det(tau)) +
                             (delta(mu).dot(tau) * delta(mu)).sum(axis=1))

    # Log likelihood of Gaussian mixture distribution
    def logp_gmix(mus, pi, tau):
        def logp_(value):
            logps = [tt.log(pi[i]) + logp_normal(mu, tau, value) for i, mu in enumerate(mus)]
            return tt.sum(logsumexp(tt.stacklists(logps)[:, :n_samples], axis=0))
        return logp_

    if minibatch:
        mb = pm.Minibatch(data, minibatch_size)
    
    with pm.Model() as model:
        mus = [MvNormal('mu_{}'.format(i),
                        mu=pm.floatX(np.zeros(D)),
                        tau=pm.floatX(0.1 * np.eye(D)),
                        shape=(D,)) for i in range(K)]
        
        pi = Dirichlet('pi', a=pm.floatX(0.1 * np.ones(K)), shape=(K,))
        
        if minibatch:
            x = DensityDist('x', logp_gmix(mus, pi, np.eye(D)), observed=mb, total_size = len(data))
        else:
            x = DensityDist('x', logp_gmix(mus, pi, np.eye(D)), observed=data)
        
    return data,model
```


```python
def plot_2d_means(data,trace):
    plt.figure(figsize=(5, 5))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.5, c='g')
    mu_0, mu_1 = trace['mu_0'], trace['mu_1']
    plt.scatter(mu_0[:, 0], mu_0[:, 1], c="r", s=10)
    plt.scatter(mu_1[:, 0], mu_1[:, 1], c="b", s=10)
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    
def run_method(algorithm, kind, optimizer=None):
    if kind == 'Variational':
        if optimizer:
            inference = algorithm.fit(obj_optimizer=optimizer,
                                      callbacks=[CheckParametersConvergence()])
        else:
            inference = algorithm.fit(callbacks=[CheckParametersConvergence()])
        trace = inference.sample(500)
        return trace,inference
    else:
        trace = sample(1000, algorithm)
        return trace
```


```python
data,model = create_model()
with model:
    metropolis_trace = run_method(pm.Metropolis(),'MCMC')
    advi_trace, advi = run_method(pm.ADVI(),'Variational')
    full_advi_trace, full_advi = run_method(pm.FullRankADVI(),'Variational')
    svdg_trace, svdg = run_method(pm.SVGD(),'Variational')
```

#### Metrópolis


```python
pm.plot_posterior(metropolis_trace)
```


```python
plot_2d_means(data,metropolis_trace)
```


```python
pm.traceplot(metropolis_trace);
```

#### ADVI


```python
pm.plot_posterior(advi_trace)
```


```python
plot_2d_means(data,advi_trace)
```


```python
plot_elbo(advi)
```

#### Full-Rank ADVI


```python
pm.plot_posterior(full_advi_trace)
```


```python
plot_2d_means(data,full_advi_trace)
```


```python
plot_elbo(full_advi)
```

#### SVDG


```python
pm.plot_posterior(svdg_trace)
```


```python
plot_2d_means(data,svdg_trace)
```

Complicando el modelo


```python
data,model = create_model(D=10,K=5,n_samples=1000)
```

Algoritmos variacionales


```python
with model:
    advi_trace, advi = run_method(pm.ADVI(),'Variational')
    full_advi_trace, full_advi = run_method(pm.FullRankADVI(),'Variational')
    
    # lento
    #svdg_trace, svdg = run_method(pm.SVGD(),'Variational')
```


```python
plot_elbo(advi)
```


```python
plot_elbo(full_advi)
```


```python
plot_elbo(svdg)
```

MCMC


```python
# A mi me truena python JAJA
with model:
    nuts_trace, advi = run_method(pm.NUTS(),'mcmc')
```

Complicando (mucho más) para presumir minibatch


```python
data,model = create_model(D=100,K=20,n_samples=100000,minibatch=True,minibatch_size=350)
```


```python
with model:
    advi_trace, advi = run_method(pm.ADVI(),'Variational',optimizer=pm.adagrad(learning_rate=1e-1))
    #full_advi_trace, full_advi = run_method(pm.FullRankADVI(),'Variational')
    #svdg_trace, svdg = run_method(pm.SVGD(),'Variational')
```


```python
plot_elbo(advi)
```


```python
plot_elbo(full_advi)
```


```python
plot_elbo(svdg)
```
