---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.0'
      jupytext_version: 0.8.5
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.7.1
---

```python
import seaborn as sns
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pymc3 as pm
```

```python
five_thirty_eight = [
    "#30a2da",
    "#fc4f30",
    "#e5ae38",
    "#6d904f",
    "#8b8b8b",
]
sns.set()
sns.set_palette(five_thirty_eight)
```

```python
def p():
    return lambda x: 0.7*stats.norm.pdf(x, loc=3, scale=1) + 0.3*stats.norm.pdf(x, loc=-3, scale=1)
```

```python
y = stats.bernoulli.rvs(0.3, size=10000)
y[y==0] = -1
x = [stats.norm.rvs(loc=3*y[i], scale=1, size=1) for i in range(10000)]
```

```python
sns.distplot(x, kde=False, norm_hist=True)
plt.title("Modelo de mezclas")
plt.savefig('./output/mezclas.png')
```

```python
def q(mu, sigma):
    return lambda x: stats.norm.pdf(x, loc=mu, scale=sigma)
```

```python
def KL(q, p, x, kind='reverse'):
    if kind == 'reverse':
        w = q(x)/p(x)
        idx = np.where(w > 0)
        return np.mean(q(x[idx])*np.log(q(x[idx])/p(x[idx])))
    w = p(x)/q(x)
    idx = np.where(w > 0)
    return np.mean(p(x[idx])*np.log(p(x[idx])/q(x[idx])))
    
```

```python
z = stats.uniform.rvs(loc=-7,scale=14,size=10000)
```

```python
sigmas = np.linspace(0.01, 10, num=1000)
f_KLs = [KL(q(0,s), p(), z, kind='forward') for s in sigmas]
```

```python
sigma_opt = sigmas[np.argmin(f_KLs)]
print(sigma_opt)
print(np.min(f_KLs))
print(KL(q(0, sigma_opt), p(), z, kind='reverse'))
```

```python
w = stats.norm.rvs(loc=0, scale=3.14, size=10000)
```

```python
sns.distplot(x, kde=False, norm_hist=True)
sns.distplot(w, kde=False, norm_hist=True)
plt.title('Modelo de mezclas con aproximador Q')
plt.savefig('./output/aprox.png')
```

```python
sigmas = np.linspace(0.01, 10, num=1000)
b_KLs = [KL(q(3,s), p(), z, kind='reverse') for s in sigmas]
```

```python
sigma_opt = sigmas[np.argmin(b_KLs)]
print(sigma_opt)
print(KL(q(0, sigma_opt), p(), z, kind='forward'))
print(np.min(b_KLs))
```

```python
w = stats.norm.rvs(loc=-3, scale=1.275, size=10000)
```

```python
sns.distplot(x, kde=False, norm_hist=True)
sns.distplot(w, kde=False, norm_hist=True)
plt.title('Modelo de mezclas con aproximador Q')
plt.savefig('./output/aprox2.png')
```

```python
print(KL(q(-3,1.275), p(), z, kind='forward'))
print(KL(q(-3,1.275), p(), z, kind='reverse'))
```

```python
np.mean(x[idx]*np.log(x[idx]/w[idx]))
```

```python
np.mean(w[idx]*np.log(w[idx]/x[idx]))
```

```python

```
