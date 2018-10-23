# Propuesta de proyecto: Inferencia variacional por diferenciación automática

## Introducción

Durante el curso atacamos uno de los problemas más usuales de la estadística bayesiana: el muestreo de una distribución posterior $q$ a través de métodos de Monte Carlo con Cadena de Markov. Sin embargo, los métodos MCMC son computacionalmente intensivos y tomar una muestra puede ser imposible. Una alternativa es convertir el problema en uno de optimización, proponiendo una familia de densidades $\mathcal{Q}$ para y eligiendo
$$
q^* = \arg\min_{\mathcal{Q}} D_\mathrm{KL}(q || p(\cdot|\mathbf{X}))
$$
donde $D_\textrm{KL}$ es la divergencia de Kullback-Liebler. Este método de aproximación es *inferencia variacional*. 

En este proyecto exploraremos el marco teórico de la inferencia bayesiana variacional, y presenteremos un algoritmo publicado en 2016 que resuelve el problema por optimización estocástica sobre una función gradiente que construye de manera automática, siguiendo [1].  Finalmente, compararemos ambos métodos muestreando un conjunto grande de datos con un modelo complicado (tenemos tres opciones pero no hemos definido cuál) y estudiaremos el desempeño del algoritmo con los métodos planteados en [2].



## Referencias

[1] Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., & Blei, D. M. (2016). Automatic Differentiation Variational Inference, 1–38. https://doi.org/10.3847/0004-637X/819/1/50

[2] Yao, Y., Vehtari, A., Simpson, D., & Gelman, A. (2018) Yes, but did it work?: Evaluating variational inference. arXiv preprint arXiv:1802.02538.

[3] Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). Variational Inference: A Review for Statisticians. *Journal of the American Statistical Association*, *112*(518), 859–877. https://doi.org/10.1080/01621459.2017.1285773