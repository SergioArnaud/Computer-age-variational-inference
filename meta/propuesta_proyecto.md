# Propuesta de proyecto: Computer age variational inference

## Introducción

Durante el curso atacamos uno de los problemas más usuales de la estadística bayesiana: el muestreo de una distribución posterior $q$ a través de métodos de Monte Carlo con Cadena de Markov. Sin embargo, los métodos MCMC son computacionalmente intensivos y tomar una muestra puede ser imposible. Una alternativa es convertir el problema en uno de optimización, proponiendo una familia de densidades $\mathcal{Q}$ para y eligiendo
$$
q^* = \arg\min_{\mathcal{Q}} D_\mathrm{KL}(q || p(\cdot|\mathbf{X}))
$$
donde $D_\textrm{KL}$ es la divergencia de Kullback-Liebler. Este método de aproximación es *inferencia variacional*. 

En este proyecto exploraremos el marco teórico de la inferencia bayesiana variacional, y presenteremos dos algoritmos estado del arte publicados en 2016 que resuelven el problema por optimización estocástica. El primero, de Kuckelbir *et. al* [1] reparametriza de manera automática para optimizar de una familia $\mathcal{Q}$ fija de antemano. El segundo, de Liu y Wang [4] construye una especie de descenso en gradiente en un reproducing kernel hilbert space (RKHS) apropiado de manera que se minimice la divergencia de Kullback-Leibler. Finalmente, presentamos una manera de evaluar la distribución variacional [2].



## Estructura

1. **Introducción.** El problema general de inferencia bayesiana. Problemas de escalabilidad de los métodos de Markov Chain Monte Carlo. (t.e. 2 min.)
2. **Inferencia variacional.** Presentar el planteamiento variacional, idea intuitiva de la divergencia de Kullback-Leibler. (t.e. 3 min.)
3. **ADVI** (t.e. 5 min.)
4. **SVGD** (t.e. 5 min.)
5. **Ejemplo.** El modelo de Terra o unos datos simulados. Mostrar gráficas con la posterior verdadera y ambas aproximaciones. Comparar la  $\hat{k}$  de [2] en ambos modelos sin explicar qué significa (sólo si fueron buenas o no). (t.e. 3 min.)
6. **Conclusión.** Señalar problemas más complicados que la inferencia bayesiana (aprendizaje por refuerzo, modelos gráficos probabilísticos) . (t.e. 2 min.)

## Referencias

[1] Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., & Blei, D. M. (2016). Automatic Differentiation Variational Inference, 1–38. https://doi.org/10.3847/0004-637X/819/1/50

[2] Yao, Y., Vehtari, A., Simpson, D., & Gelman, A. (2018) Yes, but did it work?: Evaluating variational inference. *arXiv preprint arXiv:1802.02538.*

[3] Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). Variational Inference: A Review for Statisticians. *Journal of the American Statistical Association*, *112*(518), 859–877. https://doi.org/10.1080/01621459.2017.1285773

[4] Liu, Q., Wang, D. (2016) Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm. *arXiv preprint arXiv:1608.04471v2*

[5] Liu, Q., Lee, J., Jordan, M. (2016) A Kernelized Stein Discrepancy for Goodness-of-fit Tests. *arXiv preprint arXiv:1602.03253v2* 