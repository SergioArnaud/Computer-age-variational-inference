## Evaluación

La inferencia variacional reduce el tiempo de cómputo para muestrear posteriores complicadas, pero da sólo un aproximado. ¿Cómo podemos verificar qué tan bueno es el aproximado? Primero, es fácil diagnosticar la convergencia del algoritmo de optimización monitoreando el cambio en $\mathrm{ELBO}$, y esto siempre debería hacerse. Sin embargo, puede ser que incluso habiendo convergido el algoritmo a un máximo global de $\mathrm{ELBO}$, la familia  $\mathscr{Q}$ haya sido elegido tan desafortunadamente que aún el óptimo es malo o que por la poca penalización que $D_{KL}(q\ ||\ p)​$ le pone a las colas ligeras tenagmos una densidad muy distinta a la verdadera.g

Yao *et al.* (2018) proponen dos diagnósticos cuantitativos, uno para la calidad de la posterior variacional y otro para el sesgo de un estimador puntual bajo tal posterior. 

