import numpy as np
from algorithms.algorithm import Algorithm
import math

class UCB1(Algorithm):
    """
    Implementación del algoritmo UCB1 para el problema del bandido multibrazo.
    
    UCB1 es un método basado en límites superiores de confianza que balancea la exploración y explotación.
    En cada iteración, elige el brazo con el mayor valor UCB1, el cual combina la recompensa esperada 
    con un término de incertidumbre basado en la desigualdad de Hoeffding.

    **Referencias:**
    - Peter Auer et al., 2002: "Finite-time Analysis of the Multiarmed Bandit Problem"
    """

    def __init__(self, k: int, c: float = 1.0):
        """
        Inicializa el algoritmo UCB1.

        :param k: Número de brazos disponibles en el bandido.
        :param c: Parámetro de ajuste para la exploración (default = 1.0).
        """
        super().__init__(k)
        self.c = c  # Factor de ajuste en el término de exploración

    def select_arm(self) -> int:
        """
        Selecciona un brazo siguiendo la política UCB1.

        **Funcionamiento:**
        - Si hay brazos no explorados, se elige uno de ellos primero.
        - Para cada brazo, se calcula el índice UCB1 usando la ecuación:
          \[
          UCB1(a) = Q(a) + c \times \sqrt{\frac{2 \ln t}{N(a)}}
          \]
        - Se selecciona el brazo con el mayor valor UCB1.

        :return: Índice del brazo seleccionado.
        """

        # Seleccionar primero cualquier brazo que no haya sido explorado aún
        for arm in range(self.k):
            if self.counts[arm] == 0:
                return arm

        # Número total de selecciones realizadas hasta ahora
        total_selections = sum(self.counts)

        # Calcular el valor UCB1 para cada brazo
        ucb_values = np.zeros(self.k)

        for arm in range(self.k):
            if self.counts[arm] > 0:
                # Término de confianza (exploración), basado en la desigualdad de Hoeffding
                confidence_bound = self.c * math.sqrt((2 * math.log(total_selections)) / self.counts[arm])
                # Cálculo del índice UCB1 para este brazo
                ucb_values[arm] = self.values[arm] + confidence_bound

        # Seleccionar el brazo con el mayor valor UCB1
        chosen_arm = np.argmax(ucb_values)
        return chosen_arm

    def update(self, chosen_arm: int, reward: float):
        """
        Actualiza la estimación de recompensa del brazo seleccionado.

        **Funcionamiento:**
        - Se incrementa el contador de selecciones del brazo.
        - Se actualiza la estimación de la recompensa promedio con la nueva observación.

        :param chosen_arm: Brazo seleccionado.
        :param reward: Recompensa obtenida al seleccionar el brazo.
        """

        # Incrementar la cantidad de veces que el brazo ha sido seleccionado
        self.counts[chosen_arm] += 1

        # Actualizar el valor estimado de la recompensa del brazo seleccionado
        self.values[chosen_arm] += (reward - self.values[chosen_arm]) / self.counts[chosen_arm]

    def reset(self):
        """
        Reinicia los valores del algoritmo para ejecutar nuevos experimentos.
        """
        super().reset()
