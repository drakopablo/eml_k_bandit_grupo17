import numpy as np
from algorithms.algorithm import Algorithm
import math

class UCB2(Algorithm):
    """
    Implementación del algoritmo UCB2 para el problema del bandido multibrazo.
    
    UCB2 es una variante de UCB1 que introduce un mecanismo basado en 'épocas' para equilibrar exploración y explotación.
    En lugar de actualizar la estimación de cada brazo en cada paso, este algoritmo selecciona los brazos en bloques de tamaño creciente.
    
    **Referencias:**
    - Peter Auer et al., 2002: "Finite-time Analysis of the Multiarmed Bandit Problem"
    """

    def __init__(self, k: int, alpha: float = 0.1):
        """
        Inicializa el algoritmo UCB2.

        :param k: Número de brazos disponibles en el bandido.
        :param alpha: Parámetro que ajusta la tasa de exploración en función del crecimiento de las épocas.

        **Explicación del Parámetro `alpha`:**
        - Controla el equilibrio entre exploración y explotación.
        - Un `alpha` bajo implica que los bloques de exploración crecen más lentamente, lo que favorece la explotación.
        - Un `alpha` alto favorece la exploración al aumentar más rápido la duración de los bloques.
        """

        super().__init__(k)
        assert 0 < alpha < 1, "El parámetro alpha debe estar en el rango (0,1)."

        self.alpha = alpha  # Parámetro de exploración
        self.epoch_counts = np.zeros(k, dtype=int)  # Cantidad de veces que cada brazo ha sido seleccionado en una época
        self.epoch_lengths = np.ones(k, dtype=int)  # Longitud del bloque de tiempo de cada brazo

    def select_arm(self) -> int:
        """
        Selecciona un brazo siguiendo la política UCB2.

        **Funcionamiento:**
        - Se calcula el índice de confianza UCB2 para cada brazo con la fórmula:
          \[
          UCB2(a) = Q(a) + \sqrt{\frac{(1+\alpha) \ln (e t / \tau(k_a))}{2 \tau(k_a)}}
          \]
        - Se elige el brazo con el mayor valor de UCB2.

        :return: Índice del brazo seleccionado.
        """

        # Si hay brazos no explorados, seleccionarlos primero
        for arm in range(self.k):
            if self.counts[arm] == 0:
                return arm

        # Calcular el valor UCB2 para cada brazo
        total_selections = sum(self.counts)  # t: Número total de selecciones hasta ahora
        ucb_values = np.zeros(self.k)

        for arm in range(self.k):
            if self.counts[arm] > 0:
                tau_k = self.epoch_lengths[arm]  # Longitud del bloque para este brazo
                confidence_bound = math.sqrt((1 + self.alpha) * math.log(math.e * total_selections / tau_k) / (2 * tau_k))
                ucb_values[arm] = self.values[arm] + confidence_bound

        # Seleccionar el brazo con el índice UCB2 más alto
        chosen_arm = np.argmax(ucb_values)
        return chosen_arm

    def update(self, chosen_arm: int, reward: float):
        """
        Actualiza la estimación de recompensa del brazo seleccionado.

        **Funcionamiento:**
        - Se actualiza la media de la recompensa del brazo seleccionado.
        - Se incrementa el contador de selecciones para ese brazo.
        - Se verifica si se ha alcanzado el final de la época y, si es así, se amplía la duración del siguiente bloque.

        :param chosen_arm: Brazo que fue seleccionado.
        :param reward: Recompensa obtenida tras seleccionar el brazo.
        """

        # Actualización del promedio de recompensa del brazo seleccionado
        self.counts[chosen_arm] += 1  # Incrementar el número de selecciones de este brazo
        self.values[chosen_arm] += (reward - self.values[chosen_arm]) / self.counts[chosen_arm]

        # Incrementar el contador dentro de la época
        self.epoch_counts[chosen_arm] += 1

        # Si se ha completado la época para este brazo, actualizar la longitud del siguiente bloque
        if self.epoch_counts[chosen_arm] >= self.epoch_lengths[chosen_arm]:
            self.epoch_counts[chosen_arm] = 0  # Reiniciar el contador de la época
            self.epoch_lengths[chosen_arm] = math.ceil((1 + self.alpha) * self.epoch_lengths[chosen_arm])  # Ampliar el bloque de exploración

    def reset(self):
        """
        Reinicia los valores de recompensa estimada y estructuras auxiliares para ejecutar nuevos experimentos.
        """
        super().reset()
        self.epoch_counts = np.zeros(self.k, dtype=int)  # Reiniciar contadores de épocas
        self.epoch_lengths = np.ones(self.k, dtype=int)  # Reiniciar longitud de bloques
