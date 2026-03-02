"""
Module: plotting/plotting.py
Description: Contiene funciones para generar gráficas de comparación de algoritmos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from typing import List

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from algorithms import Algorithm


def get_algorithm_label(algo: Algorithm) -> str:
    """
    Genera una etiqueta descriptiva para el algoritmo incluyendo sus parámetros.

    :param algo: Instancia de un algoritmo.
    :type algo: Algorithm
    :return: Cadena descriptiva para el algoritmo.
    :rtype: str
    """
    if not isinstance(algo, Algorithm):
        raise ValueError("El algoritmo debe ser de la clase Algorithm o una subclase.")

    label = type(algo).__name__

    # Añadimos parámetros comunes cuando existen en el algoritmo.
    params = []
    if hasattr(algo, "epsilon"):
        params.append(f"epsilon={algo.epsilon}")
    if hasattr(algo, "temperature"):
        params.append(f"temperature={algo.temperature}")

    if params:
        label += f" ({', '.join(params)})"

    return label


def plot_average_rewards(steps: int, rewards: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Recompensa Promedio vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param rewards: Matriz de recompensas promedio.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), rewards[idx], label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Recompensa Promedio', fontsize=14)
    plt.title('Recompensa Promedio vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()


def plot_optimal_selections(steps: int, optimal_selections: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Porcentaje de Selección del Brazo Óptimo vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param optimal_selections: Matriz de porcentaje de selecciones óptimas.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), optimal_selections[idx], label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('% Selección brazo óptimo', fontsize=14)
    plt.title('Porcentaje de selección del brazo óptimo vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()

def plot_arm_statistics(arm_stats: List[dict], algorithms: List[Algorithm], optimal_arm: int):
    """
    Genera histogramas de las recompensas promedio por brazo y el número de veces seleccionado.
    Muestra el desempeño de cada brazo y cuántas veces ha sido elegido por cada algoritmo.
    
    :param arm_stats: Lista de diccionarios con estadísticas de cada brazo para cada algoritmo.
    :type arm_stats: List[dict]
    :param algorithms: Lista de instancias de algoritmos comparados.
    :type algorithms: List[Algorithm]
    :param optimal_arm: Índice del brazo óptimo basado en la recompensa esperada real.
    :type optimal_arm: int
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
    plt.figure(figsize=(14, 7))

    bar_width = 0.3  # Ancho de cada barra en el histograma
    x_positions = np.arange(len(arm_stats[0]['means']))  # Posiciones X de los brazos

    for idx, algo in enumerate(algorithms):
        stats = arm_stats[idx]
        x_offset = (idx - len(algorithms) / 2) * bar_width  # Desplazamiento para evitar solapamientos
        bars = plt.bar(x_positions + x_offset, stats['means'], width=bar_width, alpha=0.7, label=get_algorithm_label(algo))

        for i, (mean, count) in enumerate(zip(stats['means'], stats['counts'])):
            plt.text(i + x_offset, mean + 0.1, f"{count}", ha='center', fontsize=12)

    # Destacar el brazo óptimo con una estrella dorada (★)
    plt.scatter(optimal_arm, max(arm_stats[0]['means']), color='gold', s=200, marker='*', edgecolors='black', label="Brazo Óptimo")

    plt.xlabel('Índice del Brazo', fontsize=14)
    plt.ylabel('Recompensa Promedio', fontsize=14)
    plt.title('Estadísticas de Selección de los Brazos', fontsize=16)
    plt.xticks(x_positions, [f"{i}" for i in x_positions])  # Asegura que los brazos tengan etiquetas en X
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()

def plot_regret(steps: int, regret_accumulated: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Regret Acumulado vs Pasos de Tiempo.
    El regret mide la diferencia entre la recompensa obtenida y la mejor posible, acumulándose con el tiempo.
    
    :param steps: Número de pasos de tiempo.
    :type steps: int
    :param regret_accumulated: Matriz con la evolución del regret acumulado para cada algoritmo en cada paso.
    :type regret_accumulated: np.ndarray
    :param algorithms: Lista de instancias de algoritmos comparados.
    :type algorithms: List[Algorithm]
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), regret_accumulated[idx], label=label, linewidth=2)
    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Regret Acumulado', fontsize=14)
    plt.title('Evolución del Regret Acumulado', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()
