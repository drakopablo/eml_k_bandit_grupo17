"""
Module: plotting/sutton.py
Description: Gráfica de distribución de recompensas por brazo.
"""

from typing import List
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_reward_distributions(arms: List, samples_per_arm: int = 1000):
    """
    Genera una gráfica mostrando la distribución
    de recompensas de cada brazo.

    Parameters
    ----------
    arms : List
        Lista de brazos (deben tener métodos pull()
        y get_expected_value()).
    samples_per_arm : int
        Número de muestras simuladas por brazo.
    """

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    k = len(arms)

    data = []
    for arm in arms:
        samples = [arm.pull() for _ in range(samples_per_arm)]
        data.append(samples)

    plt.figure(figsize=(14, 7))

    sns.violinplot(data=data, inner=None, color="lightgray")

    # Marcar medias verdaderas
    for i, arm in enumerate(arms):
        true_mean = arm.get_expected_value()
        plt.scatter(i, true_mean, color="black", s=60, zorder=3)

    plt.xlabel("Acción", fontsize=14)
    plt.ylabel("Distribución de recompensa", fontsize=14)
    plt.title("Distribución de recompensas por brazo", fontsize=16)

    plt.xticks(range(k), range(1, k + 1))
    plt.tight_layout()
    plt.show()
