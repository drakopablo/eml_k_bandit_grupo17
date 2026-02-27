"""
Module: plotting/sutton.py
Description: Gráfica de distribución de recompensas por brazo.
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_reward_distributions(arms, samples_per_arm=1000):
    """
    Plotea distribuciones de recompensa tipo Sutton & Barto (2018).

    Parameters
    ----------
    arms : list
        Lista de brazos (cada uno debe tener método pull() y get_expected_value()).
    samples_per_arm : int
        Número de muestras simuladas por brazo para dibujar la distribución.
    """

    k = len(arms)

    # Generar muestras
    data = []
    for arm in arms:
        samples = [arm.pull() for _ in range(samples_per_arm)]
        data.append(samples)

    # Crear figura
    plt.figure(figsize=(12, 6))

    # Violin plot
    sns.violinplot(data=data, inner=None, color="lightgray")

    # Dibujar medias verdaderas
    for i, arm in enumerate(arms):
        true_mean = arm.get_expected_value()
        plt.scatter(i, true_mean, color="black", s=50, zorder=3)
        plt.text(i, true_mean, f" q*({i+1})", fontsize=9)

    # Línea horizontal en 0
    plt.axhline(0, linestyle="--", color="gray", linewidth=1)

    plt.xlabel("Action")
    plt.ylabel("Reward distribution")
    plt.title("Distribución de recompensas por brazo")
    plt.xticks(range(k), range(1, k+1))
    plt.tight_layout()
    plt.show()

# MEJORA EN EL EJE X:
    ax = plt.gca()
    # Si k es grande, solo mostramos etiquetas cada cierto intervalo
    if k > 20:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(20))
    else:
        plt.xticks(range(k), range(1, k+1))
        
    plt.tight_layout()
    plt.show()
