"""
Module: plotting/sutton.py
Description: Gráfica de distribución de recompensas por brazo.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker # Necesario para el eje X dinámico

def plot_reward_distributions(arms, samples_per_arm=1000):
    """
    Versión optimizada para manejar desde k=10 hasta k=100.
    """
    k = len(arms)

    # Generar muestras
    data = []
    for arm in arms:
        samples = [arm.pull() for _ in range(samples_per_arm)]
        data.append(samples)

    # Crear figura más ancha para k grande
    plt.figure(figsize=(14, 6))

    # Violin plot
    sns.violinplot(data=data, inner=None, color="lightgray")

    # Dibujar medias verdaderas
    for i, arm in enumerate(arms):
        true_mean = arm.get_expected_value()
        # Punto más pequeño si k es grande para no saturar
        point_size = 20 if k > 20 else 50
        plt.scatter(i, true_mean, color="black", s=point_size, zorder=3)
        
        # Solo poner el texto si k es pequeño
        if k <= 20:
            plt.text(i, true_mean, f" q*({i+1})", fontsize=8)

    # Línea horizontal en 0
    plt.axhline(0, linestyle="--", color="gray", linewidth=1)

    plt.xlabel("Acción (Brazo)")
    plt.ylabel("Distribución de Recompensa")
    plt.title(f"Distribución de recompensas por brazo (k={k})")

    # LOGICA DINÁMICA DEL EJE X
    ax = plt.gca()
    if k > 20:
        # Poner marcas cada 5 unidades para que sea legible
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    else:
        # Comportamiento normal para pocos brazos
        plt.xticks(range(k), range(1, k+1))

    plt.tight_layout()
    plt.show()
