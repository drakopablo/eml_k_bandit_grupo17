# Práctica 1: Problema del Bandido de k-Brazos

## Información
- **Alumnos:** Rodríguez Carmon, Adrián; Guillén Marquina, Pablo; Pozo Torres, Yolanda Victoria
- **Asignatura:** Extensiones de Machine Learning
- **Curso:** 2025/20256
- **Grupo:** grupo_17

## Descripción
Este proyecto implementa y compara distintos algoritmos de solución para el problema del **bandido multi-brazo**.  
Se analizan estrategias clásicas y avanzadas, incluyendo **ε-greedy**, **UCB** y **Softmax**.

El objetivo es evaluar el rendimiento de cada algoritmo en términos de:
- **Recompensa promedio** a lo largo del tiempo.  
- **Selección del brazo óptimo** y su evolución.  
- **Estadísticas de cada brazo**, incluyendo su frecuencia de selección y recompensa media.  
- **Regret acumulado**, para medir la pérdida respecto a la estrategia óptima.  

### Modelado del Problema  
Los experimentos utilizan un **bandido multi-brazo con diferentes distribuciones de recompensa**,  
simulando entornos con diferentes niveles de incertidumbre:  

- **Bernoulli**: Cada brazo devuelve recompensa (1) con cierta probabilidad o fallo (0).  
- **Binomial**: Simula múltiples intentos con éxito/fallo en un número fijo de ensayos.  
- **Normal**: Modela entornos con recompensas continuas sujetas a ruido gaussiano.  

Cada algoritmo debe adaptarse a estas distribuciones para maximizar la recompensa acumulada.  

## Estructura
A continuación, comentamos la estructura del contenido de la carpeta:
- **`src/algorithms/`**: Contiene las implementaciones de los algoritmos de toma de decisiones, como **Epsilon-Greedy, UCB1, UCB2 y Softmax**.
- **`src/arms/`**: Contiene las implementaciones de los distintos tipos de brazos del bandido multi-brazo, como **Bernoulli, Binomial y Normal**.
- **`src/plotting/`**: Funciones para la visualización de resultados mediante gráficos.
- **`main.ipynb`**: Notebook principal que permite lanzar los distintos experimentos.
- **`estudio_*.ipynb`**: Notebooks específicos donde se analizan en detalle diferentes estrategias y algoritmos.
- **`requirements.txt`**: Lista de dependencias necesarias para la ejecución del proyecto.
- **`README.md`**: Documento principal con información general del proyecto y su estructura.

Este repositorio permite ejecutar experimentos en **Google Colab** o en **un entorno local**, garantizando reproducibilidad en los análisis y resultados.

El repositorio contiene los siguientes notebooks de análisis:  

- **`estudio_egreedy.ipynb`**: Evaluación de estrategias ε-greedy con diferentes valores de ε.  
- **`estudio_UCB.ipynb`**: Implementación y análisis de UCB clásico.  
- **`estudio_softmax.ipynb`**: Estudio del método softmax.

## Instalación y Uso  

Este repositorio permite ejecutar experimentos en **Google Colab** o en **un entorno local**, garantizando reproducibilidad en los análisis y resultados.

El notebook **main.ipynb** es el punto de inicio del proyecto. Desde él, se proporciona acceso a los notebooks de los estudios.  

Para poner en marcha la ejecución del proyecto, simplemente sigue estos pasos:  

1. **Abrir main.ipynb** utilizando el siguiente enlace para Google Colab: [Open in Colab](https://colab.research.google.com/github/drakopablo/eml_k_bandit_grupo17/blob/main/main.ipynb) o bien en el entorno local (VSCode/Jupyter) 

2. **Acceder a los notebooks de los experimentos**:  
   Al finalizar la ejecución, en la parte inferior del notebook principal, aparecerán enlaces directos a los notebooks individuales para cada método estudiado:  
     - **ε-Greedy**  
     - **Upper Confidence Bound (UCB)**  
     - **Softmax**  
       
   Basta con hacer clic en cualquier enlace para abrir y ejecutar el estudio correspondiente.

3. **Ejecutar todas las celdas** en orden automático:  
   En la barra de menú de Colab, haz clic en **Entorno de ejecución > Ejecutar todas**.  
   Esto iniciará una secuencia de pasos automática, que incluye:  
     - **Clonación del repositorio** con todo el código y los experimentos.  
     - **Ejecución del estudio** mostrando los datos y gráficas correspondientes. 

## Tecnologías Utilizadas  
- **Lenguaje:** Python 3.x  
- **Bibliotecas:** NumPy, Matplotlib, Pandas, SciPy  
- **Entorno de ejecución:** Jupyter Notebook, Google Colab  
