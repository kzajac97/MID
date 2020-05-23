import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

sns.set()


def covariance_plot(covariance_matrix: np.array) -> None:
    """
    Utility for quick matrix heatmap plot with default parameters
    :param covariance_matrix: square numpy matrix or array to display
    """
    plt.figure(figsize=[12, 12])
    _ = sns.heatmap(covariance_matrix, annot=True, fmt=".2f", cbar=False, linewidth=0.1, cmap="coolwarm")

    return
