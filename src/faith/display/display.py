import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def visualize(y: np.ndarray,
              t: np.ndarray,
              labels: list,
              xlabel: str = None,
              ylabel: str = None,
              title: str = None,
    ) -> plt.Figure:

    fig, axs = plt.subplots()

    for k, label in enumerate(labels):
        sns.lineplot(x=t[:, k], y=y[:, k], label=label)

    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    axs.set_title(title)
    plt.show()

    return fig

def spectrogram(y: np.ndarray,
                t: np.ndarray,
                f: np.ndarray,
                labels: list,
                xlabel: str = None,
                ylabel: str = None,
                title: str = None,
    ) -> plt.Figure:

    fig, axs = plt.subplots()

    for k, label in enumerate(labels):
        sns.heatmap(y[:, :, k], xticklabels=t[:, k], yticklabels=f, ax=axs, label=label)

    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    axs.set_title(title)
    plt.show()

    return fig
