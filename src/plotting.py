import matplotlib.pyplot as plt
import numpy as np
from typing import Sequence, Optional
from mpl_toolkits.mplot3d import Axes3D

def plot_time_series_multi(
        signals: list[np.ndarray],
        t : np.ndarray,
        labels: Optional[list[str]] = None,
        title: str = "",
        ylabel: str = "Value",
        xlabel: str = "Time",
        save_path: Optional[str] = None
    ) -> None:
    """
    Plots multiple time series on the same figure.
    """
    plt.figure(figsize=(10, 4))
    for idx, sig in enumerate(signals):
        label = labels[idx] if labels and idx < len(labels) else None
        plt.plot(t, sig, linewidth=0.8, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if labels:
        plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()
        
def plot_trajectory_3d(
        trajectory: np.ndarray,
        title: str = "",
        labels: Optional[list[str]] = None,
        save_path: Optional[str] = None
    ) -> None:
    """
    Plots a 3D trajectory.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], linewidth=0.8, alpha=0.7)
    ax.set_title(title)
    if labels and len(labels) == 3:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
    else:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()