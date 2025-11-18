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

def plot_delay_analysis(
        signal: np.ndarray,
        delay_acf: int,
        delay_mi_hist: int,
        delay_mi_kde: int,
        save_path: Optional[str] = None
    ) -> None:
    """
    Plot delay estimates from all three methods.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # ACF delay visualization
    from statsmodels.tsa.stattools import acf
    acf_vals = acf(signal, nlags=200, fft=True)
    axes[0].stem(np.arange(len(acf_vals)), acf_vals, basefmt=' ')
    axes[0].axvline(x=delay_acf, color='r', linestyle='--', linewidth=2, label=f'τ = {delay_acf}')
    axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[0].set_title('ACF Method')
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('ACF')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Text info
    axes[1].axis('off')
    info_text = f"""
    Delay Estimation Results
    
    ACF Method:
    τ = {delay_acf}
    
    MI Histogram:
    τ = {delay_mi_hist}
    
    MI KDE:
    τ = {delay_mi_kde}
    
    Consensus τ = {int(np.mean([delay_acf, delay_mi_hist, delay_mi_kde]))}
    """
    axes[1].text(0.1, 0.5, info_text, fontsize=12, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Delay comparison bar chart
    methods = ['ACF', 'MI Hist', 'MI KDE']
    delays = [delay_acf, delay_mi_hist, delay_mi_kde]
    colors = ['blue', 'green', 'red']
    axes[2].bar(methods, delays, color=colors, alpha=0.7)
    axes[2].axhline(y=np.mean(delays), color='k', linestyle='--', linewidth=2, label=f'Mean = {int(np.mean(delays))}')
    axes[2].set_title('Delay Comparison')
    axes[2].set_ylabel('Delay τ')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()


def plot_embedding_2d(
        embedding: np.ndarray,
        title: str = "2D Delay Embedding",
        save_path: Optional[str] = None
    ) -> None:
    """
    Plot 2D delay embedding (first two components).
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], s=1, alpha=0.5)
    plt.title(title)
    plt.xlabel('x(t)')
    plt.ylabel(f'x(t+τ)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()


def plot_embedding_3d(
        embedding: np.ndarray,
        title: str = "3D Delay Embedding",
        save_path: Optional[str] = None
    ) -> None:
    """
    Plot 3D delay embedding (first three components).
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use color gradient for better visualization
    colors = np.arange(len(embedding))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], 
                        c=colors, cmap='viridis', s=1, alpha=0.6)
    
    ax.set_title(title)
    ax.set_xlabel('x(t)')
    ax.set_ylabel('x(t+τ)')
    ax.set_zlabel('x(t+2τ)')
    
    plt.colorbar(scatter, ax=ax, label='Time progression')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()