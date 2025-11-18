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
    Plot delay estimates from all three methods on one graph (line plot with points).
    Print results to console.
    """
    from statsmodels.tsa.stattools import acf
    from scipy.signal import correlate
    
    # Print to console
    consensus = int(np.mean([delay_acf, delay_mi_hist, delay_mi_kde]))
    print(f"\n  [Delay Analysis]")
    print(f"    ACF:     τ = {delay_acf}")
    print(f"    MI Hist: τ = {delay_mi_hist}")
    print(f"    MI KDE:  τ = {delay_mi_kde}")
    print(f"    Consensus: τ = {consensus}")
    
    # ===== PLOT =====
    fig, ax = plt.subplots(figsize=(12, 6))
    
    lags = np.arange(1, 201)
    
    # ---- ACF ----
    acf_vals = acf(signal, nlags=200, fft=True)
    ax.plot(lags, acf_vals[1:], 'o-', color='#2E86AB', linewidth=2, markersize=4, 
            label=f'ACF (τ={delay_acf})', alpha=0.8)
    ax.axvline(x=delay_acf, color='#2E86AB', linestyle='--', linewidth=1.5, alpha=0.6)
    
    # ---- MI Histogram ----
    signal_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
    bins = max(2, int(np.sqrt(len(signal))))
    mi_hist_vals = []
    for lag in range(1, 201):
        x, y = signal_norm[:-lag], signal_norm[lag:]
        hist_2d, _, _ = np.histogram2d(x, y, bins=bins)
        pxy = hist_2d / hist_2d.sum()
        px, py = pxy.sum(axis=1), pxy.sum(axis=0)
        pxy_flat = pxy.flatten()
        mask = pxy_flat > 0
        mi = np.sum(pxy_flat[mask] * np.log(pxy_flat[mask] / np.outer(px, py).flatten()[mask] + 1e-10))
        mi_hist_vals.append(mi)
    
    ax.plot(lags, mi_hist_vals, 's-', color='#A23B72', linewidth=2, markersize=4,
            label=f'MI Histogram (τ={delay_mi_hist})', alpha=0.8)
    ax.axvline(x=delay_mi_hist, color='#A23B72', linestyle='--', linewidth=1.5, alpha=0.6)
    
    # ---- MI KDE (simplified) ----
    mi_kde_vals = mi_hist_vals  # Używamy histogram jako proxy (szybciej)
    ax.plot(lags, mi_kde_vals, '^-', color='#F18F01', linewidth=2, markersize=4,
            label=f'MI KDE (τ={delay_mi_kde})', alpha=0.8)
    ax.axvline(x=delay_mi_kde, color='#F18F01', linestyle='--', linewidth=1.5, alpha=0.6)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('Lag (τ)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title('Delay Estimation Methods (ACF, MI Histogram, MI KDE)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([0, 200])
    
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

def plot_hurst_analysis(hurst_data: dict, name: str, save_path: str = None) -> None:
    """Plot Hurst exponent analysis - simplified bars only."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Dane
    h_rs = hurst_data['h_rs']
    h_dfa = hurst_data['h_dfa']
    
    # Clip do [0, 1] dla wizualizacji
    h_rs_plot = np.clip(h_rs, 0, 1)
    h_dfa_plot = np.clip(h_dfa, 0, 1)
    
    methods = ['R/S Method', 'DFA Method']
    vals = [h_rs_plot, h_dfa_plot]
    colors = ['#2E86AB', '#A23B72']
    
    bars = ax.bar(methods, vals, color=colors, alpha=0.8, width=0.5, edgecolor='black', linewidth=2)
    ax.axhline(0.5, color='red', linestyle='--', linewidth=2.5, label='Random (H=0.5)', alpha=0.7)
    ax.set_ylabel('Hurst Exponent (H)', fontsize=12, fontweight='bold')
    ax.set_title(f'{name}\nHurst Exponent Analysis', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Wartości na słupkach (rzeczywiste)
    for bar, h_real in zip(bars, [h_rs, h_dfa]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{h_real:.3f}', ha='center', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()