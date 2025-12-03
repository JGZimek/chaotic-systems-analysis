import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from mpl_toolkits.mplot3d import Axes3D

def plot_time_series_multi(signals, t, labels=None, title="", ylabel="Value", xlabel="Time", save_path=None):
    plt.figure(figsize=(10, 4))
    for idx, sig in enumerate(signals):
        lbl = labels[idx] if labels and idx < len(labels) else None
        plt.plot(t, sig, linewidth=0.8, label=lbl)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if labels: plt.legend()
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300)
    plt.close()
        
def plot_delay_analysis(signal, delay_acf, delay_mi_hist, delay_mi_kde, save_path=None):
    from statsmodels.tsa.stattools import acf
    
    # Wyświetlanie konsoli przeniesione do main, tu tylko plot
    fig, ax = plt.subplots(figsize=(12, 6))
    lags = np.arange(1, 101) # Rysujemy pierwsze 100 dla czytelności
    
    # ACF
    try:
        acf_vals = acf(signal, nlags=100, fft=True)
        ax.plot(lags, acf_vals[1:], label=f'ACF (τ={delay_acf})', color='blue', alpha=0.7)
        ax.axvline(x=delay_acf, color='blue', linestyle='--')
    except: pass

    # Tylko zaznaczamy pionowe linie dla MI, bo obliczanie całego wykresu w plot_delay jest kosztowne
    # (funkcje MI zwracają tylko tau, nie całą tablicę wartości w tej implementacji)
    ax.axvline(x=delay_mi_hist, color='red', linestyle='--', label=f'MI Hist (τ={delay_mi_hist})')
    
    ax.set_title("Delay Estimation Points")
    ax.set_xlabel("Lag")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path: plt.savefig(save_path, dpi=300)
    plt.close()

def plot_embedding_3d(embedding, title="3D Embedding", save_path=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = np.arange(len(embedding))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], 
                        c=colors, cmap='viridis', s=1, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel('x(t)')
    ax.set_ylabel('x(t+τ)')
    ax.set_zlabel('x(t+2τ)')
    plt.colorbar(scatter, ax=ax, label='Time')
    if save_path: plt.savefig(save_path, dpi=300)
    plt.close()