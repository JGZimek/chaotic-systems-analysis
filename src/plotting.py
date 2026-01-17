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

def plot_original_system_3d(trajectory, title="Original Chaotic System", save_path=None, 
                             labels=None):
    """
    Rysuje oryginalny układ chaotyczny w przestrzeni fazowej 3D (x, y, z).
    Styl dopasowany do plot_embedding_3d dla spójności wizualnej.
    
    Args:
        trajectory: Tablica numpy o kształcie (N, 3) z trajektorią [x, y, z]
        title: Tytuł wykresu
        save_path: Ścieżka do zapisu pliku PNG
        labels: Lista etykiet osi [xlabel, ylabel, zlabel] lub None dla domyślnych
    """
    if labels is None:
        labels = ['x', 'y', 'z']
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = np.arange(len(trajectory))
    scatter = ax.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                        c=colors, cmap='viridis', s=2, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    plt.colorbar(scatter, ax=ax, label='Time')
    if save_path: plt.savefig(save_path, dpi=300)
    plt.close()

def plot_embedding_saturation(data, tau, title="Embedding Dimension Saturation", 
                              save_path=None, max_dim=10):
    """
    Rysuje nasycenie wymiaru zanurzenia poprzez wykresy log(C(r)) vs log(r).
    
    Pokazuje jak wymiary korelacyjne D2 dla różnych dE stają się niezależne od dE
    (charakterystyczne dla systemów chaotycznych).
    
    Analogicznie do Fig. 9 z artykułu:
    "Compound method of time series classification"
    
    Args:
        data: Sygnał (szereg czasowy)
        tau: Opóźnienie czasowe
        title: Tytuł wykresu
        save_path: Ścieżka do zapisu
        max_dim: Maksymalny wymiar zanurzenia do sprawdzenia (domyślnie 10)
    """
    from scipy.spatial import cKDTree
    
    # Normalizacja sygnału
    data_norm = (data - np.mean(data)) / np.std(data)
    
    # Definiuj zakresy promieni (w skali log)
    std_data = np.std(data_norm)
    r_vals = np.logspace(np.log10(std_data * 0.05), np.log10(std_data * 2.0), 30)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Kolory dla różnych wymiarów
    colors_palette = plt.cm.viridis(np.linspace(0, 1, max_dim))
    
    # Dla każdego wymiaru zanurzenia m, oblicz i narysuj C(r)
    for m in range(1, max_dim + 1):
        N = len(data_norm)
        M = N - (m - 1) * tau
        
        if M < 10:
            continue
            
        # Rekonstrukcja atraktora
        orbit = np.array([data_norm[i : i + (m * tau) : tau] for i in range(M)])
        
        # Subsampling dla szybkości
        if len(orbit) > 1000:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(orbit), 1000, replace=False)
            orbit = orbit[idx]
            M_actual = 1000
        else:
            M_actual = len(orbit)
        
        # Obliczenie całki korelacyjnej
        tree = cKDTree(orbit)
        C_r = []
        for r in r_vals:
            count = tree.count_neighbors(tree, r) - M_actual
            norm_count = count / (M_actual * (M_actual - 1)) if M_actual > 1 else 1e-10
            C_r.append(max(norm_count, 1e-10))  # Unikaj log(0)
        
        C_r = np.array(C_r)
        
        # Konwersja do skali log
        log_r = np.log(r_vals)
        log_Cr = np.log(C_r)
        
        # Rysowanie - grubsza linia dla wyższych wymiarów
        linewidth = 1.5 + 0.3 * (m - 1)
        ax.plot(log_r, log_Cr, linewidth=linewidth, color=colors_palette[m-1], 
               label=f'dE = {m}', alpha=0.8)
    
    ax.set_xlabel('ln(ε)', fontsize=12)
    ax.set_ylabel('ln(C(ε))', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()