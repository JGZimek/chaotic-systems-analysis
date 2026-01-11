import numpy as np
from statsmodels.tsa.stattools import acf
from scipy.stats import gaussian_kde
from src.chaos_metrics import correlation_dimension_and_entropy

def estimate_delay_acf(signal: np.ndarray, max_lag: int = 100) -> int:
    try:
        acf_vals = acf(signal, nlags=max_lag, fft=True)
        zero_crossings = np.where(np.diff(np.sign(acf_vals)))[0]
        if len(zero_crossings) > 0: return int(zero_crossings[0])
    except: pass
    return max(1, max_lag // 4) 

def estimate_delay_mi_histogram(signal: np.ndarray, max_lag: int = 100) -> int:
    """
    Fraser-Swinney: Pierwsze minimum MI.
    POPRAWKA: Zmniejszono liczbę binów dla stabilności.
    """
    signal = np.asarray(signal).flatten()
    
    # POPRAWKA: Zbyt duża liczba binów powoduje szum w estymacji MI i fałszywe minima.
    # Dla N=10000, bins=44 daje ~5 pkt/bin 2D (duża wariancja).
    # bins=16 daje ~40 pkt/bin 2D (lepsza statystyka).
    bins = 16
    
    mi_vals = []
    
    for lag in range(1, max_lag + 1):
        x = signal[:-lag]
        y = signal[lag:]
        
        # Obliczenie histogramu 2D
        hist_2d, _, _ = np.histogram2d(x, y, bins=bins)
        
        # Prawdopodobieństwa
        total = np.sum(hist_2d)
        pxy = hist_2d / total
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        
        # Wzajemna informacja I(X;Y) = sum p(x,y) * log(p(x,y) / (p(x)p(y)))
        px_py = px[:, None] * py[None, :]
        
        # Unikamy log(0) i dzielenia przez 0
        nzs = pxy > 0
        mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
        mi_vals.append(mi)
        
        # Sprawdzamy warunek minimum lokalnego (z małym buforem na start)
        if len(mi_vals) > 2:
            if mi_vals[-2] < mi_vals[-3] and mi_vals[-2] < mi_vals[-1]:
                return lag - 1 
                
    return np.argmin(mi_vals) + 1

def estimate_embedding_dim_corrint(signal: np.ndarray, delay: int, max_dim: int = 8, saturation_threshold: float = 0.1) -> int:
    """Metoda nasycenia (G-P)."""
    signal = np.asarray(signal).flatten()
    calc_signal = signal[:min(len(signal), 5000)]
    corr_dims = []
    
    for dim in range(1, max_dim + 1):
        try:
            d2, _, _, _ = correlation_dimension_and_entropy(calc_signal, dim, delay)
            corr_dims.append(d2)
        except: corr_dims.append(0)
    
    corr_dims = np.array(corr_dims)
    for i in range(len(corr_dims) - 1):
        m = i + 1
        d2_curr = corr_dims[i]
        d2_next = corr_dims[i+1]
        
        rel_change = (d2_next - d2_curr) / d2_curr if d2_curr > 0 else 1.0
        
        # 1. Nasycenie
        is_saturated = rel_change < saturation_threshold
        # 2. Space Filling Check - POPRAWIONE
        # Wymuszamy, aby wymiar przestrzeni (m) był o pełną 0.8 większy od wymiaru fraktalnego D2.
        fits_in_space = d2_curr < (m - 0.8)
            
        if is_saturated and fits_in_space:
            return m
    return max_dim

def create_delay_embedding(signal: np.ndarray, embedding_dim: int, delay: int) -> np.ndarray:
    signal = np.asarray(signal).flatten()
    n_samples = len(signal) - (embedding_dim - 1) * delay
    if n_samples <= 0: return np.zeros((1, embedding_dim))
    indices = np.arange(embedding_dim) * delay + np.arange(n_samples)[:, np.newaxis]
    return signal[indices]