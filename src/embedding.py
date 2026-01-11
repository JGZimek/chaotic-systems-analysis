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
    """Fraser-Swinney: Pierwsze minimum."""
    signal = np.asarray(signal).flatten()
    bins = int(np.floor(np.sqrt(len(signal) / 5))) 
    bins = max(5, min(bins, 100))
    mi_vals = []
    
    for lag in range(1, max_lag + 1):
        x = signal[:-lag]; y = signal[lag:]
        hist_2d, _, _ = np.histogram2d(x, y, bins=bins)
        pxy = hist_2d / np.sum(hist_2d)
        px = np.sum(pxy, axis=1); py = np.sum(pxy, axis=0)
        px_py = px[:, None] * py[None, :]
        nzs = pxy > 0
        mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
        mi_vals.append(mi)
        if len(mi_vals) > 2 and mi_vals[-2] < mi_vals[-3] and mi_vals[-2] < mi_vals[-1]:
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
        # 2. Space Filling Check: Atraktor nie może wypełniać całej przestrzeni
        fits_in_space = d2_curr < (m - 0.4)
        
        if is_saturated and fits_in_space:
            return m
    return max_dim

def create_delay_embedding(signal: np.ndarray, embedding_dim: int, delay: int) -> np.ndarray:
    signal = np.asarray(signal).flatten()
    n_samples = len(signal) - (embedding_dim - 1) * delay
    if n_samples <= 0: return np.zeros((1, embedding_dim))
    indices = np.arange(embedding_dim) * delay + np.arange(n_samples)[:, np.newaxis]
    return signal[indices]