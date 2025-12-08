import numpy as np
from statsmodels.tsa.stattools import acf
from scipy.stats import gaussian_kde
from nolds import corr_dim
from typing import Tuple, Dict

def estimate_delay_acf(signal: np.ndarray, max_lag: int = 100) -> int:
    try:
        acf_vals = acf(signal, nlags=max_lag, fft=True)
        zero_crossings = np.where(np.diff(np.sign(acf_vals)))[0]
        if len(zero_crossings) > 0:
            return int(zero_crossings[0])
    except Exception:
        pass
    return max(1, max_lag // 4) 

def estimate_delay_mi_histogram(signal: np.ndarray, max_lag: int = 100) -> int:
    signal = np.asarray(signal).flatten()
    bins = int(np.floor(np.sqrt(len(signal) / 5))) 
    bins = max(5, min(bins, 100))
    
    mi_vals = []
    
    for lag in range(1, max_lag + 1):
        x = signal[:-lag]
        y = signal[lag:]
        
        hist_2d, _, _ = np.histogram2d(x, y, bins=bins)
        pxy = hist_2d / np.sum(hist_2d)
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        px_py = px[:, None] * py[None, :]
        
        nzs = pxy > 0
        mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
        mi_vals.append(mi)
        
        if len(mi_vals) > 2:
            if mi_vals[-2] < mi_vals[-3] and mi_vals[-2] < mi_vals[-1]:
                return lag - 1 
                
    return np.argmin(mi_vals) + 1

def estimate_delay_mi_kde(signal: np.ndarray, max_lag: int = 100) -> int:
    signal = np.asarray(signal).flatten()
    if len(signal) > 1000:
        step = len(signal) // 1000
        signal = signal[::step]
    
    mi_vals = []
    for lag in range(1, max_lag + 1):
        x = signal[:-lag]
        y = signal[lag:]
        try:
            kernel = gaussian_kde(np.vstack([x, y]))
            pts = np.vstack([x, y])
            p_xy = kernel(pts)
            p_x = gaussian_kde(x)(x)
            p_y = gaussian_kde(y)(y)
            mi = np.mean(np.log(p_xy / (p_x * p_y + 1e-10)))
            mi_vals.append(mi)
        except:
            mi_vals.append(np.inf)
        if len(mi_vals) > 2:
            if mi_vals[-2] < mi_vals[-3] and mi_vals[-2] < mi_vals[-1]:
                return lag - 1
    return np.argmin(mi_vals) + 1

def estimate_embedding_dim_corrint(
    signal: np.ndarray,
    delay: int,
    max_dim: int = 8, # Zmniejszmy max_dim, żeby nie szukał w nieskończoność
    saturation_threshold: float = 0.05 
) -> int:
    """
    Estimate embedding dimension.
    """
    signal = np.asarray(signal).flatten()
    calc_signal = signal[:min(len(signal), 3000)] 
    
    corr_dims = []
    
    for dim in range(1, max_dim + 1):
        try:
            embedded = create_delay_embedding(calc_signal, dim, delay)
            d2 = corr_dim(embedded, emb_dim=1, fit='poly') 
            corr_dims.append(d2)
        except Exception:
            corr_dims.append(0)
    
    corr_dims = np.array(corr_dims)
    
    # 1. Sprawdzenie nasycenia (przyrost mniejszy niż 5%)
    for i in range(1, len(corr_dims) - 1):
        diff = corr_dims[i+1] - corr_dims[i]
        
        # Warunek 1: Płaskie plateau
        if diff < saturation_threshold * corr_dims[i]:
            # Bezpiecznik dla chaosu (Lorenz D2~2.06 -> dE min 3)
            if corr_dims[i] > 1.8:
                return max(3, i + 1)
            return i + 1

    # 2. Jeśli nie ma nasycenia, sprawdźmy "kolanko" (gdzie przyrost zwalnia najmocniej)
    # Obliczamy drugą pochodną (zmianę przyrostów)
    diffs = np.diff(corr_dims)
    if len(diffs) > 2:
        # Szukamy punktu, gdzie przyrost drastycznie maleje
        changes = diffs[:-1] - diffs[1:]
        best_elbow = np.argmax(changes) + 2
        return min(best_elbow + 1, max_dim)

    return max_dim

def create_delay_embedding(
    signal: np.ndarray,
    embedding_dim: int,
    delay: int
) -> np.ndarray:
    signal = np.asarray(signal).flatten()
    n = len(signal)
    n_samples = n - (embedding_dim - 1) * delay
    
    if n_samples <= 0:
        return np.zeros((1, embedding_dim))
    
    indices = np.arange(embedding_dim) * delay + np.arange(n_samples)[:, np.newaxis]
    return signal[indices]