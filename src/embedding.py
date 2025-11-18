import numpy as np
from statsmodels.tsa.stattools import acf
from scipy.stats import gaussian_kde
from nolds import corr_dim
from typing import Tuple, Dict

def estimate_delay_acf(signal: np.ndarray, max_lag: int = 100) -> int:
    """Estimate time delay using ACF (statsmodels - gotowa)."""
    acf_vals = acf(signal, nlags=max_lag, fft=True)
    zero_crossings = np.where(np.diff(np.sign(acf_vals)))[0]
    return int(zero_crossings[0]) if len(zero_crossings) > 0 else max_lag // 2


def estimate_delay_mi_histogram(signal: np.ndarray, max_lag: int = 100) -> int:
    """Estimate time delay using MI (adaptive histogram - Fraser & Swinney)."""
    signal = np.asarray(signal).flatten()
    bins = max(2, int(np.sqrt(len(signal))))
    
    mi_vals = []
    for lag in range(1, max_lag + 1):
        x, y = signal[:-lag], signal[lag:]
        
        hist_2d, _, _ = np.histogram2d(x, y, bins=bins)
        pxy = hist_2d / hist_2d.sum()
        px, py = pxy.sum(axis=1), pxy.sum(axis=0)
        
        pxy_flat = pxy.flatten()
        mask = pxy_flat > 0
        mi = np.sum(pxy_flat[mask] * np.log(pxy_flat[mask] / np.outer(px, py).flatten()[mask]))
        mi_vals.append(mi)
    
    return np.argmin(mi_vals) + 1


def estimate_delay_mi_kde(signal: np.ndarray, max_lag: int = 100) -> int:
    """Estimate time delay using MI (kernel density estimator - scipy.gaussian_kde)."""
    signal = np.asarray(signal).flatten()
    
    mi_vals = []
    for lag in range(1, max_lag + 1):
        x = signal[:-lag]
        y = signal[lag:]
        xy = np.column_stack([x, y])
        
        kde_xy = gaussian_kde(xy.T)
        kde_x = gaussian_kde(x)
        kde_y = gaussian_kde(y)
        
        px_py = kde_x(x) * kde_y(y)
        px_py[px_py < 1e-10] = 1e-10
        
        mi = np.mean(np.log(kde_xy(xy.T) / px_py))
        mi_vals.append(mi)
    
    return np.argmin(mi_vals) + 1


def estimate_embedding_dim_corrint(
    signal: np.ndarray,
    delay: int,
    max_dim: int = 15,
    saturation_threshold: float = 0.01
) -> int:
    """
    Estimate embedding dimension using correlation integral saturation.
    """
    signal = np.asarray(signal).flatten()[:min(len(signal), 1000)]  # Subsample for speed
    corr_dims = []
    
    for dim in range(1, max_dim + 1):
        try:
            embedded = create_delay_embedding(signal, dim, delay)
            d2 = corr_dim(embedded, 2, fit='poly')
            corr_dims.append(d2)
        except:
            corr_dims.append(corr_dims[-1] if corr_dims else 0)
    
    corr_dims = np.array(corr_dims)
    
    # Find saturation: first dim where change < threshold
    for i in range(2, len(corr_dims)):
        if abs(corr_dims[i] - corr_dims[i-1]) < saturation_threshold:
            return i
    
    # Fallback: elbow method
    if len(corr_dims) > 3:
        elbow = np.argmin(np.diff(corr_dims, n=2)) + 2
        return int(elbow) if elbow > 1 else max(2, max_dim // 2)
    
    return max(2, max_dim // 2)

def create_delay_embedding(
    signal: np.ndarray,
    embedding_dim: int,
    delay: int
) -> np.ndarray:
    """Create delay embedding (Takens' theorem - wektoryzowane numpy)."""
    signal = np.asarray(signal).flatten()
    n = len(signal)
    n_samples = n - (embedding_dim - 1) * delay
    
    if n_samples <= 0:
        raise ValueError(f"Not enough data for embedding_dim={embedding_dim}, delay={delay}")
    
    indices = np.arange(embedding_dim) * delay + np.arange(n_samples)[:, np.newaxis]
    return signal[indices]


def analyze_phase_space_reconstruction(
    signal: np.ndarray,
    max_lag: int = 100,
    max_dim: int = 15
) -> Dict:
    """
    Complete phase space reconstruction analysis.
    Estymacja T i dE wszystkimi metodami.
    """
    signal = np.asarray(signal).flatten()
    
    # Estymacja opóźnienia T (3 metody)
    delay_acf = estimate_delay_acf(signal, max_lag=max_lag)
    delay_mi_hist = estimate_delay_mi_histogram(signal, max_lag=max_lag)
    delay_mi_kde = estimate_delay_mi_kde(signal, max_lag=max_lag)
    
    # Średnie opóźnienie
    delay_mean = int(np.mean([delay_acf, delay_mi_hist, delay_mi_kde]))
    
    # Estymacja wymiaru zanurzenia dE (przy średnim delay)
    embedding_dim = estimate_embedding_dim_corrint(signal, delay_mean, max_dim=max_dim)
    
    # Tworzymy ostateczny embedding
    embedding = create_delay_embedding(signal, embedding_dim, delay_mean)
    
    return {
        'delay_acf': delay_acf,
        'delay_mi_histogram': delay_mi_hist,
        'delay_mi_kde': delay_mi_kde,
        'delay_recommended': delay_mean,
        'embedding_dim': embedding_dim,
        'embedding': embedding,
        'embedding_shape': embedding.shape
    }
