import numpy as np
from statsmodels.tsa.stattools import acf
from scipy.stats import gaussian_kde
from nolds import corr_dim
from typing import Tuple, Dict

def estimate_delay_acf(signal: np.ndarray, max_lag: int = 100) -> int:
    """
    Estimate time delay using ACF.
    Returns the first zero crossing.
    """
    # Autokorelacja do wyznaczenia opóźnienia
    try:
        acf_vals = acf(signal, nlags=max_lag, fft=True)
        # Znajdź indeksy, gdzie znak się zmienia (przejście przez zero)
        zero_crossings = np.where(np.diff(np.sign(acf_vals)))[0]
        if len(zero_crossings) > 0:
            return int(zero_crossings[0])
    except Exception:
        pass
    return max(1, max_lag // 4) # Fallback


def estimate_delay_mi_histogram(signal: np.ndarray, max_lag: int = 100) -> int:
    """
    Estimate time delay using Mutual Information (Histogram method).
    Returns the FIRST LOCAL MINIMUM (Fraser & Swinney algorithm).
    """
    # Wzajemna informacja: histogram adaptacyjny
    signal = np.asarray(signal).flatten()
    bins = int(np.floor(np.sqrt(len(signal) / 5))) # Heurystyka doboru koszyków
    bins = max(5, min(bins, 100))
    
    mi_vals = []
    
    # Obliczamy MI dla każdego lagu
    for lag in range(1, max_lag + 1):
        x = signal[:-lag]
        y = signal[lag:]
        
        # Szybki histogram 2D
        hist_2d, _, _ = np.histogram2d(x, y, bins=bins)
        
        # Normalizacja do gęstości prawdopodobieństwa
        pxy = hist_2d / np.sum(hist_2d)
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        
        px_py = px[:, None] * py[None, :]
        
        # Obliczenie MI (unikając log(0))
        nzs = pxy > 0
        mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
        mi_vals.append(mi)
        
        # --- WARUNEK STOPU: PIERWSZE LOKALNE MINIMUM ---
        # Sprawdzamy czy wartość spadła, a potem wzrosła
        if len(mi_vals) > 2:
            if mi_vals[-2] < mi_vals[-3] and mi_vals[-2] < mi_vals[-1]:
                return lag - 1 # Zwracamy indeks minimum
                
    # Jeśli nie znaleziono lokalnego minimum, zwróć globalne lub argmin
    return np.argmin(mi_vals) + 1


def estimate_delay_mi_kde(signal: np.ndarray, max_lag: int = 100) -> int:
    """
    Estimate time delay using Mutual Information (KDE method).
    Very slow, used mainly for verification. Returns first local minimum.
    """
    # Estymator jądrowy
    signal = np.asarray(signal).flatten()
    
    # Downsampling dla wydajności (KDE jest bardzo wolne O(N^2))
    if len(signal) > 1000:
        step = len(signal) // 1000
        signal = signal[::step]
    
    mi_vals = []
    for lag in range(1, max_lag + 1):
        x = signal[:-lag]
        y = signal[lag:]
        
        try:
            kernel = gaussian_kde(np.vstack([x, y]))
            # Estymacja w punktach próbkowania
            pts = np.vstack([x, y])
            p_xy = kernel(pts)
            
            # Marginesy (aproksymacja)
            p_x = gaussian_kde(x)(x)
            p_y = gaussian_kde(y)(y)
            
            mi = np.mean(np.log(p_xy / (p_x * p_y + 1e-10)))
            mi_vals.append(mi)
        except:
            mi_vals.append(np.inf)
            
        # Warunek pierwszego lokalnego minimum
        if len(mi_vals) > 2:
            if mi_vals[-2] < mi_vals[-3] and mi_vals[-2] < mi_vals[-1]:
                return lag - 1

    return np.argmin(mi_vals) + 1

def estimate_embedding_dim_corrint(
    signal: np.ndarray,
    delay: int,
    max_dim: int = 10,
    saturation_threshold: float = 0.005 # ZMNIEJSZONO PRÓG (było np. 0.05)
) -> int:
    """
    Estimate embedding dimension using correlation integral saturation.
    """
    signal = np.asarray(signal).flatten()
    # Zwiększamy liczbę próbek do analizy dla większej precyzji (jeśli masz szybki PC)
    calc_signal = signal[:min(len(signal), 3000)] 
    
    corr_dims = []
    
    # Obliczamy D2 dla wymiarów od 1 do max_dim
    for dim in range(1, max_dim + 1):
        try:
            embedded = create_delay_embedding(calc_signal, dim, delay)
            # fit='poly' jest stabilniejsze
            d2 = corr_dim(embedded, emb_dim=1, fit='poly') 
            corr_dims.append(d2)
        except Exception:
            corr_dims.append(corr_dims[-1] if corr_dims else 0)
    
    corr_dims = np.array(corr_dims)
    
    # Wyświetlmy wartości w konsoli dla debugowania (opcjonalne)
    # print(f"DEBUG D2 values: {corr_dims}")

    # LOGIKA NASYCENIA (Saturation)
    # Szukamy momentu, gdzie przyrost jest minimalny LUB zaczyna spadać/fluktuować
    for i in range(1, len(corr_dims) - 1):
        slope = corr_dims[i+1] - corr_dims[i]
        
        # Jeśli przyrost jest znikomy
        if slope < saturation_threshold:
            # Ale upewnijmy się, że to nie jest fałszywy stop (np. dla Lorenza D2 powinno być > 2.0)
            if corr_dims[i] > 1.95: 
                # Jeśli mamy już ok. 2.0 i nie rośnie, to dla bezpieczeństwa bierzemy i+1 (czyli 3)
                # bo struktury >2.0 wymagają pudełka 3D
                return i + 2 
            return i + 1

    # Jeśli nie znaleziono nasycenia, używamy metody "kolanka" (największa zmiana nachylenia)
    diffs = np.diff(corr_dims)
    if len(diffs) > 2:
        # Szukamy gdzie przyrost gwałtownie maleje
        elbow_idx = np.argmin(np.diff(diffs)) 
        return elbow_idx + 2

    return max(2, max_dim - 1)

def create_delay_embedding(
    signal: np.ndarray,
    embedding_dim: int,
    delay: int
) -> np.ndarray:
    """Create delay embedding (Takens' theorem)."""
    signal = np.asarray(signal).flatten()
    n = len(signal)
    n_samples = n - (embedding_dim - 1) * delay
    
    if n_samples <= 0:
        # Fallback dla zbyt krótkich sygnałów
        return np.zeros((1, embedding_dim))
    
    indices = np.arange(embedding_dim) * delay + np.arange(n_samples)[:, np.newaxis]
    return signal[indices]