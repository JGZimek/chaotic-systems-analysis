import numpy as np
from nolds import hurst_rs

def estimate_hurst_rs(signal: np.ndarray) -> float:
    """Hurst exponent via Rescaled Range (R/S) analysis."""
    signal = np.asarray(signal).flatten()
    if len(signal) < 100: return None
    try:
        return float(hurst_rs(signal))
    except:
        return None

def classify_hurst(h: float) -> str:
    if h is None: return "Unable"
    if h < 0.45: return "Mean-reverting"
    elif h <= 0.55: return "Random walk"
    else: return "Trending"

def analyze_hurst(signal: np.ndarray) -> dict:
    h = estimate_hurst_rs(signal)
    h_val = h if h is not None else 0.5
    return {'h': h_val, 'trend': classify_hurst(h_val)}