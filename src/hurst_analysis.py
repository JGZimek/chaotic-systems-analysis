import numpy as np
from nolds import dfa

def estimate_hurst_dfa(signal: np.ndarray) -> float:
    """Hurst exponent via DFA."""
    signal = np.asarray(signal).flatten()
    if len(signal) < 100:
        return None
    try:
        return float(dfa(signal))
    except Exception:
        return None

def classify_hurst(h: float) -> str:
    """Classify trend based on Hurst exponent."""
    if h is None:
        return "Unable"
    if h < 0.5:
        return "Mean-reverting"
    elif h < 0.6:
        return "Random walk"
    else:
        return "Trending"

def analyze_hurst(signal: np.ndarray) -> dict:
    """Return only H and trend classification."""
    h = estimate_hurst_dfa(signal)
    h_val = h if h is not None else 0.5
    return {
        'h': h_val,
        'trend': classify_hurst(h_val),
    }
