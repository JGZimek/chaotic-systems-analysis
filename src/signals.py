import numpy as np
from typing import Optional, Any

def get_random_signal(length: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generates a random signal (uniform in [0,1]) of the given length.
    """

    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, length)

def get_periodic_signal(length: int, freq: float = 1.0, phase: float = 0.0) -> np.ndarray:
    """
    Generates a periodic signal sin(f*t) + cos(f*t) of the given length.
    """

    t = np.linspace(0, 2 * np.pi, length)
    return np.sin(freq * t + phase) + np.cos(freq * t + phase)




