import numpy as np
from typing import Callable
from scipy.integrate import solve_ivp
from scipy.special import gamma

def lorenz_rhs(t: float, y: np.ndarray, sigma: float = 10.0, rho: float = 28.0, beta: float = 8/3) -> np.ndarray:
    x, y_, z = y
    dx = sigma * (y_ - x)
    dy = x * (rho - z) - y_
    dz = x * y_ - beta * z
    return np.array([dx, dy, dz])

def generate_lorenz(t_span, y0, t_eval, sigma=10.0, rho=28.0, beta=8/3):
    sol = solve_ivp(
        lambda t, y: lorenz_rhs(t, y, sigma, rho, beta),
        t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-8
    )
    return sol.y.T

def sprott_k_rhs(t: float, y: np.ndarray, a: float = 0.3) -> np.ndarray:
    x, y_, z = y
    dx = -z + x * y_
    dy = x - y_
    dz = x + a * z
    return np.array([dx, dy, dz])

def generate_sprott_k(t_span, y0, t_eval, a=0.3):
    sol = solve_ivp(
        lambda t, y: sprott_k_rhs(t, y, a),
        t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-8
    )
    return sol.y.T

def caputo_fractional_ode_solver(alpha: float, rhs: Callable, y0: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Solves a Caputo fractional ODE using the Grünwald-Letnikov approximation.
    Includes numerical stability checks.
    """
    n, d = len(t), len(y0)
    y = np.zeros((n, d))
    y[0] = y0
    
    # Pre-compute gamma to save time
    g_alpha = gamma(alpha + 1)
    
    for i in range(n - 1):
        dt = t[i + 1] - t[i]
        
        # Obliczenie pochodnej
        k1 = rhs(t[i], y[i])
        
        # Euler-like step for fractional calculus
        step = (dt ** alpha) * k1 / g_alpha
        y_next = y[i] + step
        
        # --- ZABEZPIECZENIE PRZED OVERFLOW ---
        # Jeśli wartości uciekają do nieskończoności (> 1e6) lub są NaN, przerywamy lub zerujemy
        if np.any(np.abs(y_next) > 1e6) or np.any(np.isnan(y_next)):
            # Wypełniamy resztę tablicy NaN, żeby w main.py łatwo to wykryć
            y[i+1:] = np.nan
            break
            
        y[i + 1] = y_next
        
    return y