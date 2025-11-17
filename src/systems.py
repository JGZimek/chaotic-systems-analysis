import numpy as np
from typing import Callable
from scipy.integrate import solve_ivp
from scipy.special import gamma

def lorenz_rhs(t: float, y: np.ndarray, sigma: float = 10.0, rho: float = 28.0, beta: float = 8/3) -> np.ndarray:
    """
    Computes the right-hand side of the Lorenz system of equations.
    """
    x, y_, z = y
    dx = sigma * (y_ - x)
    dy = x * (rho - z) - y_
    dz = x * y_ - beta * z
    return np.array([dx, dy, dz])

def generate_lorenz(
        t_span: tuple[float, float],
        y0: np.ndarray,
        t_eval: np.ndarray,
        sigma: float = 10.0, rho: float = 28.0, beta: float = 8/3
    ) -> np.ndarray:
    """
    Generates a trajectory of the Lorenz system.
    """
    sol = solve_ivp(
        lambda t, y: lorenz_rhs(t, y, sigma, rho, beta),
        t_span,
        y0,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-8
    )
    return sol.y.T # Transpose to have shape (len(t_eval), 3)

def sprott_k_rhs(t: float, y: np.ndarray, a: float = 0.3) -> np.ndarray:
    """
    Computes the right-hand side of the Sprott K system of equations.
    """
    x, y_, z = y
    dx = -z + x * y_
    dy = x - y_
    dz = x + a * z
    return np.array([dx, dy, dz])

def generate_sprott_k(
        t_span: tuple[float, float],
        y0: np.ndarray,
        t_eval: np.ndarray,
        a: float = 0.3
    ) -> np.ndarray:
    """
    Generates a trajectory of the Sprott K system.
    """
    sol = solve_ivp(
        lambda t, y: sprott_k_rhs(t, y, a),
        t_span,
        y0,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-8
    )
    return sol.y.T # Transpose to have shape (len(t_eval), 3)

def caputo_fractional_ode_solver(
        aplha: float,
        rhs: Callable[[float, np.ndarray], np.ndarray],
        y0: np.ndarray,
        t: np.ndarray
    ) -> np.ndarray:
    """
    Solves a Caputo fractional ODE using the Grünwald-Letnikov approximation.
    """
    n, d = len(t), len(y0)
    y = np.zeros((n, d))
    y[0] = y0
    for i in range(n - 1):
        dt = t[i + 1] - t[i]
        y[i + 1] = y[i] + (dt ** aplha) * rhs(t[i], y[i]) / gamma(aplha + 1)
    return y