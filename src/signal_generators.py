"""
Signal generation module for chaotic systems analysis.
Contains functions to generate baseline signals and chaotic systems.
"""

import numpy as np
from scipy.integrate import odeint


def generate_random_signal(n_samples: int, seed: int = None) -> np.ndarray:
    """
    Generate uniform random signal [0, 1].
    
    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility (optional)
    
    Returns:
        Random signal as numpy array
    """
    if seed is not None:
        np.random.seed(seed)
    
    return np.random.uniform(0, 1, n_samples)


def generate_periodic_signal(n_samples: int) -> np.ndarray:
    """
    Generate periodic signal: sin(n/100) + cos(n/50).
    
    Args:
        n_samples: Number of samples to generate
    
    Returns:
        Periodic signal as numpy array
    """
    n = np.arange(n_samples)
    return np.sin(n / 100) + np.cos(n / 50)


def lorenz_ode(state, t, sigma=10, rho=28, beta=8/3):
    """
    Lorenz system ODE equations.
    
    dx/dt = sigma(y - x)
    dy/dt = x(rho - z) - y
    dz/dt = xy - beta*z
    
    Args:
        state: [x, y, z] current state
        t: time (for odeint compatibility)
        sigma: Lorenz parameter (default: 10)
        rho: Lorenz parameter (default: 28)
        beta: Lorenz parameter (default: 8/3)
    
    Returns:
        [dx/dt, dy/dt, dz/dt]
    """
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]


def generate_lorenz_system(
    n_samples: int,
    initial_state: list = None,
    dt: float = 0.01,
    sigma: float = 10,
    rho: float = 28,
    beta: float = 8/3
) -> tuple:
    """
    Generate Lorenz system trajectory.
    
    Args:
        n_samples: Number of samples
        initial_state: Initial conditions [x0, y0, z0] (default: [1, 1, 1])
        dt: Time step (default: 0.01)
        sigma: Lorenz sigma parameter (default: 10)
        rho: Lorenz rho parameter (default: 28)
        beta: Lorenz beta parameter (default: 8/3)
    
    Returns:
        Tuple of (t, x, y, z) time series
    """
    if initial_state is None:
        initial_state = [1.0, 1.0, 1.0]
    
    # Time array
    t = np.arange(0, n_samples * dt, dt)
    
    # Integrate ODE
    solution = odeint(
        lorenz_ode,
        initial_state,
        t,
        args=(sigma, rho, beta)
    )
    
    x = solution[:, 0]
    y = solution[:, 1]
    z = solution[:, 2]
    
    return t, x, y, z


# Example usage for testing
if __name__ == "__main__":
    print("Testing signal_generators module...")
    
    # Test random signal
    random_sig = generate_random_signal(1000)
    print(f"Random signal: shape={random_sig.shape}, mean={random_sig.mean():.3f}")
    
    # Test periodic signal
    periodic_sig = generate_periodic_signal(1000)
    print(f"Periodic signal: shape={periodic_sig.shape}, range=[{periodic_sig.min():.3f}, {periodic_sig.max():.3f}]")
    
    # Test Lorenz
    t, x, y, z = generate_lorenz_system(5000)
    print(f"Lorenz system: shape={x.shape}, time range=[{t.min():.2f}, {t.max():.2f}]")
