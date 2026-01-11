import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable

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

def caputo_fractional_ode_solver(alpha: float, rhs: Callable, y0: np.ndarray, t: np.ndarray, memory_length: int = 2000) -> np.ndarray:
    """
    Rozwiązuje układ FDE metodą Grünwald-Letnikov.
    """
    n = len(t)
    d = len(y0)
    y = np.zeros((n, d))
    y[0] = y0
    
    h = t[1] - t[0] # Krok czasowy
    h_alpha = h ** alpha
    
    # === POPRAWKA ===
    # Limit pamięci
    limit_mem = min(n, memory_length)
    
    # Tablica wag musi mieć rozmiar limit_mem + 1, aby pomieścić indeksy od 0 do limit_mem.
    # Wzór GL sumuje od j=1 do k. Jeśli k=2000, potrzebujemy w[2000].
    w = np.zeros(limit_mem + 1) 
    
    w[0] = 1.0
    for j in range(1, limit_mem + 1):
        w[j] = (1 - (alpha + 1) / j) * w[j-1]
    
    print(f"  -> Fractional Solver (GL Method, alpha={alpha:.2f}, memory={limit_mem}) initialized.")
    
    for i in range(n - 1):
        # 1. Określenie długości historii w tym kroku
        # Nie może przekroczyć limit_mem
        current_len = min(i + 1, limit_mem)
        
        # 2. Pobranie historii stanów (odwrócona kolejność: y[i], y[i-1]...)
        # shape: (current_len, d)
        history = y[i+1-current_len : i+1][::-1] 
        
        # 3. Pobranie odpowiednich wag
        # Potrzebujemy wag w_1, w_2, ..., w_{current_len}
        # shape: (current_len,)
        weights = w[1 : current_len + 1]
        
        # 4. Obliczenie splotu (pamięci)
        # np.dot obsłuży mnożenie wektora wag przez kolumny historii
        memory_term = np.dot(weights, history)
        
        # 5. Obliczenie funkcji i krok naprzód
        f_val = rhs(t[i], y[i])
        
        # Schemat GL przesunięty: y_{n+1} = h^alpha * f(y_n) - sum(w_j * y_{n+1-j})
        y[i+1] = h_alpha * f_val - memory_term

    return y