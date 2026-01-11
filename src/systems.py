import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable

# ==========================================
# 1. SYSTEM LORENZA (Klasyczny, rzędu 1)
# ==========================================

def lorenz_rhs(t: float, y: np.ndarray, sigma: float = 10.0, rho: float = 28.0, beta: float = 8/3) -> np.ndarray:
    """
    Definiuje prawą stronę równań różniczkowych systemu Lorenza.
    Układ:
        dx/dt = sigma * (y - x)
        dy/dt = x * (rho - z) - y
        dz/dt = x * y - beta * z
    """
    x, y_, z = y
    dx = sigma * (y_ - x)
    dy = x * (rho - z) - y_
    dz = x * y_ - beta * z
    return np.array([dx, dy, dz])

def generate_lorenz(t_span, y0, t_eval, sigma=10.0, rho=28.0, beta=8/3):
    """
    Rozwiązuje układ Lorenza numerycznie przy użyciu standardowej biblioteki SciPy.
    
    Użyta metoda: RK45 (Runge-Kutta rzędu 4/5) - standard przemysłowy dla
    układów nieułamkowych (rzędu całkowitego).
    """
    sol = solve_ivp(
        lambda t, y: lorenz_rhs(t, y, sigma, rho, beta),
        t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-8
    )
    # Transpozycja (T) jest potrzebna, aby wynik miał kształt (N, 3) a nie (3, N)
    return sol.y.T

# ==========================================
# 2. SYSTEM SPROTTA K (Klasyczny, rzędu 1)
# ==========================================

def sprott_k_rhs(t: float, y: np.ndarray, a: float = 0.3) -> np.ndarray:
    """
    Równania systemu Sprotta (przypadek K).
    Charakteryzuje się prostszą budową algebraiczną niż Lorenz,
    ale generuje równie złożony chaos.
    """
    x, y_, z = y
    dx = -z + x * y_
    dy = x - y_
    dz = x + a * z
    return np.array([dx, dy, dz])

def generate_sprott_k(t_span, y0, t_eval, a=0.3):
    """
    Rozwiązuje klasyczny układ Sprotta K metodą Rungego-Kutty.
    Służy jako punkt odniesienia dla wersji ułamkowej.
    """
    sol = solve_ivp(
        lambda t, y: sprott_k_rhs(t, y, a),
        t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-8
    )
    return sol.y.T

# ==========================================
# 3. SOLVER UŁAMKOWY (Fractional Derivative)
# ==========================================

def gl_fractional_ode_solver(alpha: float, rhs: Callable, y0: np.ndarray, t: np.ndarray, memory_length: int = 2000) -> np.ndarray:
    """
    Rozwiązuje układ równań różniczkowych ułamkowego rzędu (FDE).
    
    Metoda: Grünwald-Letnikov (GL).
    Jest to numeryczna realizacja definicji pochodnej ułamkowej, która opiera się
    na dyskretnym splocie historii procesu z wagami binomialnymi.
    
    Argumenty:
        alpha: Rząd pochodnej (0 < alpha <= 1). Dla alpha=1 mamy zwykłe ODE.
               Im mniejsze alpha, tym większe tłumienie ("lepkość" pamięci).
        rhs:   Funkcja prawej strony równania (np. sprott_k_rhs).
        y0:    Warunki początkowe [x0, y0, z0].
        t:     Wektor czasu (musi być równomierny, np. co 0.01s).
        memory_length: Długość bufora pamięci (Zasada Krótkiej Pamięci).
                       Optymalizacja, która zapobiega wykładniczemu wzrostowi 
                       czasu obliczeń.
    """
    n = len(t)
    d = len(y0)
    y = np.zeros((n, d)) # Tablica na wynikową trajektorię
    y[0] = y0            # Ustawienie punktu startowego
    
    # Krok czasowy (musi być stały dla metody GL)
    h = t[1] - t[0] 
    # Prekalkulacja czynnika skalującego h^alpha
    h_alpha = h ** alpha
    
    # === OPTYMALIZACJA PAMIĘCI ===
    # Zamiast pamiętać całą historię od t=0 (co byłoby bardzo wolne dla dużych N),
    # stosujemy zasadę "Short Memory Principle". Pamiętamy tylko 'memory_length' kroków wstecz.
    # Wagi dla bardzo dawnych próbek są bliskie zera, więc ich pominięcie nie generuje dużego błędu.
    limit_mem = min(n, memory_length)
    
    # === PREKALKULACJA WAG GL ===
    # Wagi w metodzie GL liczy się ze wzoru rekurencyjnego opartego na symbolu Newtona:
    # w_0 = 1
    # w_j = (1 - (alpha + 1)/j) * w_{j-1}
    # Liczymy je RAZ przed pętlą, zamiast w każdym kroku (ogromne przyspieszenie).
    w = np.zeros(limit_mem + 1)
    w[0] = 1.0
    for j in range(1, limit_mem + 1):
        w[j] = (1 - (alpha + 1) / j) * w[j-1]
    
    print(f"  -> Fractional Solver (Grünwald-Letnikov, alpha={alpha:.2f}, memory={limit_mem}) initialized.")
    
    # Główna pętla symulacji (krok po kroku)
    for i in range(n - 1):
        # 1. Określenie ile kroków wstecz bierzemy pod uwagę w tej iteracji
        current_len = min(i + 1, limit_mem)
        
        # 2. Pobranie historii stanów (odwrócona kolejność: od najnowszego do najstarszego)
        # y[i] to stan obecny, y[i-1] to poprzedni itd.
        # Składnia [::-1] odwraca tablicę.
        history = y[i+1-current_len : i+1][::-1] 
        
        # 3. Pobranie odpowiednich wag dla tej historii
        # (pomijamy w[0], bo w[0] dotyczy wyrazu przyszłego w schemacie niejawnym, 
        # tutaj używamy schematu przesuniętego).
        weights = w[1 : current_len + 1]
        
        # 4. OBLICZENIE SPLOTU (MEMORY TERM)
        # To jest serce pochodnej ułamkowej. Mnożymy przeszłe stany przez wagi.
        # Używamy np.dot (iloczyn skalarny) - to jest wektoryzacja, 
        # działa znacznie szybciej niż pętla 'for'.
        memory_term = np.dot(weights, history)
        
        # 5. Obliczenie wartości funkcji pola wektorowego w bieżącym punkcie
        f_val = rhs(t[i], y[i])
        
        # 6. KROK NUMERYCZNY (Schemat GL)
        # Wzór: y_{n+1} = (h^alpha * f(y_n)) - (suma ważona historii)
        # Odpowiada to dyskretyzacji operatora D^alpha.
        y[i+1] = h_alpha * f_val - memory_term

    return y