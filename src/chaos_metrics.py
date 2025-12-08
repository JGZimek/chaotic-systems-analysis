import numpy as np
from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression

def correlation_integral_curve(data, m, tau, r_vals, n_samples=1000):
    """
    Pomocnicza funkcja licząca całkę korelacyjną C(r) dla zadanego wymiaru m.
    Używa cKDTree dla wydajności.
    """
    N = len(data)
    M = N - (m - 1) * tau
    
    # Tworzenie orbity (rekonstrukcja)
    orbit = np.array([data[i : i + (m * tau) : tau] for i in range(M)])
    
    # Próbkowanie dla wydajności (przy dużych N obliczenia trwają O(N^2))
    if len(orbit) > n_samples:
        # Używamy stałego seeda dla powtarzalności próbkowania między wymiarami m i m+1
        rng = np.random.default_rng(42)
        idx = rng.choice(len(orbit), n_samples, replace=False)
        orbit = orbit[idx]
        M_actual = n_samples
    else:
        M_actual = len(orbit)

    tree = cKDTree(orbit)
    C_r = []
    
    # Liczymy liczbę par w odległości < r
    for r in r_vals:
        # count_neighbors zwraca liczbę par. Odejmujemy M_actual (pary punkt-sam-ze-sobą)
        count = tree.count_neighbors(tree, r) - M_actual
        # Normalizacja: liczba par to M*(M-1)
        if M_actual > 1:
            norm_count = count / (M_actual * (M_actual - 1))
        else:
            norm_count = 0
        C_r.append(norm_count)
        
    return np.array(C_r)

def correlation_dimension_and_entropy(data, m, tau, r_vals=None):
    """
    Estymacja wymiaru korelacyjnego (D2) oraz entropii korelacyjnej (K2).
    Zgodnie z metodą Grassbergera-Procaccii.
    
    Returns:
        D2 (float): Wymiar korelacyjny
        K2 (float): Entropia korelacyjna (estymata KS entropy)
        log_r (array): Oś X do wykresu
        log_Cr (array): Oś Y do wykresu (dla wymiaru m)
    """
    # 1. Ustalenie zakresu promieni r, jeśli nie podano
    if r_vals is None:
        std_data = np.std(data)
        # Zakres logarytmiczny od małego ułamka odchylenia do połowy rozpiętości
        r_vals = np.logspace(np.log10(std_data * 0.05), np.log10(std_data * 2.0), 20)
    
    # 2. Obliczenie C(r) dla wymiaru m oraz m+1 (potrzebne do entropii)
    C_m = correlation_integral_curve(data, m, tau, r_vals)
    C_m1 = correlation_integral_curve(data, m+1, tau, r_vals)
    
    # Filtrowanie zer (logarytm)
    valid = (C_m > 0) & (C_m1 > 0)
    
    if np.sum(valid) < 3:
        return 0.0, 0.0, np.log(r_vals), np.zeros_like(r_vals)

    log_r = np.log(r_vals[valid])
    log_Cm = np.log(C_m[valid])
    log_Cm1 = np.log(C_m1[valid])
    
    # 3. Obliczenie D2 (nachylenie log C_m vs log r)
    # Fitujemy linię prostą do obszaru skalowania (całego dostępnego w tej prostej wersji)
    reg = LinearRegression().fit(log_r.reshape(-1, 1), log_Cm)
    D2 = reg.coef_[0]
    
    # 4. Obliczenie K2 (Entropia)
    # Teoretycznie: K2 ~ (1 / tau) * ln( C_m(r) / C_{m+1}(r) )
    # W skali logarytmicznej: K2 ~ (log(C_m) - log(C_{m+1})) / tau
    # K2 bierzemy jako średnią wartość różnicy w obszarze skalowania
    
    differences = (log_Cm - log_Cm1)
    # Dzielimy przez tau (zakładając tau jako jednostkę czasu próbkowania dt=1 w tym kontekście, 
    # lub można użyć tau*dt jeśli chcemy jednostki fizyczne czasu). 
    # W standardowych implementacjach często podaje się per iteracja/step:
    K2_est = np.mean(differences) 
    
    # Aby wynik był nieujemny (numeryczne artefakty mogą dać ujemne wartości przy szumie)
    K2 = max(0.0, K2_est)
    
    return D2, K2, log_r, log_Cm

def largest_lyapunov_exponent(data, m, tau, dt=0.01, k=5):
    """
    Estymacja największego wykładnika Lapunowa (LLE) metodą Rosensteina.
    """
    N = len(data)
    M = N - (m - 1) * tau
    
    # Rekonstrukcja
    orbit = np.array([data[i : i + (m * tau) : tau] for i in range(M)])
    
    # Znalezienie najbliższych sąsiadów
    tree = cKDTree(orbit)
    # k+1 bo najbliższy to punkt sam w sobie
    dists, separate_idxs = tree.query(orbit, k=k+1) 
    
    # Wybieramy najbliższego sąsiada (indeks 1, bo 0 to punkt własny)
    nearest_idxs = separate_idxs[:, 1] 
    
    # Śledzenie dywergencji
    max_iter = min(50, M - 1) 
    divergence = []
    
    for i in range(max_iter):
        dist_sum = 0
        count = 0
        # Próbkowanie co 10 punkt dla szybkości, jeśli orbita długa
        step = 1 if M < 5000 else 10
        
        for j in range(0, M - i, step):
            idx_curr = j
            idx_neigh = nearest_idxs[j]
            
            if idx_neigh + i < M:
                v1 = orbit[idx_curr+i]
                v2 = orbit[idx_neigh+i]
                d = np.linalg.norm(v1 - v2)
                if d > 1e-10: # Unikamy log(0)
                    dist_sum += np.log(d)
                    count += 1
        if count > 0:
            divergence.append(dist_sum / count)
            
    # Dopasowanie liniowe
    time_steps = np.arange(len(divergence)) * dt
    if len(divergence) > 5:
        # Fitujemy do liniowej części (pomijamy ewentualne stany początkowe/nasycenie)
        reg = LinearRegression().fit(time_steps.reshape(-1, 1), np.array(divergence))
        return reg.coef_[0]
    return 0.0

def box_counting_dimension(data, m, tau, bins_range=(2, 50)):
    """
    Wymiar pudełkowy (Box-counting).
    """
    N = len(data)
    M = N - (m - 1) * tau
    orbit = np.array([data[i : i + (m * tau) : tau] for i in range(M)])
    
    counts = []
    sizes = []
    
    orbit_min = orbit.min(axis=0)
    orbit_max = orbit.max(axis=0)
    orbit_norm = (orbit - orbit_min) / (orbit_max - orbit_min + 1e-10)
    
    # Sprawdzamy kilka wielkości pudełek
    # Używamy skali logarytmicznej dla bins, żeby punkty na wykresie były równomiernie rozłożone
    bins_list = np.unique(np.logspace(np.log10(2), np.log10(bins_range[1]), 10).astype(int))
    
    for bins in bins_list:
        H, _ = np.histogramdd(orbit_norm, bins=bins)
        N_eps = np.sum(H > 0)
        
        if N_eps > 0:
            counts.append(np.log(N_eps))
            sizes.append(np.log(bins)) 
            
    if len(counts) > 2:
        reg = LinearRegression().fit(np.array(sizes).reshape(-1, 1), np.array(counts))
        return reg.coef_[0]
    return 0.0