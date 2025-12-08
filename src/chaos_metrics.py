import numpy as np
from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression

def correlation_integral_curve(data, m, tau, r_vals, n_samples=1000):
    """
    Pomocnicza funkcja licząca całkę korelacyjną C(r) dla zadanego wymiaru m.
    """
    N = len(data)
    M = N - (m - 1) * tau
    
    # Tworzenie orbity
    orbit = np.array([data[i : i + (m * tau) : tau] for i in range(M)])
    
    # Próbkowanie dla wydajności
    if len(orbit) > n_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(orbit), n_samples, replace=False)
        orbit = orbit[idx]
        M_actual = n_samples
    else:
        M_actual = len(orbit)

    tree = cKDTree(orbit)
    C_r = []
    
    for r in r_vals:
        # count_neighbors zwraca liczbę par. Odejmujemy M_actual (pary punkt-sam-ze-sobą)
        count = tree.count_neighbors(tree, r) - M_actual
        if M_actual > 1:
            norm_count = count / (M_actual * (M_actual - 1))
        else:
            norm_count = 0
        C_r.append(norm_count)
        
    return np.array(C_r)

def correlation_dimension_and_entropy(data, m, tau, r_vals=None):
    """
    Estymacja wymiaru korelacyjnego (D2) oraz entropii korelacyjnej (K2).
    """
    if r_vals is None:
        std_data = np.std(data)
        r_vals = np.logspace(np.log10(std_data * 0.05), np.log10(std_data * 2.0), 20)
    
    C_m = correlation_integral_curve(data, m, tau, r_vals)
    C_m1 = correlation_integral_curve(data, m+1, tau, r_vals)
    
    valid = (C_m > 0) & (C_m1 > 0)
    
    if np.sum(valid) < 3:
        return 0.0, 0.0, np.log(r_vals), np.zeros_like(r_vals)

    log_r = np.log(r_vals[valid])
    log_Cm = np.log(C_m[valid])
    log_Cm1 = np.log(C_m1[valid])
    
    # D2
    reg = LinearRegression().fit(log_r.reshape(-1, 1), log_Cm)
    D2 = reg.coef_[0]
    
    # K2
    differences = (log_Cm - log_Cm1)
    K2_est = np.mean(differences)
    K2 = max(0.0, K2_est)
    
    return D2, K2, log_r, log_Cm

def largest_lyapunov_exponent(data, m, tau, dt=0.01, k=5):
    """
    Estymacja LLE metodą Rosensteina.
    """
    N = len(data)
    M = N - (m - 1) * tau
    orbit = np.array([data[i : i + (m * tau) : tau] for i in range(M)])
    
    tree = cKDTree(orbit)
    dists, separate_idxs = tree.query(orbit, k=k+1) 
    nearest_idxs = separate_idxs[:, 1] 
    
    max_iter = min(50, M - 1) 
    divergence = []
    
    for i in range(max_iter):
        dist_sum = 0
        count = 0
        step = 1 if M < 5000 else 10
        
        for j in range(0, M - i, step):
            idx_curr = j
            idx_neigh = nearest_idxs[j]
            
            if idx_neigh + i < M:
                v1 = orbit[idx_curr+i]
                v2 = orbit[idx_neigh+i]
                d = np.linalg.norm(v1 - v2)
                if d > 1e-10:
                    dist_sum += np.log(d)
                    count += 1
        if count > 0:
            divergence.append(dist_sum / count)
            
    time_steps = np.arange(len(divergence)) * dt
    if len(divergence) > 5:
        reg = LinearRegression().fit(time_steps.reshape(-1, 1), np.array(divergence))
        return reg.coef_[0]
    return 0.0

def box_counting_dimension(data, m, tau, bins_range=(2, 50)):
    """
    Wymiar pudełkowy (Box-counting).
    POPRAWIONA WERSJA: Używa podejścia 'sparse' (zbiór aktywnych pudełek),
    aby uniknąć MemoryError przy wysokich wymiarach (np. m=8).
    """
    N = len(data)
    M = N - (m - 1) * tau
    
    # Rekonstrukcja
    orbit = np.array([data[i : i + (m * tau) : tau] for i in range(M)])
    
    # Normalizacja do [0, 1]
    orbit_min = orbit.min(axis=0)
    orbit_max = orbit.max(axis=0)
    # Dodajemy mały epsilon do mianownika, żeby uniknąć dzielenia przez zero
    # i żeby max value wpadało do ostatniego pudełka, a nie poza zakres
    orbit_norm = (orbit - orbit_min) / (orbit_max - orbit_min + 1e-9)
    
    counts = []
    sizes = []
    
    # Lista liczby podziałów (od 2 do 50)
    bins_list = np.unique(np.logspace(np.log10(2), np.log10(bins_range[1]), 10).astype(int))
    
    for bins in bins_list:
        # Zamiast np.histogramdd, używamy set() do przechowywania indeksów
        # To jest kluczowe dla uniknięcia MemoryError!
        # Obliczamy indeksy pudełek dla każdego punktu
        box_indices = np.floor(orbit_norm * bins).astype(int)
        
        # Konwertujemy wiersze na tuple (bo są hashowalne) i wrzucamy do setu
        # Set automatycznie usuwa duplikaty -> otrzymujemy liczbę zajętych pudełek
        occupied_boxes = set(map(tuple, box_indices))
        N_eps = len(occupied_boxes)
        
        if N_eps > 0:
            counts.append(np.log(N_eps))
            sizes.append(np.log(bins)) 
            
    if len(counts) > 2:
        # D_box to nachylenie wykresu log(N) vs log(1/eps) = log(bins)
        reg = LinearRegression().fit(np.array(sizes).reshape(-1, 1), np.array(counts))
        return reg.coef_[0]
        
    return 0.0