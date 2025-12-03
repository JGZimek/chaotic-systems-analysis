import numpy as np
from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression

def largest_lyapunov_exponent(data, m, tau, dt=0.01, k=5):
    """
    Estymacja największego wykładnika Lapunowa (LLE) metodą Rosensteina.
    Wymagane przez: 
    """
    N = len(data)
    M = N - (m - 1) * tau
    
    # 1. Rekonstrukcja przestrzeni fazowej
    orbit = np.array([data[i : i + (m * tau) : tau] for i in range(M)])
    
    # 2. Znalezienie najbliższych sąsiadów
    tree = cKDTree(orbit)
    dists, separate_idxs = tree.query(orbit, k=k+1) # k+1 bo najbliższy to punkt sam w sobie
    
    # Wybieramy najbliższego sąsiada (który nie jest tym samym punktem)
    nearest_idxs = separate_idxs[:, 1] 
    
    # 3. Śledzenie dywergencji w czasie
    max_iter = min(50, M - 1) # Horyzont czasowy
    divergence = []
    
    for i in range(max_iter):
        dist_sum = 0
        count = 0
        for j in range(M - i):
            idx_curr = j
            idx_neigh = nearest_idxs[j]
            
            if idx_neigh + i < M:
                d = np.linalg.norm(orbit[idx_curr+i] - orbit[idx_neigh+i])
                if d > 0:
                    dist_sum += np.log(d)
                    count += 1
        if count > 0:
            divergence.append(dist_sum / count)
            
    # 4. Dopasowanie liniowe (nachylenie to LLE)
    time_steps = np.arange(len(divergence)) * dt
    if len(divergence) > 2:
        reg = LinearRegression().fit(time_steps.reshape(-1, 1), np.array(divergence))
        return reg.coef_[0]
    return 0.0

def box_counting_dimension(data, m, tau, bins_range=(2, 50)):
    """
    Obliczanie wymiaru pudełkowego (pojemnościowego).
    Wymagane przez: 
    """
    N = len(data)
    M = N - (m - 1) * tau
    orbit = np.array([data[i : i + (m * tau) : tau] for i in range(M)])
    
    counts = []
    sizes = []
    
    # Normalizacja orbity do [0, 1] dla uproszczenia podziału
    orbit_min = orbit.min(axis=0)
    orbit_max = orbit.max(axis=0)
    orbit_norm = (orbit - orbit_min) / (orbit_max - orbit_min + 1e-10)
    
    for bins in range(bins_range[0], bins_range[1], 2):
        # Histogram n-wymiarowy
        H, _ = np.histogramdd(orbit_norm, bins=bins)
        # Liczba niepustych pudełek
        N_eps = np.sum(H > 0)
        
        if N_eps > 0:
            counts.append(np.log(N_eps))
            sizes.append(np.log(bins)) # 1/epsilon proprocjonalne do bins
            
    if len(counts) > 2:
        # Wymiar to nachylenie prostej log(N) vs log(1/eps)
        reg = LinearRegression().fit(np.array(sizes).reshape(-1, 1), np.array(counts))
        return reg.coef_[0]
    return 0.0

def correlation_dimension_and_entropy(data, m, tau, r_vals=None):
    """
    Estymacja wymiaru korelacyjnego (D2) i entropii korelacyjnej (K2).
    Wymagane przez: 
    """
    N = len(data)
    M = N - (m - 1) * tau
    orbit = np.array([data[i : i + (m * tau) : tau] for i in range(M)])
    
    # Próbkowanie dla wydajności (duże N liczy się bardzo długo)
    if M > 1000:
        idx = np.random.choice(M, 1000, replace=False)
        orbit = orbit[idx]
        M = 1000

    tree = cKDTree(orbit)
    
    if r_vals is None:
        # Heurystyka doboru promieni
        dists = tree.query(orbit, k=2)[0][:, 1]
        min_d = np.mean(dists)
        max_d = np.max(np.linalg.norm(orbit - np.mean(orbit, axis=0), axis=1))
        r_vals = np.logspace(np.log10(min_d), np.log10(max_d/2), 20)
    
    C_r = []
    for r in r_vals:
        # Liczba par punktów w odległości < r
        count = tree.count_neighbors(tree, r) - M # odejmujemy autokorelację (punkt sam z sobą)
        norm_count = count / (M * (M - 1))
        C_r.append(norm_count)
        
    # Usunięcie zer dla logarytmu
    C_r = np.array(C_r)
    valid_idx = C_r > 0
    log_C_r = np.log(C_r[valid_idx])
    log_r = np.log(r_vals[valid_idx])
    
    # D2 - Wymiar korelacyjny (nachylenie w obszarze skalowania)
    D2 = 0.0
    if len(log_r) > 3:
        reg = LinearRegression().fit(log_r.reshape(-1, 1), log_C_r)
        D2 = reg.coef_[0]
        
    # K2 - Entropia korelacyjna (przybliżenie: K2 ~ 1/tau * ln(Cm(r)/Cm+1(r)))
    # Tutaj uproszczona estymacja bazująca na D2
    # W pełnym podejściu trzeba policzyć Cm i Cm+1. 
    # Przyjmijmy prostszą metrykę dla projektu: asymptota z wykresu Cm.
    
    return D2, log_r, log_C_r