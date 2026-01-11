import numpy as np
from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression

def correlation_integral_curve(data, m, tau, r_vals, n_samples=2000):
    """Liczy całkę korelacyjną C(r)."""
    N = len(data)
    M = N - (m - 1) * tau
    orbit = np.array([data[i : i + (m * tau) : tau] for i in range(M)])
    
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
        count = tree.count_neighbors(tree, r) - M_actual
        norm_count = count / (M_actual * (M_actual - 1)) if M_actual > 1 else 0
        C_r.append(norm_count)
    return np.array(C_r)

def _fit_linear_region(log_r, log_Cr, tolerance=0.15):
    """Dopasowuje linię tylko w obszarze skalowania (ignoruje szum i nasycenie)."""
    if len(log_r) < 5: return 0.0
    
    # Obliczamy lokalne nachylenia (gradient)
    slopes = np.gradient(log_Cr, log_r)
    median_slope = np.median(slopes)
    
    # Wybieramy punkty, gdzie nachylenie jest stabilne
    valid_mask = np.abs(slopes - median_slope) < (tolerance * median_slope + 0.15)
    
    if np.sum(valid_mask) < 4:
        # Fallback: środek zakresu
        start, end = int(len(log_r)*0.2), int(len(log_r)*0.8)
        valid_mask = slice(start, end)
    
    log_r_valid = log_r[valid_mask]
    log_Cr_valid = log_Cr[valid_mask]
    
    if len(log_r_valid) < 2: return 0.0
    
    reg = LinearRegression().fit(log_r_valid.reshape(-1, 1), log_Cr_valid)
    return reg.coef_[0]

def correlation_dimension_and_entropy(data, m, tau, r_vals=None):
    """Metoda Grassbergera-Procaccii."""
    if r_vals is None:
        std_data = np.std(data)
        # Zakres r powinien pokrywać skalę atraktora
        r_vals = np.logspace(np.log10(std_data * 0.05), np.log10(std_data * 2.0), 20)
    
    C_m = correlation_integral_curve(data, m, tau, r_vals)
    C_m1 = correlation_integral_curve(data, m+1, tau, r_vals)
    
    valid = (C_m > 0) & (C_m1 > 0)
    if np.sum(valid) < 5:
        return 0.0, 0.0, np.log(r_vals), np.zeros_like(r_vals)

    log_r = np.log(r_vals[valid])
    log_Cm = np.log(C_m[valid])
    log_Cm1 = np.log(C_m1[valid])
    
    # D2: Dopasowanie w obszarze skalowania
    D2 = _fit_linear_region(log_r, log_Cm)
    
    # K2: Średnia odległość
    differences = log_Cm - log_Cm1
    # Ucinamy krańce dla stabilności K2
    trim = int(len(differences) * 0.2)
    if len(differences) > 2*trim:
        K2_est = np.mean(differences[trim:-trim])
    else:
        K2_est = np.mean(differences)
        
    return D2, max(0.0, K2_est), log_r, log_Cm

def largest_lyapunov_exponent(data, m, tau, dt=0.01, k=5):
    """Metoda Rosensteina."""
    N = len(data)
    M = N - (m - 1) * tau
    orbit = np.array([data[i : i + (m * tau) : tau] for i in range(M)])
    tree = cKDTree(orbit)
    dists, separate_idxs = tree.query(orbit, k=k+1) 
    nearest_idxs = separate_idxs[:, 1] 
    
    max_iter = min(50, M - 1) 
    divergence = []
    
    for i in range(max_iter):
        dist_sum = 0; count = 0
        step = 10 if M > 5000 else 1
        for j in range(0, M - i, step):
            idx_neigh = nearest_idxs[j]
            if idx_neigh + i < M:
                d = np.linalg.norm(orbit[j+i] - orbit[idx_neigh+i])
                if d > 1e-10:
                    dist_sum += np.log(d)
                    count += 1
        if count > 0: divergence.append(dist_sum / count)
            
    time_steps = np.arange(len(divergence)) * dt
    if len(divergence) > 5:
        # Fit tylko do początkowej (wykładniczej) fazy
        reg = LinearRegression().fit(time_steps[:15].reshape(-1, 1), np.array(divergence[:15]))
        return reg.coef_[0]
    return 0.0

def box_counting_dimension(data, m, tau, bins_range=(2, 50)):
    """Sparse Box Counting."""
    N = len(data); M = N - (m - 1) * tau
    orbit = np.array([data[i : i + (m * tau) : tau] for i in range(M)])
    orbit_min = orbit.min(axis=0); orbit_max = orbit.max(axis=0)
    orbit_norm = (orbit - orbit_min) / (orbit_max - orbit_min + 1e-9)
    
    counts, sizes = [], []
    bins_list = np.unique(np.logspace(np.log10(2), np.log10(bins_range[1]), 10).astype(int))
    
    for bins in bins_list:
        box_indices = np.floor(orbit_norm * bins).astype(int)
        occupied_boxes = set(map(tuple, box_indices))
        if len(occupied_boxes) > 0:
            counts.append(np.log(len(occupied_boxes)))
            sizes.append(np.log(bins)) 
            
    if len(counts) > 2:
        reg = LinearRegression().fit(np.array(sizes).reshape(-1, 1), np.array(counts))
        return reg.coef_[0]
    return 0.0