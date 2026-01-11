import numpy as np
from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression

# ==========================================
# 1. METODA GRASSBERGERA-PROCACCII (Wymiar korelacyjny D2 i Entropia K2)
# ==========================================

def correlation_integral_curve(data, m, tau, r_vals, n_samples=2000):
    """
    Liczy całkę korelacyjną C(r) dla zadanej listy promieni r.
    C(r) to prawdopodobieństwo znalezienia dwóch punktów na atraktorze w odległości < r.
    """
    N = len(data)
    # Efektywna liczba punktów w przestrzeni fazowej po rekonstrukcji
    M = N - (m - 1) * tau
    
    # Rekonstrukcja przestrzeni fazowej metodą opóźnień (Time Delay Embedding)
    # Tworzymy macierz trajektorii o wymiarach (M, m)
    orbit = np.array([data[i : i + (m * tau) : tau] for i in range(M)])
    
    # Optymalizacja (Subsampling):
    # Obliczanie odległości dla wszystkich par to O(N^2). Dla dużych zbiorów
    # losujemy podzbiór punktów (n_samples), co przyspiesza obliczenia.
    if len(orbit) > n_samples:
        rng = np.random.default_rng(42) # Stałe ziarno dla powtarzalności
        idx = rng.choice(len(orbit), n_samples, replace=False)
        orbit = orbit[idx]
        M_actual = n_samples
    else:
        M_actual = len(orbit)

    # Budowa drzewa KD-Tree do szybkiego wyszukiwania sąsiadów w przestrzeni wielowymiarowej
    tree = cKDTree(orbit)
    C_r = []
    
    # Pętla po promieniach r
    for r in r_vals:
        # count_neighbors zwraca liczbę par punktów odległych o mniej niż r.
        # Odejmujemy M_actual, aby nie liczyć par punktu z samym sobą (odległość 0).
        count = tree.count_neighbors(tree, r) - M_actual
        
        # Normalizacja wyniku (prawdopodobieństwo)
        norm_count = count / (M_actual * (M_actual - 1)) if M_actual > 1 else 0
        C_r.append(norm_count)
        
    return np.array(C_r)

def _fit_linear_region(log_r, log_Cr, tolerance=0.15):
    """
    Dopasowuje linię prostą do wykresu log-log, ale tylko w 'obszarze skalowania' (Scaling Region).
    Ignoruje szum (małe r) i nasycenie (duże r).
    """
    if len(log_r) < 5: return 0.0
    
    # Obliczamy lokalne nachylenia (gradient) w każdym punkcie
    slopes = np.gradient(log_Cr, log_r)
    # Wyznaczamy medianę nachylenia jako punkt odniesienia
    median_slope = np.median(slopes)
    
    # Wybieramy punkty, gdzie nachylenie jest stabilne (bliskie medianie z zadaną tolerancją)
    # Pozwala to odrzucić 'ogony' wykresu, które zakłóciłyby regresję.
    valid_mask = np.abs(slopes - median_slope) < (tolerance * median_slope + 0.15)
    
    # Fallback: Jeśli wykres jest zbyt poszarpany, bierzemy środkowe 60% zakresu
    if np.sum(valid_mask) < 4:
        start, end = int(len(log_r)*0.2), int(len(log_r)*0.8)
        valid_mask = slice(start, end)
    
    log_r_valid = log_r[valid_mask]
    log_Cr_valid = log_Cr[valid_mask]
    
    if len(log_r_valid) < 2: return 0.0
    
    # Dopasowanie regresji liniowej do wybranego fragmentu
    reg = LinearRegression().fit(log_r_valid.reshape(-1, 1), log_Cr_valid)
    return reg.coef_[0] # Nachylenie to szukany wymiar D2

def correlation_dimension_and_entropy(data, m, tau, r_vals=None):
    """
    Główna funkcja metody Grassbergera-Procaccii.
    Zwraca:
      - D2: Wymiar korelacyjny (nachylenie wykresu całki)
      - K2: Entropię korelacyjną (odległość między krzywymi dla m i m+1)
    """
    # Dynamiczne generowanie zakresu promieni r (skala logarytmiczna)
    # Zakres zależy od odchylenia standardowego sygnału.
    if r_vals is None:
        std_data = np.std(data)
        r_vals = np.logspace(np.log10(std_data * 0.05), np.log10(std_data * 2.0), 20)
    
    # 1. Obliczenie krzywych całkowych dla wymiaru m oraz m+1
    C_m = correlation_integral_curve(data, m, tau, r_vals)
    C_m1 = correlation_integral_curve(data, m+1, tau, r_vals)
    
    # Filtrowanie zer (logarytm z zera to -inf)
    valid = (C_m > 0) & (C_m1 > 0)
    if np.sum(valid) < 5:
        return 0.0, 0.0, np.log(r_vals), np.zeros_like(r_vals)

    # Przejście do skali logarytmicznej
    log_r = np.log(r_vals[valid])
    log_Cm = np.log(C_m[valid])
    log_Cm1 = np.log(C_m1[valid])
    
    # D2: Wyznaczenie wymiaru fraktalnego (nachylenie prostej)
    D2 = _fit_linear_region(log_r, log_Cm)
    
    # K2: Estymacja entropii Kolmogorowa-Sinaia
    # K2 jest proporcjonalne do pionowej odległości między log(Cm) a log(Cm+1)
    differences = log_Cm - log_Cm1
    
    # Odrzucamy skrajne wartości dla stabilności wyniku
    trim = int(len(differences) * 0.2)
    if len(differences) > 2*trim:
        K2_est = np.mean(differences[trim:-trim])
    else:
        K2_est = np.mean(differences)
        
    # Zwracamy D2 i K2 (entropia nie może być ujemna, stąd max(0, ...))
    return D2, max(0.0, K2_est), log_r, log_Cm

# ==========================================
# 2. NAJWIĘKSZY WYKŁADNIK LAPUNOWA (Metoda Rosensteina)
# ==========================================

def largest_lyapunov_exponent(data, m, tau, dt=0.01, k=5):
    """
    Estymacja LLE metodą Rosensteina.
    Bada średnie tempo dywergencji (rozbiegania się) sąsiednich trajektorii.
    """
    N = len(data)
    M = N - (m - 1) * tau
    # Rekonstrukcja atraktora
    orbit = np.array([data[i : i + (m * tau) : tau] for i in range(M)])
    
    # Znalezienie najbliższych sąsiadów dla każdego punktu
    tree = cKDTree(orbit)
    # Szukamy k+1 sąsiadów (indeks 0 to sam punkt, indeks 1 to najbliższy sąsiad)
    dists, separate_idxs = tree.query(orbit, k=k+1) 
    nearest_idxs = separate_idxs[:, 1] 
    
    max_iter = min(50, M - 1) # Horyzont czasowy śledzenia dywergencji
    divergence = []
    
    # Pętla po krokach czasowych w przyszłość
    for i in range(max_iter):
        dist_sum = 0; count = 0
        # Optymalizacja: dla dużych zbiorów bierzemy co 10 punkt startowy
        step = 10 if M > 5000 else 1
        
        for j in range(0, M - i, step):
            idx_neigh = nearest_idxs[j] # Indeks sąsiada
            
            # Sprawdzamy czy punkty nadal mieszczą się w tablicy po czasie 'i'
            if idx_neigh + i < M:
                # Obliczamy odległość euklidesową po czasie 'i'
                d = np.linalg.norm(orbit[j+i] - orbit[idx_neigh+i])
                
                # Sumujemy logarytmy odległości (ln d ~ lambda * t)
                if d > 1e-10:
                    dist_sum += np.log(d)
                    count += 1
        if count > 0: divergence.append(dist_sum / count)
            
    # Tworzymy oś czasu
    time_steps = np.arange(len(divergence)) * dt
    if len(divergence) > 5:
        # Fitujemy prostą tylko do początkowej fazy wykładniczej (pierwsze 15 kroków)
        # Później trajektorie rozbiegają się na rozmiar całego atraktora (nasycenie)
        reg = LinearRegression().fit(time_steps[:15].reshape(-1, 1), np.array(divergence[:15]))
        return reg.coef_[0] # Nachylenie to LLE
    return 0.0

# ==========================================
# 3. WYMIAR PUDEŁKOWY (Box Counting)
# ==========================================

def box_counting_dimension(data, m, tau, bins_range=(2, 50)):
    """
    Metoda pudełkowa (Sparse Box Counting).
    Liczy ile pudełek o boku epsilon potrzeba do pokrycia atraktora.
    """
    N = len(data); M = N - (m - 1) * tau
    orbit = np.array([data[i : i + (m * tau) : tau] for i in range(M)])
    
    # Normalizacja danych do hipersześcianu [0, 1]^m
    orbit_min = orbit.min(axis=0); orbit_max = orbit.max(axis=0)
    orbit_norm = (orbit - orbit_min) / (orbit_max - orbit_min + 1e-9)
    
    counts, sizes = [], []
    # Sprawdzamy ~10 różnych rozmiarów siatki (liczby pudełek na wymiar)
    bins_list = np.unique(np.logspace(np.log10(2), np.log10(bins_range[1]), 10).astype(int))
    
    for bins in bins_list:
        # Zamiana współrzędnych na indeksy całkowite (do którego pudełka wpada punkt?)
        box_indices = np.floor(orbit_norm * bins).astype(int)
        
        # Używamy zbioru (set) krotek do zliczenia unikalnych zajętych pudełek
        # To oszczędza pamięć (tzw. sparse matrix approach)
        occupied_boxes = set(map(tuple, box_indices))
        
        if len(occupied_boxes) > 0:
            counts.append(np.log(len(occupied_boxes))) # log N(eps)
            sizes.append(np.log(bins))                 # log (1/eps)
            
    # Regresja liniowa: D_box to nachylenie prostej
    if len(counts) > 2:
        reg = LinearRegression().fit(np.array(sizes).reshape(-1, 1), np.array(counts))
        return reg.coef_[0]
    return 0.0