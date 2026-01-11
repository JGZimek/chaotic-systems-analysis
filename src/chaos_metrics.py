import numpy as np
from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression

# ==========================================
# 1. WYMIAR KORELACYJNY I ENTROPIA (Grassberger-Procaccia)
# ==========================================

def correlation_integral_curve(data, m, tau, r_vals, n_samples=2000):
    """
    Oblicza wartości całki korelacyjnej C(r) dla zadanych promieni r.
    
    Teoria:
        C(r) to prawdopodobieństwo, że dwa losowe punkty na atraktorze 
        są od siebie oddalone o mniej niż 'r'.
        Wzór: C(r) ~ 1/N^2 * sum(heaviside(r - ||xi - xj||))
    
    Argumenty:
        data: Sygnał wejściowy (szereg czasowy).
        m: Wymiar zanurzenia (embedding dimension).
        tau: Opóźnienie (delay).
        r_vals: Tablica promieni 'r', dla których liczymy C(r).
        n_samples: Liczba losowych punktów referencyjnych (przyspieszenie obliczeń).
                   Pełne liczenie to O(N^2), subsampling to O(n_samples * N).
    """
    N = len(data)
    # Efektywna liczba punktów w przestrzeni fazowej po rekonstrukcji
    M = N - (m - 1) * tau
    
    # Rekonstrukcja przestrzeni fazowej (tylko w pamięci, "w locie")
    # Tworzymy macierz o wymiarach (M, m)
    orbit = np.array([data[i : i + (m * tau) : tau] for i in range(M)])
    
    # Optymalizacja (Subsampling):
    # Jeśli punktów jest bardzo dużo (>2000), wybieramy losową podpróbkę,
    # aby cKDTree działało szybciej. Wynik jest statystycznie zbieżny.
    if len(orbit) > n_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(orbit), n_samples, replace=False)
        orbit = orbit[idx]
        M_actual = n_samples
    else:
        M_actual = len(orbit)

    # Budowa drzewa KD-Tree do szybkiego wyszukiwania sąsiadów w przestrzeni m-wymiarowej
    tree = cKDTree(orbit)
    C_r = []
    
    # Główna pętla po promieniach r
    for r in r_vals:
        # count_neighbors zwraca liczbę par w odległości < r.
        # Odejmujemy M_actual, aby nie liczyć par (i, i) - odległość punktu od samego siebie to 0.
        count = tree.count_neighbors(tree, r) - M_actual
        
        # Normalizacja wyniku do przedziału [0, 1]
        norm_count = count / (M_actual * (M_actual - 1)) if M_actual > 1 else 0
        C_r.append(norm_count)
        
    return np.array(C_r)

def _fit_linear_region(log_r, log_Cr, tolerance=0.15):
    """
    Inteligentne dopasowanie linii prostej do wykresu log-log (Scaling Region).
    
    Problem:
        Wykres log(C(r)) od log(r) nie jest prosty w całym zakresie.
        - Dla bardzo małych r: dominuje szum pomiarowy (nachylenie jest błędne).
        - Dla dużych r: nasycenie (cały atraktor mieści się w kuli r), nachylenie spada do 0.
    
    Rozwiązanie:
        Szukamy "płaskowyżu" na wykresie pochodnej (gradientu).
        Tam, gdzie pochodna jest stała, tam funkcja pierwotna jest liniowa.
    """
    if len(log_r) < 3: return 0.0
    
    # 1. Obliczamy lokalne nachylenie w każdym punkcie
    grads = np.gradient(log_Cr, log_r)
    
    # 2. Szukamy najdłuższego fragmentu, gdzie nachylenie jest stabilne
    # (różni się od średniej w tym oknie o mniej niż 'tolerance')
    best_slope = 0.0
    max_len = 0
    
    # Przesuwne okno po gradiencie
    for i in range(len(grads) - 2):
        for j in range(i + 2, len(grads)):
            window = grads[i:j]
            mean_grad = np.mean(window)
            # Sprawdzenie stabilności nachylenia w oknie
            if np.all(np.abs(window - mean_grad) < tolerance * mean_grad):
                if (j - i) > max_len:
                    max_len = j - i
                    best_slope = mean_grad
                    
    # Jeśli nie znaleziono stabilnego obszaru, zwracamy średnią z całości (fallback)
    return best_slope if max_len > 0 else np.mean(grads)

def correlation_dimension_and_entropy(signal, m, tau, n_r=20):
    """
    Oblicza Wymiar Korelacyjny (D2) oraz Entropię Korelacyjną (K2).
    
    Algorytm:
        1. Policz C(r) dla wymiaru 'm'.
        2. Nachylenie wykresu log-log to D2.
        3. Policz C(r) dla wymiaru 'm+1'.
        4. Entropia K2 wynika z pionowej odległości między krzywymi dla m i m+1.
    """
    # Zakres promieni r (skala logarytmiczna)
    std_dev = np.std(signal)
    r_vals = np.logspace(np.log10(std_dev * 0.01), np.log10(std_dev * 1.0), n_r)
    
    # Krok 1: Całka dla wymiaru m
    Cr_m = correlation_integral_curve(signal, m, tau, r_vals)
    
    # Filtrujemy zera (log(0) to -inf)
    valid_idx = (Cr_m > 0)
    if np.sum(valid_idx) < 5: return 0.0, 0.0, r_vals, Cr_m
    
    log_r = np.log(r_vals[valid_idx])
    log_Cr_m = np.log(Cr_m[valid_idx])
    
    # Krok 2: Wyznaczenie D2 (nachylenia) w obszarze skalowania
    d2 = _fit_linear_region(log_r, log_Cr_m)
    
    # Krok 3: Całka dla wymiaru m+1 (potrzebna do entropii K2)
    # K2 ~ (1/tau) * ln( Cm(r) / Cm+1(r) )
    try:
        Cr_m1 = correlation_integral_curve(signal, m + 1, tau, r_vals)
        # Bierzemy średnią odległość w obszarze skalowania
        valid_idx_k2 = (Cr_m > 0) & (Cr_m1 > 0)
        if np.sum(valid_idx_k2) > 5:
             # Uśredniamy różnicę logarytmów
             log_diff = np.log(Cr_m[valid_idx_k2]) - np.log(Cr_m1[valid_idx_k2])
             k2 = np.mean(log_diff) / (tau * (1.0)) # (tau * dt) jeśli uwzględniamy czas fizyczny
             k2 = max(0, k2)
        else:
            k2 = 0.0
    except:
        k2 = 0.0
        
    return d2, k2, r_vals, Cr_m

# ==========================================
# 2. NAJWIĘKSZY WYKŁADNIK LAPUNOWA (LLE)
# ==========================================

def largest_lyapunov_exponent(data, m, tau, dt=0.01, window=50):
    """
    Estymacja LLE metodą Rosensteina (1993).
    Jest to standardowa metoda dla małych zbiorów danych.
    
    Idea:
        Śledzimy pary najbliższych sąsiadów w przestrzeni fazowej.
        Jeśli system jest chaotyczny, odległość między nimi rośnie wykładniczo:
        d(t) ~ exp(lambda * t)
        Wykres log(d(t)) od t powinien być linią prostą o nachyleniu lambda.
    """
    N = len(data)
    M = N - (m - 1) * tau
    
    # Rekonstrukcja przestrzeni fazowej
    orbit = np.array([data[i : i + (m * tau) : tau] for i in range(M)])
    
    # Znalezienie najbliższego sąsiada dla KAŻDEGO punktu trajektorii
    tree = cKDTree(orbit)
    # k=2, bo pierwszym sąsiadem punktu jest on sam (odległość 0)
    dists, idxs = tree.query(orbit, k=2) 
    
    # idxs[:, 1] to indeksy najbliższych sąsiadów (nie licząc samego siebie)
    nearest_neighbors = idxs[:, 1]
    
    divergence = []
    
    # Śledzenie dywergencji przez 'window' kroków w przód
    for i in range(window):
        dist_sum = 0.0
        count = 0
        
        # Dla każdego punktu 'j' i jego sąsiada
        for j in range(M):
            idx_neigh = nearest_neighbors[j]
            
            # Sprawdzamy czy nie wychodzimy poza tablicę
            if j + i < M and idx_neigh + i < M:
                # Obliczamy odległość pary po czasie 'i'
                d = np.linalg.norm(orbit[j+i] - orbit[idx_neigh+i])
                
                # Unikamy log(0)
                if d > 1e-10:
                    dist_sum += np.log(d)
                    count += 1
                    
        # Średnia logarytmiczna dywergencja w kroku 'i'
        if count > 0: divergence.append(dist_sum / count)
            
    # Dopasowanie prostej do wykresu dywergencji
    time_steps = np.arange(len(divergence)) * dt
    
    # Ważne: Dopasowujemy tylko do początkowej fazy (np. pierwsze 15 kroków),
    # zanim trajektorie "rozbiegną się" na rozmiar całego atraktora (nasycenie).
    if len(divergence) > 5:
        reg = LinearRegression().fit(time_steps[:15].reshape(-1, 1), np.array(divergence[:15]))
        return reg.coef_[0] # To jest nasze LLE
        
    return 0.0

# ==========================================
# 3. WYMIAR PUDEŁKOWY (Box Counting)
# ==========================================

def box_counting_dimension(data, m, tau, bins_range=(2, 50)):
    """
    Oblicza wymiar fraktalny metodą pudełkową (Box-Counting).
    Jest to metoda geometryczna, prostsza niż korelacyjna, ale mniej dokładna dla chaosu.
    
    Wzór: N(epsilon) ~ epsilon^(-D)
    Gdzie N(epsilon) to liczba pudełek o boku epsilon potrzebnych do pokrycia atraktora.
    """
    N = len(data); M = N - (m - 1) * tau
    orbit = np.array([data[i : i + (m * tau) : tau] for i in range(M)])
    
    # Normalizacja atraktora do kostki jednostkowej [0,1]^m
    orbit_min = orbit.min(axis=0); orbit_max = orbit.max(axis=0)
    orbit_norm = (orbit - orbit_min) / (orbit_max - orbit_min + 1e-9)
    
    counts, sizes = [], []
    
    # Sprawdzamy różne rozmiary siatki (liczba pudełek na wymiar)
    # logspace generuje np. 4, 8, 16, 32... pudełek
    bins_list = np.unique(np.logspace(np.log10(2), np.log10(bins_range[1]), num=10).astype(int))
    
    for b in bins_list:
        # epsilon (rozmiar pudełka) jest odwrotnością liczby binów
        eps = 1.0 / b
        
        # Zamiana współrzędnych ciągłych na indeksy całkowite (do którego pudełka wpada punkt?)
        # HACK: Przesuwamy o epsilon/2, żeby uniknąć problemów na brzegach
        digitized = np.floor(orbit_norm * b).astype(int)
        
        # Zliczanie UNIKALNYCH pudełek, które zawierają przynajmniej jeden punkt
        # Każde pudełko reprezentujemy jako krotkę indeksów (ix, iy, iz...)
        active_boxes = len(np.unique(digitized, axis=0))
        
        counts.append(active_boxes)
        sizes.append(eps)
        
    # Dopasowanie liniowe na wykresie log(N) vs log(1/eps)
    if len(counts) > 2:
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        return -coeffs[0] # Wymiar to minus nachylenie (bo eps maleje)
        
    return 0.0