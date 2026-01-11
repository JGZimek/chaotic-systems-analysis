import numpy as np
from statsmodels.tsa.stattools import acf
from scipy.stats import gaussian_kde
from src.chaos_metrics import correlation_dimension_and_entropy

# ==========================================
# 1. ESTYMACJA OPÓŹNIENIA CZASOWEGO (TAU)
# ==========================================

def estimate_delay_acf(signal: np.ndarray, max_lag: int = 100) -> int:
    """
    Szacuje opóźnienie czasowe (tau) przy użyciu Funkcji Autokorelacji (ACF).
    
    Metoda: 'First Zero Crossing'.
    Szukamy pierwszego przesunięcia, dla którego korelacja sygnału z samym sobą spada do zera.
    Uwaga: Metoda ta zakłada liniową niezależność. Dla systemów chaotycznych (nieliniowych)
    często daje wyniki zbyt wysokie (np. Lorenz tau=50 zamiast 17).
    """
    try:
        # Obliczenie funkcji autokorelacji (z wykorzystaniem szybkiej transformaty Fouriera - FFT)
        acf_vals = acf(signal, nlags=max_lag, fft=True)
        
        # Znalezienie miejsc, gdzie znak korelacji się zmienia (przejście przez zero)
        zero_crossings = np.where(np.diff(np.sign(acf_vals)))[0]
        
        # Zwracamy indeks pierwszego przejścia przez zero
        if len(zero_crossings) > 0: return int(zero_crossings[0])
    except: pass
    
    # Fallback: jeśli nie znaleziono zera, zwracamy 1/4 badanego zakresu
    return max(1, max_lag // 4) 

def estimate_delay_mi_histogram(signal: np.ndarray, max_lag: int = 100) -> int:
    """
    Szacuje opóźnienie metodą Wzajemnej Informacji (Mutual Information).
    Jest to podejście Frasera-Swinneya: szukamy pierwszego lokalnego minimum MI.
    
    Metoda ta jest lepsza dla chaosu, ponieważ wykrywa nieliniowe zależności między próbkami.
    """
    signal = np.asarray(signal).flatten()
    
    # === POPRAWKA STABILNOŚCI ===
    # Wcześniej liczba binów była liczona dynamicznie (wychodziło ok. 44), co przy N=10000
    # dawało zbyt mało punktów na jeden kubełek histogramu 2D (~5 pkt). Powodowało to szum 
    # i fałszywe minima (np. tau=4 dla okresowego).
    #
    # Ustawienie bins=16 daje średnio ~40 punktów na kubełek.
    # To wygładza histogram i pozwala znaleźć prawdziwe minimum globalne.
    bins = 16
    
    mi_vals = []
    
    for lag in range(1, max_lag + 1):
        # Tworzymy dwie serie: oryginalną (x) i przesuniętą (y)
        x = signal[:-lag]
        y = signal[lag:]
        
        # Obliczenie histogramu 2D (rozkład łączny prawdopodobieństwa p(x,y))
        hist_2d, _, _ = np.histogram2d(x, y, bins=bins)
        
        # Normalizacja histogramu do prawdopodobieństw
        total = np.sum(hist_2d)
        pxy = hist_2d / total
        
        # Rozkłady brzegowe p(x) i p(y)
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        
        # Wzór na Mutual Information: I(X;Y) = sum( p(x,y) * log( p(x,y) / (p(x)*p(y)) ) )
        px_py = px[:, None] * py[None, :]
        
        # Unikamy logarytmu z zera i dzielenia przez zero
        nzs = pxy > 0
        mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
        mi_vals.append(mi)
        
        # Algorytm Frasera-Swinneya: Zwróć pierwsze LOKALNE MINIMUM.
        # Sprawdzamy, czy obecna wartość jest mniejsza od sąsiadów.
        if len(mi_vals) > 2:
            # Warunek "dołka": [wysoko, nisko, wyżej]
            if mi_vals[-2] < mi_vals[-3] and mi_vals[-2] < mi_vals[-1]:
                return lag - 1 
                
    # Jeśli nie znaleziono minimum lokalnego, zwróć minimum globalne z całego zakresu
    return np.argmin(mi_vals) + 1

# ==========================================
# 2. ESTYMACJA WYMIARU ZANURZENIA (dE)
# ==========================================

def estimate_embedding_dim_corrint(signal: np.ndarray, delay: int, max_dim: int = 8, saturation_threshold: float = 0.1) -> int:
    """
    Szacuje minimalny wymiar zanurzenia (dE) metodą Nasycenia Całki Korelacyjnej (G-P).
    Badamy, przy jakim wymiarze 'm' wymiar fraktalny D2 przestaje rosnąć.
    """
    signal = np.asarray(signal).flatten()
    # Ograniczenie próbek dla przyspieszenia obliczeń (G-P jest O(N^2))
    calc_signal = signal[:min(len(signal), 5000)]
    corr_dims = []
    
    # 1. Pętla po kolejnych wymiarach (np. od 1 do 8)
    for dim in range(1, max_dim + 1):
        try:
            # Obliczamy D2 (Correlation Dimension) dla danej przestrzeni
            d2, _, _, _ = correlation_dimension_and_entropy(calc_signal, dim, delay)
            corr_dims.append(d2)
        except: corr_dims.append(0)
    
    corr_dims = np.array(corr_dims)
    
    # 2. Analiza przyrostów D2
    for i in range(len(corr_dims) - 1):
        m = i + 1              # Aktualnie badany wymiar przestrzeni
        d2_curr = corr_dims[i] # Wymiar fraktalny w tej przestrzeni
        d2_next = corr_dims[i+1]
        
        # Obliczenie względnej zmiany (czy D2 jeszcze rośnie?)
        rel_change = (d2_next - d2_curr) / d2_curr if d2_curr > 0 else 1.0
        
        # Warunek A: Nasycenie. 
        # Jeśli zmiana jest mniejsza niż próg (np. 10%), uznajemy że D2 się ustaliło.
        is_saturated = rel_change < saturation_threshold
        
        # Warunek B: Topologiczny margines bezpieczeństwa (Space Filling Check).
        # === KLUCZOWE DLA SPROTTA K ===
        # Jeśli D2 atraktora wynosi 1.81, to matematycznie mieści się w 2D (bo 1.81 < 2).
        # Jednak w 2D jest "ciasno" i trajektorie mogą się przecinać (fałszywi sąsiedzi).
        # Wymuszamy warunek: d2_curr < (m - 0.8).
        # Dla m=2: 1.81 < 1.2 -> FAŁSZ (wymuszamy pójście do m=3).
        # Dla m=3: 1.81 < 2.2 -> PRAWDA (jest wystarczająco dużo miejsca).
        fits_in_space = d2_curr < (m - 0.8)
        
        if is_saturated and fits_in_space:
            return m
            
    return max_dim

# ==========================================
# 3. REKONSTRUKCJA ATRAKTORA (Time Delay Embedding)
# ==========================================

def create_delay_embedding(signal: np.ndarray, embedding_dim: int, delay: int) -> np.ndarray:
    """
    Tworzy macierz trajektorii w przestrzeni fazowej zgodnie z twierdzeniem Takensa.
    
    Wejście: [x(1), x(2), x(3), ...]
    Wyjście (dla dim=3, delay=tau):
    [ [x(1),       x(1+tau),       x(1+2tau)      ],
      [x(2),       x(2+tau),       x(2+2tau)      ],
      ... ]
    """
    signal = np.asarray(signal).flatten()
    n_samples = len(signal) - (embedding_dim - 1) * delay
    
    if n_samples <= 0: return np.zeros((1, embedding_dim))
    
    # Szybka wektoryzacja numpy (zamiast pętli for)
    # Tworzymy indeksy dla wszystkich kolumn naraz
    indices = np.arange(embedding_dim) * delay + np.arange(n_samples)[:, np.newaxis]
    
    # Zwracamy macierz punktów
    return signal[indices]