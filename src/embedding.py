import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from statsmodels.tsa.stattools import acf
from scipy.stats import gaussian_kde
from typing import Optional
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

def calculate_mutual_information_curve(signal: np.ndarray, max_lag: int = 100, bins: int = 16) -> np.ndarray:
    """
    Pomocnicza funkcja obliczająca krzywą Wzajemnej Informacji I(T) dla opóźnień T.
    
    Zwraca:
        mi_values (np.ndarray): Tablica wartości MI.
    """
    signal = np.asarray(signal).flatten()
    mi_vals = []
    
    # === POPRAWKA STABILNOŚCI (przeniesiona z oryginalnej funkcji) ===
    # Wcześniej liczba binów była liczona dynamicznie (wychodziło ok. 44), co przy N=10000
    # dawało zbyt mało punktów na jeden kubełek histogramu 2D (~5 pkt). Powodowało to szum 
    # i fałszywe minima (np. tau=4 dla okresowego).
    #
    # Ustawienie bins=16 daje średnio ~40 punktów na kubełek.
    # To wygładza histogram i pozwala znaleźć prawdziwe minimum globalne.
    
    for lag in range(max_lag + 1):
        if lag == 0:
            x = signal
            y = signal
        else:
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
        # Używamy log2 aby wynik był w bitach (zgodnie z literaturą)
        px_py = px[:, None] * py[None, :]
        
        # Unikamy logarytmu z zera i dzielenia przez zero
        nzs = pxy > 0
        mi = np.sum(pxy[nzs] * np.log2(pxy[nzs] / px_py[nzs]))
        mi_vals.append(mi)
        
    return np.array(mi_vals)

def find_optimal_delay_from_curve(mi_values: np.ndarray) -> int:
    """
    Znajduje optymalne opóźnienie T_opt na podstawie krzywej MI.
    Realizuje kryteria Frasera-Swinneya oraz kryterium spadku wartości (4/5).
    """
    # 1. Algorytm Frasera-Swinneya: Pierwsze LOKALNE MINIMUM.
    # Sprawdzamy, czy obecna wartość jest mniejsza od sąsiadów.
    # Zaczynamy od 1, bo 0 to autoinformacja (maksimum).
    for i in range(1, len(mi_values) - 1):
        if mi_values[i] < mi_values[i-1] and mi_values[i] < mi_values[i+1]:
            return i
            
    # 2. Kryterium alternatywne (jeśli brak wyraźnego minimum):
    # Wybór Td, dla którego MI spada do ok. 4/5 wartości początkowej I(0).
    target_val = 0.8 * mi_values[0]
    below_thresh = np.where(mi_values < target_val)[0]
    
    if len(below_thresh) > 0:
        return below_thresh[0]
        
    # Fallback: jeśli nic nie znaleziono, zwracamy minimum globalne (lub 1)
    return np.argmin(mi_values) if np.argmin(mi_values) > 0 else 1

def estimate_delay_mi_histogram(signal: np.ndarray, max_lag: int = 100) -> int:
    """
    Szacuje opóźnienie metodą Wzajemnej Informacji (Mutual Information).
    Jest to podejście Frasera-Swinneya: szukamy pierwszego lokalnego minimum MI.
    
    Metoda ta jest lepsza dla chaosu, ponieważ wykrywa nieliniowe zależności między próbkami.
    """
    # Obliczamy krzywą MI
    mi_curve = calculate_mutual_information_curve(signal, max_lag, bins=16)
    # Wyznaczamy optymalne opóźnienie
    return find_optimal_delay_from_curve(mi_curve)

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

# ==========================================
# 4. FUNKCJE WIZUALIZACYJNE (DODATKOWE)
# ==========================================

def plot_time_series_multi(signals, t, labels=None, title="", ylabel="Value", xlabel="Time", save_path=None):
    plt.figure(figsize=(10, 4))
    for idx, sig in enumerate(signals):
        lbl = labels[idx] if labels and idx < len(labels) else None
        plt.plot(t, sig, linewidth=0.8, label=lbl)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if labels: plt.legend()
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300)
    plt.close()
        
def plot_delay_analysis(signal, delay_acf, delay_mi_hist, delay_mi_kde=None, save_path=None):
    """
    Rysuje wykres Autokorelacji z zaznaczonymi punktami estymacji opóźnienia.
    Uwaga: Dla pełnej analizy MI użyj plot_paper_style_analysis.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    lags = np.arange(1, 101) # Rysujemy pierwsze 100 dla czytelności
    
    # ACF
    try:
        acf_vals = acf(signal, nlags=100, fft=True)
        ax.plot(lags, acf_vals[1:], label=f'ACF (τ={delay_acf})', color='blue', alpha=0.7)
        ax.axvline(x=delay_acf, color='blue', linestyle='--')
    except: pass

    ax.axvline(x=delay_mi_hist, color='red', linestyle='--', label=f'MI Hist (τ={delay_mi_hist})')
    
    ax.set_title("Delay Estimation Points")
    ax.set_xlabel("Lag")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path: plt.savefig(save_path, dpi=300)
    plt.close()

def plot_embedding_3d(embedding, title="3D Embedding", save_path=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = np.arange(len(embedding))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], 
                        c=colors, cmap='viridis', s=1, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel('x(t)')
    ax.set_ylabel('x(t+τ)')
    ax.set_zlabel('x(t+2τ)')
    plt.colorbar(scatter, ax=ax, label='Time')
    if save_path: plt.savefig(save_path, dpi=300)
    plt.close()

def plot_paper_style_analysis(signal, system_name="System", max_lag=100, save_path=None):
    """
    Generuje wykresy I(T) oraz ACF w stylu Rys. 5-8 z literatury.
    Rysuje obok siebie funkcję autokorelacji i krzywą wzajemnej informacji.
    """
    # 1. Obliczenia
    # ACF
    acf_vals = acf(signal, nlags=max_lag, fft=True)
    zero_crossings = np.where(np.diff(np.sign(acf_vals)))[0]
    tau_acf = zero_crossings[0] if len(zero_crossings) > 0 else 0

    # MI (Wzajemna Informacja) - obliczamy pełną krzywą
    mi_curve = calculate_mutual_information_curve(signal, max_lag=max_lag)
    tau_mi = find_optimal_delay_from_curve(mi_curve)

    # 2. Rysowanie
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Wykres A: Autokorelacja (ACF)
    axes[0].plot(range(len(acf_vals)), acf_vals, color='blue', linewidth=1.5)
    axes[0].axhline(0, color='black', linewidth=0.8, linestyle='-')
    axes[0].axvline(tau_acf, color='red', linestyle='--', alpha=0.7, label=f'First Zero: T={tau_acf}')
    axes[0].set_title(f"Autocorrelation Function: {system_name}")
    axes[0].set_xlabel("Time Delay (T)")
    axes[0].set_ylabel("ACF")
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend()

    # Wykres B: Wzajemna Informacja (MI) - Styl literaturowy
    axes[1].plot(range(len(mi_curve)), mi_curve, color='black', linewidth=1.5)
    axes[1].axvline(tau_mi, color='red', linestyle='--', label=f'$T_{{opt}}={tau_mi}$')
    
    # Opcjonalnie można dodać linię dla kryterium 4/5 I(0)
    # axes[1].axhline(0.8 * mi_curve[0], color='green', linestyle=':', label='4/5 I(0)')

    axes[1].set_title(f"Mutual Information $I(T)$: {system_name}")
    axes[1].set_xlabel("Time Delay (T)")
    axes[1].set_ylabel("I(T) [bits]")
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Zapisano wykres: {save_path}")
    plt.close()
    
    return tau_mi