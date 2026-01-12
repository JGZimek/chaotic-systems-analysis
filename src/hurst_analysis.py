import numpy as np
from nolds import hurst_rs
import matplotlib.pyplot as plt

# ==========================================
# ANALIZA WYKŁADNIKA HURSTA (Metoda R/S)
# ==========================================

def estimate_hurst_rs(signal: np.ndarray) -> float:
    """
    Szacuje wykładnik Hursta (H) metodą Przeskalowanego Zakresu (Rescaled Range - R/S).
    
    Teoria:
        Wykładnik Hursta mierzy "pamięć" szeregu czasowego.
        Analiza R/S bada, jak zmienia się zakres wahań (Range) w stosunku do 
        odchylenia standardowego (Scale) wraz ze zmianą długości okna czasowego.
        Zależność ta przyjmuje postać: (R/S) ~ n^H.
    
    Zastosowanie biblioteki 'nolds':
        Używamy gotowej, zoptymalizowanej implementacji 'hurst_rs', która automatycznie
        dobiera podziały okien i dopasowuje linię regresji.
    """
    # Upewniamy się, że sygnał jest płaską tablicą 1D
    signal = np.asarray(signal).flatten()
    
    # Zabezpieczenie: Metoda R/S wymaga minimalnej liczby próbek do statystyki.
    if len(signal) < 100: return None
    
    try:
        # Obliczenie H. Funkcja zwraca pojedynczą liczbę float.
        return float(hurst_rs(signal))
    except:
        # W przypadku błędu numerycznego (np. sygnał stały, same zera)
        return None

def classify_hurst(h: float) -> str:
    """
    Interpretuje wartość wykładnika Hursta i klasyfikuje typ dynamiki sygnału.
    
    Przedziały interpretacji:
        0.0 < H < 0.45: "Mean-reverting" (Antypersystentny).
                        Sygnał często zmienia kierunek. Jeśli wzrósł, to prawdopodobnie spadnie.
                        Typowe dla procesów stabilizujących się.
                        
        0.45 <= H <= 0.55: "Random walk" (Błądzenie losowe / Szum).
                           Brak pamięci. Przeszłość nie wpływa na przyszłość (np. rzut monetą).
                           Dla idealnego szumu Browna H = 0.5.
                           
        0.55 < H < 1.0: "Trending" (Persystentny).
                        Sygnał ma "długą pamięć". Jeśli rośnie, to prawdopodobnie będzie dalej rósł.
                        Typowe dla trendów giełdowych, pogody i systemów chaotycznych.
    """
    if h is None: return "Unable" # Brak wyniku
    
    if h < 0.45: 
        return "Mean-reverting"
    elif h <= 0.55: 
        return "Random walk"
    else: 
        return "Trending"

def analyze_hurst(signal: np.ndarray) -> dict:
    """
    Funkcja pomocnicza (wrapper), która wykonuje pełną analizę Hursta
    i zwraca wyniki w formie słownika.
    
    Zwraca:
        {'h': float, 'trend': str}
    """
    # 1. Obliczenie wartości liczbowej
    h = estimate_hurst_rs(signal)
    
    # Jeśli obliczenia się nie udały, przyjmujemy bezpieczną wartość 0.5 (szum)
    h_val = h if h is not None else 0.5
    
    # 2. Zwrócenie wyniku wraz z interpretacją słowną
    return {'h': h_val, 'trend': classify_hurst(h_val)}

def plot_hurst_rs(signal: np.ndarray, title: str = "Hurst R/S Analysis", save_path: str = None):
    """
    Rysuje wykres log-log analizy R/S (Rescaled Range) dla wyznaczenia wykładnika Hursta.
    
    Co pokazuje wykres:
        - Oś X: logarytm długości podokna (n).
        - Oś Y: logarytm średniej wartości statystyki R/S.
        - Punkty: wyniki pomiarów dla różnych długości okien.
        - Linia przerywana: dopasowanie liniowe. Nachylenie tej linii to Wykładnik Hursta (H).
    """
    signal = np.asarray(signal).flatten()
    if len(signal) < 100:
        print("Signal too short for Hurst plot.")
        return

    try:
        # Wywołanie nolds z flagą debug_data=True aby otrzymać punkty wykresu.
        # Zwraca krotkę: (H, (n_vals, rs_vals, ...))
        h_val, data = hurst_rs(signal, debug_data=True)
        
        # === POPRAWKA ===
        # Zamiast 'n_vals, rs_vals = data' (co powoduje błąd), bierzemy indeksy.
        n_vals = data[0]
        rs_vals = data[1]
        
        # Konwersja na skalę logarytmiczną (dla wykresu log-log)
        log_n = np.log10(n_vals)
        log_rs = np.log10(rs_vals)
        
        # Ponowne dopasowanie prostej y = ax + b dla celów wizualizacji
        # (nachylenie 'poly[0]' powinno być identyczne jak 'h_val')
        poly = np.polyfit(log_n, log_rs, 1)
        fit_fn = np.poly1d(poly)
        
        # Rysowanie wykresu
        plt.figure(figsize=(8, 6))
        
        # 1. Punkty pomiarowe
        plt.scatter(log_n, log_rs, c='blue', alpha=0.6, label='R/S Data Points')
        
        # 2. Linia regresji
        plt.plot(log_n, fit_fn(log_n), 'r--', linewidth=2, label=f'Fit (H={h_val:.3f})')
        
        plt.title(title)
        plt.xlabel("log10(n) [Window Size]")
        plt.ylabel("log10(R/S) [Rescaled Range]")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            # print(f"Hurst plot saved to {save_path}")
        
        plt.close()
        
    except Exception as e:
        print(f"Could not plot Hurst R/S: {e}")