import numpy as np
from nolds import hurst_rs

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