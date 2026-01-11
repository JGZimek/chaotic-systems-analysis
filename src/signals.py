import numpy as np
from typing import Optional, Any

def get_random_signal(length: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generuje sygnał losowy (szum biały) o rozkładzie jednostajnym w zakresie [0,1] 
    o zadanej długości.
    
    Argumenty:
        length (int): Liczba próbek sygnału do wygenerowania.
        seed (int, opcjonalnie): Ziarno dla generatora liczb losowych. 
                                 Pozwala na powtarzalność wyników.
    """

    # Inicjalizacja generatora liczb losowych biblioteki NumPy.
    # Jeśli podano 'seed', generator zawsze zwróci tę samą sekwencję liczb 
    # (przydatne do debugowania i powtarzalności eksperymentów).
    rng = np.random.default_rng(seed)
    
    # Generowanie tablicy numpy o długości 'length'.
    # Funkcja uniform(0, 1) losuje liczby zmiennoprzecinkowe z przedziału od 0 do 1.
    return rng.uniform(0, 1, length)

def get_periodic_signal(length: int, freq: float = 1.0, phase: float = 0.0) -> np.ndarray:
    """
    Generuje deterministyczny sygnał okresowy będący sumą sin(f*t) + cos(f*t) 
    o zadanej długości. Służy jako sygnał referencyjny (porządek).
    
    Argumenty:
        length (int): Liczba próbek.
        freq (float): Częstotliwość sygnału (mnożnik czasu).
        phase (float): Przesunięcie fazowe (start sygnału).
    """

    # Tworzenie wektora czasu 't'.
    # Funkcja linspace generuje 'length' punktów równomiernie rozmieszczonych 
    # w przedziale od 0 do 2*pi. 
    # Dzięki temu niezależnie od liczby próbek, pokrywamy ten sam zakres "kątowy".
    t = np.linspace(0, 2 * np.pi, length)
    
    # Obliczenie wartości sygnału dla każdej chwili czasu 't'.
    # Sygnał jest sumą sinusa i cosinusa.
    # Wzór: y(t) = sin(omega * t + phi) + cos(omega * t + phi)
    return np.sin(freq * t + phase) + np.cos(freq * t + phase)