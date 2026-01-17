import warnings
warnings.filterwarnings("ignore") 

import numpy as np
import matplotlib.pyplot as plt
import os

# Importy z modułów projektu
from src.systems import generate_lorenz, generate_sprott_k, sprott_k_rhs, gl_fractional_ode_solver
from src.signals import get_random_signal, get_periodic_signal
from src.plotting import plot_time_series_multi, plot_embedding_3d, plot_original_system_3d, plot_embedding_saturation # Added plot_embedding_saturation
from src.embedding import (
    estimate_delay_acf,
    estimate_delay_mi_histogram,
    estimate_embedding_dim_corrint,
    create_delay_embedding,
    plot_paper_style_analysis # ADDED THIS IMPORT
)
from src.hurst_analysis import analyze_hurst, plot_hurst_rs
from src.chaos_metrics import largest_lyapunov_exponent, box_counting_dimension, correlation_dimension_and_entropy
from src.bifurcation import analyze_bifurcation_fractional, plot_bifurcation

# ============ KONFIGURACJA ============
# Parametry symulacji
N = 10000        
T_MAX = 100.0
T = np.linspace(0, T_MAX, N)
DT = T[1] - T[0] 
y0 = np.array([0.1, 0.1, 0.1]) 

# Definicje sygnałów
signals_config = {
    'random': 'Random Signal',
    'periodic': 'Periodic Signal (f=2.0)',
    'lorenz': 'Lorenz System',
    'sprott_k': 'Sprott K System',
    'sprott_k_frac_095': 'Sprott K Frac (α=0.95)',
    'sprott_k_frac_065': 'Sprott K Frac (α=0.65)',
}

def generate_signal(signal_type):
    """Generuje odpowiedni sygnał na podstawie klucza konfiguracyjnego."""
    if signal_type == 'random':
        return get_random_signal(N, seed=42), None
    elif signal_type == 'periodic':
        return get_periodic_signal(N, freq=2.0, phase=0.0), None
    elif signal_type == 'lorenz':
        traj = generate_lorenz((T[0], T[-1]), y0, T)
        return traj[:, 0], traj  # Zwraca zarówno sygnał x(t) jak i pełną trajektorię
    elif signal_type == 'sprott_k':
        traj = generate_sprott_k((T[0], T[-1]), y0, T)
        return traj[:, 0], traj
    elif signal_type == 'sprott_k_frac_095':
        # Układ z lekkim tłumieniem (zachowany chaos)
        traj = gl_fractional_ode_solver(0.95, lambda t, y: sprott_k_rhs(t, y, a=0.3), y0, T)
        return traj[:, 0], traj
    elif signal_type == 'sprott_k_frac_065':
        # Układ z silnym tłumieniem (zbieżność do punktu stałego)
        traj = gl_fractional_ode_solver(0.65, lambda t, y: sprott_k_rhs(t, y, a=0.3), y0, T)
        return traj[:, 0], traj

# Cache for detailed trajectories (to avoid recomputation)
_detailed_trajectories_cache = {}

def generate_detailed_trajectory(signal_type):
    """
    Generuje trajektorię z większą liczbą punktów specjalnie dla wizualizacji.
    Używa 5x więcej punktów niż standardowa analiza dla lepszej szczegółowości.
    Wynik jest cachowany, aby uniknąć wielokrotnych obliczeń.
    """
    global _detailed_trajectories_cache
    
    # Sprawdź czy jest już w cache
    if signal_type in _detailed_trajectories_cache:
        return _detailed_trajectories_cache[signal_type]
    
    N_detailed = 50000  # 5x więcej punktów
    T_detailed = np.linspace(0, T_MAX, N_detailed)
    
    detailed_traj = None
    
    if signal_type == 'lorenz':
        detailed_traj = generate_lorenz((T_detailed[0], T_detailed[-1]), y0, T_detailed)
    elif signal_type == 'sprott_k':
        detailed_traj = generate_sprott_k((T_detailed[0], T_detailed[-1]), y0, T_detailed)
    elif signal_type == 'sprott_k_frac_095':
        print(f"  Generating detailed trajectory for {signal_type} (50k points, may take a moment)...")
        detailed_traj = gl_fractional_ode_solver(0.95, lambda t, y: sprott_k_rhs(t, y, a=0.3), y0, T_detailed)
    elif signal_type == 'sprott_k_frac_065':
        print(f"  Generating detailed trajectory for {signal_type} (50k points, may take a moment)...")
        detailed_traj = gl_fractional_ode_solver(0.65, lambda t, y: sprott_k_rhs(t, y, a=0.3), y0, T_detailed)
    
    # Cachuj wynik
    if detailed_traj is not None:
        _detailed_trajectories_cache[signal_type] = detailed_traj
    
    return detailed_traj

def analyze_signal(signal_type, signal, trajectory=None):
    label = signals_config[signal_type]
    print(f"\n{'='*70}")
    print(f"ANALYSIS: {label}")
    print(f"{'='*70}")
    
    if np.any(np.isnan(signal)):
        print("ERROR: Signal contains NaN values.")
        return None

    # WAŻNE: Normalizacja sygnału (Z-score)
    # Jest kluczowa dla algorytmów opartych na odległości (LLE, D2), 
    # aby uniknąć problemów ze skalą.
    signal = (signal - np.mean(signal)) / np.std(signal)

    # 0. Wykres oryginalnego systemu w przestrzeni fazowej 3D (jeśli dostępny)
    if trajectory is not None and trajectory.shape[1] == 3:
        # Dla systemów chaotycznych generuj szczegółową trajektorię z więcej punktami
        if signal_type in ['lorenz', 'sprott_k', 'sprott_k_frac_095', 'sprott_k_frac_065']:
            detailed_traj = generate_detailed_trajectory(signal_type)
            if detailed_traj is not None and len(detailed_traj) > 0:
                plot_original_system_3d(detailed_traj, 
                                        title=f"{label} - Original Phase Space",
                                        save_path=f"data/original_3d_{signal_type}.png")
                print(f"Generated original 3D phase space plot with {len(detailed_traj)} points.")
            else:
                print(f"Warning: Could not generate detailed trajectory for {signal_type}")
        else:
            # Dla sygnałów nie-chaotycznych użyj zwykłej trajektorii
            plot_original_system_3d(trajectory, 
                                    title=f"{label} - Original Phase Space",
                                    save_path=f"data/original_3d_{signal_type}.png")
            print(f"Generated original 3D phase space plot with {len(trajectory)} points.")

    # 1. Podgląd sygnału w czasie
    plot_time_series_multi(signals=[signal], t=T, labels=["Signal"],
                           title=label,
                           save_path=f"data/timeseries_{signal_type}.png")
    
    # 2. Estymacja opóźnienia (Tau)
    # Obliczamy tau z ACF dla informacji (do printa)
    delay_acf = estimate_delay_acf(signal, max_lag=200)
    
    # Używamy nowej funkcji do wizualizacji i obliczenia tau z MI
    # Zwraca ona tau obliczone metodą MI (Fraser-Swinney lub 4/5)
    tau = plot_paper_style_analysis(signal, system_name=label, max_lag=200, 
                                    save_path=f"data/delay_{signal_type}.png")
    
    # Zabezpieczenie minimalnego tau
    tau = max(1, tau)
    
    # Możemy też wywołać starą funkcję wrapper, jeśli chcemy tylko wartość do logów,
    # ale plot_paper_style_analysis już to zrobiła.
    # Dla spójności logów:
    print(f"Delay (τ) estimated: {tau} (ACF:{delay_acf}, MI:{tau})")

    # 3. Estymacja wymiaru zanurzenia (dE) - Metoda nasycenia
    dE = estimate_embedding_dim_corrint(signal, tau, max_dim=8)
    print(f"Embedding Dimension (dE) estimated: {dE}")

    # 3.5 Wykres nasycenia wymiaru zanurzenia (pokazuje log(C(r)) vs log(r) dla różnych dE)
    # Analogicznie do Fig. 9 z artykułu "Compound method of time series classification"
    plot_embedding_saturation(signal, tau, 
                             title=f"{label} - Embedding Dimension Analysis",
                             save_path=f"data/embedding_saturation_{signal_type}.png",
                             max_dim=10)

    # 4. Wizualizacja atraktora w 3D
    embedding = create_delay_embedding(signal, 3, tau)
    plot_embedding_3d(embedding,
                      title=f"{label} (τ={tau}, dE={dE})",
                      save_path=f"data/embed3d_{signal_type}.png")

    # 5. Analiza trendów (Hurst)
    hurst_res = analyze_hurst(signal)
    print(f"Hurst Exponent: {hurst_res['h']:.3f}")
    
    # Rysowanie wykresu R/S dla Hursta
    plot_hurst_rs(signal, title=f"Hurst Analysis: {label}", 
                  save_path=f"data/hurst_{signal_type}.png")

    # 6. Niezmienniki chaosu
    # Używamy wymiaru dE (lub min. 3 dla bezpieczeństwa algorytmów 3D)
    m_calc = max(dE, 3)
    
    # LLE (Rosenstein)
    lle = largest_lyapunov_exponent(signal, m=m_calc, tau=tau, dt=DT)
    print(f"Largest Lyapunov Exponent (LLE): {lle:.4f}")

    # Wymiary fraktalne (Pudełkowy i Korelacyjny) i Entropia
    d_box = box_counting_dimension(signal, m=m_calc, tau=tau)
    d_corr, k2, _, _ = correlation_dimension_and_entropy(signal, m=m_calc, tau=tau)
    
    print(f"Fractal Dimensions -> Box: {d_box:.3f}, Correlation (D2): {d_corr:.3f}")
    print(f"Correlation Entropy (K2): {k2:.3f}")

    return {
        'name': label,
        'tau': tau,
        'dE': dE,
        'H': hurst_res['h'],
        'LLE': lle,
        'D_box': d_box,
        'D2': d_corr,
        'K2': k2
    }

if __name__ == "__main__":
    # Tworzenie katalogu na wyniki
    if not os.path.exists("data"):
        os.makedirs("data")

    print("STARTING PROJECT ANALYSIS...")
    results = []
    
    # A. Główna pętla analizy sygnałów
    for signal_type in signals_config.keys():
        try:
            signal, trajectory = generate_signal(signal_type)
            res = analyze_signal(signal_type, signal, trajectory)
            if res: results.append(res)
        except Exception as e:
            print(f"ERROR {signal_type}: {e}")
            import traceback
            traceback.print_exc()

    # B. Analiza Bifurkacyjna
    print(f"\n{'='*70}\nBIFURCATION ANALYSIS\n{'='*70}")
    try:
        alphas = np.linspace(0.8, 1.0, 40) 
        T_bif = np.linspace(0, 200, 20000) 
        
        bif_data = analyze_bifurcation_fractional(alphas, y0, T_bif)
        plot_bifurcation(bif_data, save_path="data/bifurcation_sprott.png")
    except Exception as e:
        print(f"Error bifur: {e}")
        import traceback
        traceback.print_exc()

    # C. Wyświetlenie tabeli końcowej
    if results:
        print("\n" + "="*110)
        print(f"{'Signal':<25} {'τ':>5} {'dE':>4} {'Hurst':>6} {'LLE':>8} {'D_box':>6} {'D2':>6} {'K2':>6}")
        print("="*110)
        for r in results:
            print(f"{r['name']:<25} {r['tau']:>5} {r['dE']:>4} {r['H']:>6.3f} {r['LLE']:>8.4f} {r['D_box']:>6.3f} {r['D2']:>6.3f} {r['K2']:>6.3f}")
        print("="*110 + "\n")