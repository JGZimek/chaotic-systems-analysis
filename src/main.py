import warnings
warnings.filterwarnings("ignore") # Czysta konsola

import numpy as np
import matplotlib.pyplot as plt
from src.systems import generate_lorenz, generate_sprott_k, sprott_k_rhs, caputo_fractional_ode_solver
from src.signals import get_random_signal, get_periodic_signal
from src.plotting import plot_time_series_multi, plot_delay_analysis, plot_embedding_3d
from src.embedding import (
    estimate_delay_acf,
    estimate_delay_mi_histogram,
    estimate_delay_mi_kde, 
    estimate_embedding_dim_corrint,
    create_delay_embedding
)
from src.hurst_analysis import analyze_hurst
from src.chaos_metrics import largest_lyapunov_exponent, box_counting_dimension, correlation_dimension_and_entropy
from src.bifurcation import analyze_bifurcation_fractional, plot_bifurcation

# ============ CONFIGURATION ============
N = 10000        
T_MAX = 100.0
T = np.linspace(0, T_MAX, N)
DT = T[1] - T[0] 
y0 = np.array([0.1, 0.1, 0.1]) 

signals_config = {
    'random': 'Random Signal',
    'periodic': 'Periodic Signal (f=2.0)',
    'lorenz': 'Lorenz System',
    'sprott_k': 'Sprott K System',
    'sprott_k_frac_095': 'Sprott K Frac (α=0.95)',
    'sprott_k_frac_065': 'Sprott K Frac (α=0.65)',
}

def generate_signal(signal_type):
    if signal_type == 'random':
        return get_random_signal(N, seed=42)
    elif signal_type == 'periodic':
        return get_periodic_signal(N, freq=2.0, phase=0.0)
    elif signal_type == 'lorenz':
        traj = generate_lorenz((T[0], T[-1]), y0, T)
        return traj[:, 0]
    elif signal_type == 'sprott_k':
        traj = generate_sprott_k((T[0], T[-1]), y0, T)
        return traj[:, 0]
    elif signal_type == 'sprott_k_frac_095':
        traj = caputo_fractional_ode_solver(0.95, lambda t, y: sprott_k_rhs(t, y, a=0.3), y0, T)
        return traj[:, 0]
    elif signal_type == 'sprott_k_frac_065':
        traj = caputo_fractional_ode_solver(0.65, lambda t, y: sprott_k_rhs(t, y, a=0.3), y0, T)
        return traj[:, 0]

def analyze_signal(signal_type, signal):
    print(f"\n{'='*70}")
    print(f"ANALYSIS: {signals_config[signal_type]}")
    print(f"{'='*70}")
    
    if np.any(np.isnan(signal)):
        print("ERROR: Signal contains NaN values.")
        return None

    # 1. Wizualizacja
    plot_time_series_multi(signals=[signal], t=T, labels=["Signal"],
                           title=signals_config[signal_type],
                           save_path=f"data/timeseries_{signal_type}.png")
    
    # [cite_start]2. Delay (ACF, MI Hist) [cite: 21]
    delay_acf = estimate_delay_acf(signal, max_lag=200)
    delay_mi_hist = estimate_delay_mi_histogram(signal, max_lag=200)
    
    # KDE jest wolne, używamy go tylko jeśli trzeba, lub zakładamy proxy
    # Dla uproszczenia w main przyjmijmy średnią z ACF i MI Hist dla szybkości
    delay_mi_kde = delay_mi_hist # Proxy dla szybkości obliczeń w mainie
    
    tau = int(np.mean([delay_acf, delay_mi_hist]))
    tau = max(1, tau)
    print(f"Delay (τ) estimated: {tau} (ACF:{delay_acf}, MI:{delay_mi_hist})")
    
    plot_delay_analysis(signal, delay_acf, delay_mi_hist, delay_mi_kde,
                        save_path=f"data/delay_{signal_type}.png")

    # [cite_start]3. Embedding Dim (Nasycenie całki korelacyjnej) [cite: 20]
    # Usunięto korektę ekspercką - przyjmujemy wynik algorytmu
    dE = estimate_embedding_dim_corrint(signal, tau, max_dim=8)
    print(f"Embedding Dimension (dE) estimated: {dE}")

    # [cite_start]4. Reconstruction Plot [cite: 19]
    # Do wizualizacji wymuszamy 3 wymiary, aby wykres był czytelny
    embedding = create_delay_embedding(signal, 3, tau)
    plot_embedding_3d(embedding,
                      title=f"{signals_config[signal_type]} (τ={tau}, dE={dE})",
                      save_path=f"data/embed3d_{signal_type}.png")

    # [cite_start]5. Hurst [cite: 22]
    hurst_res = analyze_hurst(signal)
    print(f"Hurst Exponent: {hurst_res['h']:.3f}")

    # [cite_start]6. Lapunow [cite: 23]
    # Uwaga: Nawet jeśli dE=2, do Lapunowa bezpieczniej brać min. 3 dla układów chaosu
    lle = largest_lyapunov_exponent(signal, m=max(dE, 3), tau=tau, dt=DT)
    print(f"Largest Lyapunov Exponent (LLE): {lle:.4f}")

    # [cite_start]7. Fraktale [cite: 24-27]
    d_box = box_counting_dimension(signal, m=max(dE, 3), tau=tau)
    d_corr, _, _ = correlation_dimension_and_entropy(signal, m=max(dE, 3), tau=tau)
    
    print(f"Fractal Dimensions -> Box: {d_box:.3f}, Correlation (D2): {d_corr:.3f}")

    return {
        'name': signals_config[signal_type],
        'tau': tau,
        'dE': dE,
        'H': hurst_res['h'],
        'LLE': lle,
        'D_box': d_box,
        'D2': d_corr
    }

# ============ MAIN ============
if __name__ == "__main__":
    print("STARTING PROJECT ANALYSIS...")
    results = []
    
    # A. Analiza
    for signal_type in signals_config.keys():
        try:
            signal = generate_signal(signal_type)
            res = analyze_signal(signal_type, signal)
            if res: results.append(res)
        except Exception as e:
            print(f"ERROR {signal_type}: {e}")

    # [cite_start]B. Bifurkacja [cite: 28]
    print(f"\n{'='*70}\nBIFURCATION ANALYSIS\n{'='*70}")
    try:
        alphas = np.linspace(0.8, 1.0, 30) 
        T_bif = np.linspace(0, 100, 2000)
        bif_data = analyze_bifurcation_fractional(alphas, y0, T_bif)
        plot_bifurcation(bif_data, save_path="data/bifurcation_sprott.png")
    except Exception as e:
        print(f"Error bifur: {e}")

    # C. Tabela
    if results:
        print("\n" + "="*100)
        print(f"{'Signal':<25} {'τ':>5} {'dE':>4} {'Hurst':>6} {'LLE':>8} {'D_box':>6} {'D2':>6}")
        print("="*100)
        for r in results:
            print(f"{r['name']:<25} {r['tau']:>5} {r['dE']:>4} {r['H']:>6.3f} {r['LLE']:>8.4f} {r['D_box']:>6.3f} {r['D2']:>6.3f}")
        print("="*100 + "\n")