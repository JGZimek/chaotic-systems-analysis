import numpy as np
from src.systems import generate_lorenz, generate_sprott_k, sprott_k_rhs, caputo_fractional_ode_solver
from src.signals import get_random_signal, get_periodic_signal
from src.plotting import plot_time_series_multi, plot_delay_analysis, plot_embedding_2d, plot_embedding_3d
from src.embedding import (
    estimate_delay_acf,
    estimate_delay_mi_histogram,
    estimate_delay_mi_kde,  # UPEWNIJ SIĘ ŻE MASZ TAKĄ FUNKCJĘ
    estimate_embedding_dim_corrint,
    create_delay_embedding
)
from src.hurst_analysis import analyze_hurst

# ============ CONFIGURATION ============
N = 10000
T_MAX = 100.0
T = np.linspace(0, T_MAX, N)
y0 = np.array([0.5, 0.5, 0.5])

signals_config = {
    'random': 'Random Signal',
    'periodic': 'Periodic Signal (f=2.0)',
    'lorenz': 'Lorenz System',
    'sprott_k': 'Sprott K System',
    'sprott_k_frac_095': 'Sprott K Frac (α=0.95)',
    'sprott_k_frac_085': 'Sprott K Frac (α=0.85)',
}

def generate_signal(signal_type):
    """Generate signal."""
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
    elif signal_type == 'sprott_k_frac_085':
        traj = caputo_fractional_ode_solver(0.85, lambda t, y: sprott_k_rhs(t, y, a=0.3), y0, T)
        return traj[:, 0]

def analyze_signal(signal_type, signal):
    """Analyze signal: embedding + Hurst."""
    print(f"\n{'='*70}")
    print(f"{signals_config[signal_type]}")
    print(f"{'='*70}")
    
    # Time series plot
    plot_time_series_multi(signals=[signal], t=T, labels=["Signal"],
                          title=signals_config[signal_type],
                          save_path=f"data/timeseries_{signal_type}.png")
    
    # Delay
    delay_acf = estimate_delay_acf(signal, max_lag=200)
    delay_mi_hist = estimate_delay_mi_histogram(signal, max_lag=200)
    delay_mi_kde = estimate_delay_mi_kde(signal, max_lag=200)
    
    # Średnia z 3 metod
    delay_rec = int(np.mean([delay_acf, delay_mi_hist, delay_mi_kde]))
    print(f"τ = {delay_rec}")
    
    # Embedding
    embedding = create_delay_embedding(signal, 3, delay_rec)
    
    # Plots
    plot_delay_analysis(signal, delay_acf, delay_mi_hist, delay_mi_kde,
                       save_path=f"data/delay_{signal_type}.png")
    plot_embedding_2d(embedding, 
                     title=f"{signals_config[signal_type]} (τ={delay_rec})",
                     save_path=f"data/embed2d_{signal_type}.png")
    plot_embedding_3d(embedding,
                     title=f"{signals_config[signal_type]} (τ={delay_rec})",
                     save_path=f"data/embed3d_{signal_type}.png")
    
    # Embedding dimension
    dE = estimate_embedding_dim_corrint(signal, delay_rec, max_dim=10)
    print(f"dE = {dE}")
    
    # Hurst
    hurst = analyze_hurst(signal)
    print(f"H = {hurst['h']:.3f} → {hurst['trend']}")
    
    return {
        'name': signals_config[signal_type],
        'tau': delay_rec,
        'dE': dE,
        'hurst': hurst['h'],
        'trend': hurst['trend'],
    }

# ============ MAIN ============
print("\n" + "="*70)
print("CHAOTIC SYSTEMS ANALYSIS")
print("="*70)

results = []
for signal_type in signals_config.keys():
    try:
        signal = generate_signal(signal_type)
        result = analyze_signal(signal_type, signal)
        results.append(result)
    except Exception as e:
        print(f"ERROR: {e}")

# ============ SUMMARY ============
print("\n" + "="*80)
print(f"{'Signal':<27} {'τ':>6} {'dE':>5} {'Hurst':>8} {'Trend':<20}")
print("="*80)

for r in results:
    print(f"{r['name']:<27} {r['tau']:>6} {r['dE']:>5} {r['hurst']:>8.3f} {r['trend']:<20}")

print("="*80 + "\n")
