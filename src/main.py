import numpy as np
from src.systems import generate_lorenz, generate_sprott_k, sprott_k_rhs, caputo_fractional_ode_solver
from src.signals import get_random_signal, get_periodic_signal
from src.plotting import plot_time_series_multi, plot_delay_analysis, plot_embedding_2d, plot_embedding_3d, plot_trajectory_3d
from src.embedding import (
    estimate_delay_acf,
    estimate_delay_mi_histogram,
    estimate_embedding_dim_corrint,
    create_delay_embedding
)

# ============ CONFIGURATION ============
N = 10000
T_MAX = 100.0
T = np.linspace(0, T_MAX, N)
y0 = np.array([0.5, 0.5, 0.5])

# Signals to analyze
signals_config = {
    'random': {'name': 'Random Signal', 'color': 'blue'},
    'periodic': {'name': 'Periodic Signal (f=2.0)', 'color': 'green'},
    'lorenz': {'name': 'Lorenz System', 'color': 'purple'},
    'sprott_k': {'name': 'Sprott K System', 'color': 'orange'},
    'sprott_k_frac_095': {'name': 'Sprott K Fractional (α=0.95)', 'color': 'red'},
    'sprott_k_frac_085': {'name': 'Sprott K Fractional (α=0.85)', 'color': 'brown'},
}

def generate_signal(signal_type):
    """Generate signal based on type."""
    if signal_type == 'random':
        return get_random_signal(N, seed=42)
    elif signal_type == 'periodic':
        return get_periodic_signal(N, freq=2.0, phase=0.0)
    elif signal_type == 'lorenz':
        traj = generate_lorenz((T[0], T[-1]), y0, T)
        return traj[:, 0]  # Return X component
    elif signal_type == 'sprott_k':
        traj = generate_sprott_k((T[0], T[-1]), y0, T)
        return traj[:, 0]  # Return X component
    elif signal_type == 'sprott_k_frac_095':
        traj = caputo_fractional_ode_solver(0.95, lambda t, y: sprott_k_rhs(t, y, a=0.3), y0, T)
        return traj[:, 0]  # Return X component
    elif signal_type == 'sprott_k_frac_085':
        traj = caputo_fractional_ode_solver(0.85, lambda t, y: sprott_k_rhs(t, y, a=0.3), y0, T)
        return traj[:, 0]  # Return X component

def analyze_signal(signal_type, signal):
    """Run complete embedding analysis on signal."""
    print(f"\n{'='*70}")
    print(f"ANALYZING: {signals_config[signal_type]['name']}")
    print(f"{'='*70}")
    
    # Plot time series
    print(f"\n[PLOT] Time series...")
    plot_time_series_multi(
        signals=[signal],
        t=T,
        labels=["Signal"],
        title=f"{signals_config[signal_type]['name']} - Time Series",
        ylabel="Amplitude",
        xlabel="Time",
        save_path=f"data/timeseries_{signal_type}.png"
    )
    
    # TEST 1: ACF
    print(f"[TEST 1] Estimating delay (ACF)...")
    delay_acf = estimate_delay_acf(signal, max_lag=200)
    print(f"  ✓ τ_ACF = {delay_acf}")
    
    # TEST 2: MI Histogram
    print(f"[TEST 2] Estimating delay (MI Histogram)...")
    delay_mi_hist = estimate_delay_mi_histogram(signal, max_lag=200)
    print(f"  ✓ τ_MI = {delay_mi_hist}")
    
    # TEST 3: Consensus delay
    delay_recommended = int(np.mean([delay_acf, delay_mi_hist]))
    print(f"[TEST 3] Consensus delay: τ = {delay_recommended}")
    
    # TEST 4: Embedding
    embedding_dim = 3
    embedding = create_delay_embedding(signal, embedding_dim, delay_recommended)
    print(f"[TEST 4] Embedding created: shape {embedding.shape}")
    
    # TEST 5: Plot delay analysis
    print(f"[PLOT] Delay analysis...")
    plot_delay_analysis(
        signal=signal,
        delay_acf=delay_acf,
        delay_mi_hist=delay_mi_hist,
        delay_mi_kde=delay_acf,
        save_path=f"data/delay_analysis_{signal_type}.png"
    )
    
    # TEST 6: Plot 2D embedding
    print(f"[PLOT] 2D embedding...")
    plot_embedding_2d(
        embedding,
        title=f"{signals_config[signal_type]['name']} - 2D Embedding (τ={delay_recommended})",
        save_path=f"data/embedding_2d_{signal_type}.png"
    )
    
    # TEST 7: Plot 3D embedding
    print(f"[PLOT] 3D embedding...")
    plot_embedding_3d(
        embedding,
        title=f"{signals_config[signal_type]['name']} - 3D Embedding (τ={delay_recommended})",
        save_path=f"data/embedding_3d_{signal_type}.png"
    )
    
    # TEST 8: Embedding dimension
    print(f"[TEST 8] Estimating embedding dimension...")
    embedding_dim_estimated = estimate_embedding_dim_corrint(
        signal, 
        delay=delay_recommended, 
        max_dim=10,
        saturation_threshold=0.01
    )
    print(f"  ✓ dE = {embedding_dim_estimated}")
    
    return {
        'signal_type': signal_type,
        'name': signals_config[signal_type]['name'],
        'delay_acf': delay_acf,
        'delay_mi': delay_mi_hist,
        'delay_recommended': delay_recommended,
        'embedding_dim': embedding_dim_estimated,
    }

# ============ MAIN EXECUTION ============
print("\n" + "="*70)
print("PHASE SPACE RECONSTRUCTION - COMPREHENSIVE ANALYSIS")
print("="*70)

results = []

# Generate and analyze all signals
for signal_type in signals_config.keys():
    try:
        signal = generate_signal(signal_type)
        result = analyze_signal(signal_type, signal)
        results.append(result)
    except Exception as e:
        print(f"✗ ERROR in {signal_type}: {e}")

# ============ SUMMARY TABLE ============
print("\n" + "="*70)
print("SUMMARY - ALL SIGNALS")
print("="*70)
print(f"{'Signal Type':<30} {'τ_ACF':>8} {'τ_MI':>8} {'τ_cons':>8} {'dE':>5}")
print("-"*70)

for r in results:
    print(f"{r['name']:<30} {r['delay_acf']:>8} {r['delay_mi']:>8} {r['delay_recommended']:>8} {r['embedding_dim']:>5}")

print("="*70 + "\n")
