import numpy as np
import matplotlib.pyplot as plt
from src.systems import sprott_k_rhs, caputo_fractional_ode_solver

def analyze_bifurcation_fractional(alphas, y0, T, system_rhs='sprott_k'):
    """
    Generuje diagram bifurkacyjny.
    Odporny na błędy numeryczne (overflow/NaN).
    """
    bifurcation_points = []
    
    print(f"Computing Bifurcation Diagram for {len(alphas)} alpha values...")
    
    for alpha in alphas:
        try:
            # Rozwiązywanie równania
            traj = caputo_fractional_ode_solver(alpha, lambda t, y: sprott_k_rhs(t, y, a=0.3), y0, T)
            
            # 1. Sprawdzenie poprawności wyników
            if traj is None or traj.ndim != 2:
                continue

            # 2. Sprawdzenie czy wystąpił wybuch (NaN w wynikach)
            if np.any(np.isnan(traj)) or np.any(np.isinf(traj)):
                # Można odkomentować do debugowania:
                # print(f"  [Skipped] Alpha {alpha:.2f} unstable")
                continue

            # Odrzucamy stany przejściowe (pierwsze 50%)
            transient = int(len(traj) * 0.5)
            signal = traj[transient:, 0]
            
            # Znajdowanie lokalnych maksimów
            local_maxima = []
            for i in range(1, len(signal)-1):
                if signal[i-1] < signal[i] and signal[i] > signal[i+1]:
                    local_maxima.append(signal[i])
            
            # Zapisujemy pary (alpha, max_val)
            for val in local_maxima:
                bifurcation_points.append((alpha, val))
                
        except Exception:
            # Ignorujemy błędy dla pojedynczych iteracji
            continue
            
    return np.array(bifurcation_points)

def plot_bifurcation(bif_data, save_path="data/bifurcation.png"):
    if len(bif_data) == 0:
        print("Warning: No bifurcation data generated (system unstable for chosen range).")
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(bif_data[:, 0], bif_data[:, 1], s=0.5, c='black', alpha=0.6)
    plt.title("Bifurcation Diagram: Sprott K vs Fractional Order ($\\alpha$)")
    plt.xlabel(r"Fractional Order $\alpha$")
    plt.ylabel("$x_{max}$")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Bifurcation plot saved to {save_path}")