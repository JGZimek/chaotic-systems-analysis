import numpy as np
import matplotlib.pyplot as plt
from src.systems import sprott_k_rhs, caputo_fractional_ode_solver

def analyze_bifurcation_fractional(alphas, y0, T, system_rhs='sprott_k'):
    """
    Generuje dane do diagramu bifurkacyjnego względem rzędu pochodnej alfa.
    Zbiera lokalne maksima sygnału po ustabilizowaniu.
    """
    bifurcation_points = []
    
    print(f"Computing Bifurcation Diagram for {len(alphas)} alpha values...")
    print(f"Simulation parameters: T_max={T[-1]}, Points={len(T)}")
    
    for i, alpha in enumerate(alphas):
        try:
            if i % 5 == 0:
                print(f"  Processing alpha: {alpha:.3f}...")

            # Rozwiązywanie układu ułamkowego
            traj = caputo_fractional_ode_solver(alpha, lambda t, y: sprott_k_rhs(t, y, a=0.3), y0, T)
            
            # Walidacja wyniku
            if traj is None or traj.ndim != 2:
                continue
            if np.any(np.isnan(traj)) or np.any(np.isinf(traj)):
                print(f"    -> Unstable at alpha={alpha:.3f}")
                continue

            # Odrzucamy stany przejściowe (ostatnie 40% sygnału uznajemy za stan ustalony)
            # W układach ułamkowych stany przejściowe są bardzo długie.
            keep_ratio = 0.4
            start_idx = int(len(traj) * (1 - keep_ratio))
            signal = traj[start_idx:, 0] # Analizujemy zmienną x
            
            # Znajdowanie lokalnych maksimów (metoda Poincaré section)
            # Punkt jest maksimum, jeśli jest większy od sąsiadów
            local_maxima = []
            for j in range(1, len(signal)-1):
                if signal[j-1] < signal[j] and signal[j] > signal[j+1]:
                    local_maxima.append(signal[j])
            
            # Jeśli nie znaleziono maksimów (np. stały punkt), dodajemy ostatnią wartość
            if not local_maxima:
                local_maxima.append(signal[-1])

            # Zapisujemy pary (alpha, wartość_maksimum)
            for val in local_maxima:
                bifurcation_points.append((alpha, val))
                
        except Exception as e:
            print(f"Error at alpha={alpha}: {e}")
            continue
            
    return np.array(bifurcation_points)

def plot_bifurcation(bif_data, save_path="data/bifurcation.png"):
    if len(bif_data) == 0:
        print("Warning: No bifurcation data generated.")
        return

    plt.figure(figsize=(10, 6))
    # Używamy bardzo małych punktów (s=1) i przezroczystości, aby zobaczyć gęstość
    plt.scatter(bif_data[:, 0], bif_data[:, 1], s=1.5, c='black', alpha=0.5)
    
    plt.title(r"Bifurcation Diagram: Sprott K vs Fractional Order $\alpha$")
    plt.xlabel(r"Fractional Order $\alpha$")
    plt.ylabel(r"Local Maxima of $x(t)$")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Bifurcation plot saved to {save_path}")