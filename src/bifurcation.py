import numpy as np
import matplotlib.pyplot as plt
from src.systems import sprott_k_rhs, caputo_fractional_ode_solver

def analyze_bifurcation_fractional(alphas, y0, T, system_rhs='sprott_k'):
    bifurcation_points = []
    print(f"Computing Bifurcation Diagram for {len(alphas)} alpha values...")
    
    for i, alpha in enumerate(alphas):
        try:
            if i % 10 == 0: print(f"  Processing alpha: {alpha:.3f}...")
            # Pamięć 1000 jest wystarczająca do wykrycia bifurkacji i szybka
            traj = caputo_fractional_ode_solver(alpha, lambda t, y: sprott_k_rhs(t, y, a=0.3), y0, T, memory_length=1000)
            
            if np.any(np.isnan(traj)): continue
            
            transient = int(len(traj) * 0.5)
            signal = traj[transient:, 0]
            
            local_maxima = []
            for j in range(1, len(signal)-1):
                if signal[j-1] < signal[j] and signal[j] > signal[j+1]:
                    local_maxima.append(signal[j])
            
            if not local_maxima: local_maxima.append(signal[-1])
            for val in local_maxima:
                bifurcation_points.append((alpha, val))
        except: continue
            
    return np.array(bifurcation_points)

def plot_bifurcation(bif_data, save_path="data/bifurcation.png"):
    if len(bif_data) == 0: return
    plt.figure(figsize=(10, 6))
    plt.scatter(bif_data[:, 0], bif_data[:, 1], s=1.0, c='black', alpha=0.5)
    plt.title(r"Bifurcation Diagram: Sprott K vs Fractional Order $\alpha$")
    plt.xlabel(r"Fractional Order $\alpha$")
    plt.ylabel(r"Local Maxima of $x(t)$")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()