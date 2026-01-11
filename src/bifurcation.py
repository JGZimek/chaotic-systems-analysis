import numpy as np
import matplotlib.pyplot as plt
from src.systems import sprott_k_rhs, gl_fractional_ode_solver

# ==========================================
# ANALIZA BIFURKACYJNA (Diagramy Feigenbauma)
# ==========================================

def analyze_bifurcation_fractional(alphas, y0, T, system_rhs='sprott_k'):
    """
    Generuje dane do diagramu bifurkacyjnego w zależności od rzędu pochodnej alpha.
    
    Idea:
        Diagram bifurkacyjny pokazuje, jak zmieniają się stany ustalone układu 
        (punkty stałe, cykle graniczne, chaos) w funkcji parametru sterującego (tu: alpha).
        
    Metoda:
        Dla każdej wartości alpha:
        1. Symulujemy układ przez długi czas.
        2. Odrzucamy początkową część (stan nieustalony/transient), aby system wszedł na atraktor.
        3. Zbieramy punkty przecięcia (w tym przypadku: lokalne maksima sygnału x(t)).
           - 1 punkt na wykresie -> okres 1 (stabilny cykl).
           - 2 punkty -> podwojenie okresu.
           - Chmura punktów -> Chaos.
    """
    bifurcation_points = []
    print(f"Computing Bifurcation Diagram for {len(alphas)} alpha values...")
    
    # Pętla po kolejnych wartościach parametru alpha (np. od 0.8 do 1.0)
    for i, alpha in enumerate(alphas):
        try:
            if i % 10 == 0: print(f"  Processing alpha: {alpha:.3f}...")
            
            # 1. Symulacja numeryczna metodą Grünwalda-Letnikova
            # Używamy mniejszej pamięci (1000) dla szybkości, ponieważ do bifurkacji
            # potrzebujemy tylko ogólnego kształtu atraktora, a nie idealnej precyzji.
            traj = gl_fractional_ode_solver(
                alpha, 
                lambda t, y: sprott_k_rhs(t, y, a=0.3), 
                y0, T, memory_length=1000
            )
            
            # Zabezpieczenie przed ucieczką do nieskończoności (NaN)
            if np.any(np.isnan(traj)): continue
            
            # 2. Odrzucenie stanu nieustalonego (Transient)
            # Systemy chaotyczne potrzebują czasu, by osiąść na swoim "dziwnym atraktorze".
            # Odrzucamy pierwszą połowę symulacji.
            transient = int(len(traj) * 0.5)
            signal = traj[transient:, 0] # Bierzemy tylko zmienną 'x'
            
            # 3. Znalezienie lokalnych maksimów (szczytów sygnału)
            # To prosty sposób na stworzenie "przekroju Poincaré" dla sygnału ciągłego.
            local_maxima = []
            for j in range(1, len(signal)-1):
                # Warunek na maksimum lokalne: wartość jest większa od sąsiadów
                if signal[j-1] < signal[j] and signal[j] > signal[j+1]:
                    local_maxima.append(signal[j])
            
            # Jeśli system zbiegł do punktu stałego i nie ma oscylacji, 
            # bierzemy po prostu ostatnią wartość.
            if not local_maxima: local_maxima.append(signal[-1])
            
            # Zapisujemy pary (parametr, wartość) do wykresu
            for val in local_maxima:
                bifurcation_points.append((alpha, val))
                
        except: continue
            
    return np.array(bifurcation_points)

def plot_bifurcation(bif_data, save_path="data/bifurcation.png"):
    """
    Rysuje diagram bifurkacyjny na podstawie obliczonych punktów.
    """
    if len(bif_data) == 0: return
    
    plt.figure(figsize=(10, 6))
    
    # Rysujemy punkty jako bardzo małe kropki (s=0.5), aby zobaczyć gęstość w obszarach chaosu.
    # Oś X: Rząd pochodnej (alpha).
    # Oś Y: Lokalne maksima zmiennej x.
    plt.scatter(bif_data[:, 0], bif_data[:, 1], s=0.5, c='black', alpha=0.6)
    
    plt.title("Bifurcation Diagram: Sprott K vs Fractional Order (alpha)")
    plt.xlabel(r"Fractional Order $\alpha$")
    plt.ylabel("Local Maxima of x(t)")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Bifurcation plot saved to {save_path}")
    
    # plt.show() # Opcjonalne wyświetlanie
    plt.close()