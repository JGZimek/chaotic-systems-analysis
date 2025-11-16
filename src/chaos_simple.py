import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
from scipy.special import gamma
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Fractional Caputo Derivative Solver
# ============================================================================

def caputo_fractional_ode_solver(alpha, f, y0, t):
    """
    Solves fractional ODE: D^alpha y = f(y, t) using simplified method
    for Caputo derivative of order alpha in (0, 1]
    """
    y0 = np.asarray(y0)
    n = len(t)
    d = len(y0)
    y = np.zeros((n, d))
    y[0] = y0
    
    for i in range(n - 1):
        dt = t[i + 1] - t[i]
        y_next = y[i] + (dt ** alpha) * f(y[i], t[i]) / gamma(alpha + 1)
        y[i + 1] = y_next
    
    return y

# ============================================================================
# Define Chaotic Systems
# ============================================================================

def sprott_k(y, t, a=0.3):
    """SPROTT K system"""
    x, y_var, z = y
    dx_dt = -z + x * y_var
    dy_dt = x - y_var
    dz_dt = x + a * z
    return [dx_dt, dy_dt, dz_dt]

def rossler(y, t, a=0.2, b=0.2, c=5.7):
    """Rössler system"""
    x, y_var, z = y
    dx_dt = -y_var - z
    dy_dt = x + a * y_var
    dz_dt = b + z * (x - c)
    return [dx_dt, dy_dt, dz_dt]

# ============================================================================
# Time parameters and initial conditions
# ============================================================================

t_end = 100.0
dt = 0.01
t = np.arange(0, t_end, dt)
y0 = np.array([0.5, 0.5, 0.5])

print("Generating solutions...")

# Generate solutions
sol_sprott_k_classical = odeint(sprott_k, y0, t)
print("[OK] SPROTT K (Classical)")

sol_sprott_k_frac = caputo_fractional_ode_solver(0.95, 
    lambda y, t: np.array(sprott_k(y, t)), y0, t)
print("[OK] SPROTT K (Fractional α=0.95)")

sol_rossler_classical = odeint(rossler, y0, t)
print("[OK] Rössler (Classical)")

sol_rossler_frac = caputo_fractional_ode_solver(0.95, 
    lambda y, t: np.array(rossler(y, t)), y0, t)
print("[OK] Rössler (Fractional α=0.95)")

# ============================================================================
# Create 4 Plots (2D and 3D side by side for each system)
# ============================================================================

# Plot 1: SPROTT K Classical
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot(sol_sprott_k_classical[:, 0], sol_sprott_k_classical[:, 1], 
         sol_sprott_k_classical[:, 2], linewidth=0.5, alpha=0.8, color='#1f77b4')
ax1.scatter(y0[0], y0[1], y0[2], color='green', s=100, marker='o', zorder=10)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('SPROTT K (Classical)\na = 0.3 - 3D View')
ax1.view_init(elev=20, azim=45)

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(sol_sprott_k_classical[:, 0], sol_sprott_k_classical[:, 1], 
         linewidth=0.5, alpha=0.8, color='#1f77b4')
ax2.scatter(y0[0], y0[1], color='green', s=100, marker='o', zorder=10, label='Start')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('SPROTT K (Classical)\na = 0.3 - 2D View (X-Y)')
ax2.grid(alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('01_sprott_k_classical.png', dpi=300, bbox_inches='tight')
print("\n[OK] Saved: 01_sprott_k_classical.png")
plt.close()

# Plot 2: SPROTT K Fractional
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot(sol_sprott_k_frac[:, 0], sol_sprott_k_frac[:, 1], 
         sol_sprott_k_frac[:, 2], linewidth=0.5, alpha=0.8, color='#ff7f0e')
ax1.scatter(y0[0], y0[1], y0[2], color='green', s=100, marker='o', zorder=10)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('SPROTT K (Fractional)\nα = 0.95, a = 0.3 - 3D View')
ax1.view_init(elev=20, azim=45)

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(sol_sprott_k_frac[:, 0], sol_sprott_k_frac[:, 1], 
         linewidth=0.5, alpha=0.8, color='#ff7f0e')
ax2.scatter(y0[0], y0[1], color='green', s=100, marker='o', zorder=10, label='Start')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('SPROTT K (Fractional)\nα = 0.95, a = 0.3 - 2D View (X-Y)')
ax2.grid(alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('02_sprott_k_fractional.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: 02_sprott_k_fractional.png")
plt.close()

# Plot 3: Rössler Classical
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot(sol_rossler_classical[:, 0], sol_rossler_classical[:, 1], 
         sol_rossler_classical[:, 2], linewidth=0.5, alpha=0.8, color='#2ca02c')
ax1.scatter(y0[0], y0[1], y0[2], color='green', s=100, marker='o', zorder=10)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Rössler (Classical)\na=0.2, b=0.2, c=5.7 - 3D View')
ax1.view_init(elev=20, azim=45)

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(sol_rossler_classical[:, 0], sol_rossler_classical[:, 1], 
         linewidth=0.5, alpha=0.8, color='#2ca02c')
ax2.scatter(y0[0], y0[1], color='green', s=100, marker='o', zorder=10, label='Start')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Rössler (Classical)\na=0.2, b=0.2, c=5.7 - 2D View (X-Y)')
ax2.grid(alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('03_rossler_classical.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: 03_rossler_classical.png")
plt.close()

# Plot 4: Rössler Fractional
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot(sol_rossler_frac[:, 0], sol_rossler_frac[:, 1], 
         sol_rossler_frac[:, 2], linewidth=0.5, alpha=0.8, color='#d62728')
ax1.scatter(y0[0], y0[1], y0[2], color='green', s=100, marker='o', zorder=10)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Rössler (Fractional)\nα = 0.95, a=0.2, b=0.2, c=5.7 - 3D View')
ax1.view_init(elev=20, azim=45)

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(sol_rossler_frac[:, 0], sol_rossler_frac[:, 1], 
         linewidth=0.5, alpha=0.8, color='#d62728')
ax2.scatter(y0[0], y0[1], color='green', s=100, marker='o', zorder=10, label='Start')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Rössler (Fractional)\nα = 0.95, a=0.2, b=0.2, c=5.7 - 2D View (X-Y)')
ax2.grid(alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('04_rossler_fractional.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: 04_rossler_fractional.png")
plt.close()

print("\n" + "="*60)
print("[OK] Generation complete!")
print("="*60)
print("\nGenerated files:")
print("  1. 01_sprott_k_classical.png")
print("  2. 02_sprott_k_fractional.png")
print("  3. 03_rossler_classical.png")
print("  4. 04_rossler_fractional.png")
