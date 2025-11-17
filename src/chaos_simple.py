import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
from scipy.special import gamma
import warnings
warnings.filterwarnings('ignore')

def caputo_fractional_ode_solver(alpha, f, y0, t):
    """Solves fractional ODE: D^alpha y = f(y, t) using simplified method for Caputo derivative of order alpha in (0, 1]"""
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

def sprott_k(y, t, a=0.3):
    x, y_var, z = y
    dx_dt = -z + x * y_var
    dy_dt = x - y_var
    dz_dt = x + a * z
    return [dx_dt, dy_dt, dz_dt]

def rossler(y, t, a=0.2, b=0.2, c=5.7):
    x, y_var, z = y
    dx_dt = -y_var - z
    dy_dt = x + a * y_var
    dz_dt = b + z * (x - c)
    return [dx_dt, dy_dt, dz_dt]

t_end = 100.0
dt = 0.01
t = np.arange(0, t_end, dt)
y0 = np.array([0.5, 0.5, 0.5])

print("Generating solutions...")

sol_sprott_k_classical = odeint(sprott_k, y0, t)
print("[OK] SPROTT K (Classical)")

sol_sprott_k_frac = caputo_fractional_ode_solver(0.85,
    lambda y, t: np.array(sprott_k(y, t)), y0, t)
print("[OK] SPROTT K (Fractional α=0.85)")

sol_rossler_classical = odeint(rossler, y0, t)
print("[OK] Rössler (Classical)")

sol_rossler_frac = caputo_fractional_ode_solver(0.85,
    lambda y, t: np.array(rossler(y, t)), y0, t)
print("[OK] Rössler (Fractional α=0.85)")

# SPROTT K Classical - 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(sol_sprott_k_classical[:, 0],
        sol_sprott_k_classical[:, 1],
        sol_sprott_k_classical[:, 2],
        linewidth=0.5, alpha=0.8, color='#1f77b4')
ax.scatter(y0[0], y0[1], y0[2], color='green', s=100, marker='o', zorder=10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('SPROTT K (Classical) 3D')
plt.tight_layout()
plt.savefig('01_sprott_k_classical_3d.png', dpi=300)
plt.close()

# SPROTT K Classical - 2D
fig = plt.figure(figsize=(8, 6))
plt.plot(sol_sprott_k_classical[:, 0], sol_sprott_k_classical[:, 1],
         linewidth=0.5, alpha=0.8, color='#1f77b4')
plt.scatter(y0[0], y0[1], color='green', s=100, marker='o', zorder=10, label='Start')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('SPROTT K (Classical) 2D (X-Y)')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('02_sprott_k_classical_2d.png', dpi=300)
plt.close()

# SPROTT K Fractional - 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(sol_sprott_k_frac[:, 0],
        sol_sprott_k_frac[:, 1],
        sol_sprott_k_frac[:, 2],
        linewidth=0.5, alpha=0.8, color='#ff7f0e')
ax.scatter(y0[0], y0[1], y0[2], color='green', s=100, marker='o', zorder=10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('SPROTT K (Fractional α=0.85) 3D')
plt.tight_layout()
plt.savefig('03_sprott_k_frac_3d.png', dpi=300)
plt.close()

# SPROTT K Fractional - 2D
fig = plt.figure(figsize=(8, 6))
plt.plot(sol_sprott_k_frac[:, 0], sol_sprott_k_frac[:, 1],
         linewidth=0.5, alpha=0.8, color='#ff7f0e')
plt.scatter(y0[0], y0[1], color='green', s=100, marker='o', zorder=10, label='Start')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('SPROTT K (Fractional α=0.85) 2D (X-Y)')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('04_sprott_k_frac_2d.png', dpi=300)
plt.close()

# Rössler Classical - 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(sol_rossler_classical[:, 0],
        sol_rossler_classical[:, 1],
        sol_rossler_classical[:, 2],
        linewidth=0.5, alpha=0.8, color='#2ca02c')
ax.scatter(y0[0], y0[1], y0[2], color='green', s=100, marker='o', zorder=10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Rössler (Classical) 3D')
plt.tight_layout()
plt.savefig('05_rossler_classical_3d.png', dpi=300)
plt.close()

# Rössler Classical - 2D
fig = plt.figure(figsize=(8, 6))
plt.plot(sol_rossler_classical[:, 0], sol_rossler_classical[:, 1],
         linewidth=0.5, alpha=0.8, color='#2ca02c')
plt.scatter(y0[0], y0[1], color='green', s=100, marker='o', zorder=10, label='Start')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Rössler (Classical) 2D (X-Y)')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('06_rossler_classical_2d.png', dpi=300)
plt.close()

# Rössler Fractional - 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(sol_rossler_frac[:, 0],
        sol_rossler_frac[:, 1],
        sol_rossler_frac[:, 2],
        linewidth=0.5, alpha=0.8, color='#d62728')
ax.scatter(y0[0], y0[1], y0[2], color='green', s=100, marker='o', zorder=10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Rössler (Fractional α=0.85) 3D')
plt.tight_layout()
plt.savefig('07_rossler_frac_3d.png', dpi=300)
plt.close()

# Rössler Fractional - 2D
fig = plt.figure(figsize=(8, 6))
plt.plot(sol_rossler_frac[:, 0], sol_rossler_frac[:, 1],
         linewidth=0.5, alpha=0.8, color='#d62728')
plt.scatter(y0[0], y0[1], color='green', s=100, marker='o', zorder=10, label='Start')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Rössler (Fractional α=0.85) 2D (X-Y)')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('08_rossler_frac_2d.png', dpi=300)
plt.close()

print("\n" + "="*60)
print("[OK] Generation complete!")
print("="*60)
print("\nGenerated files:")
print("  1. 01_sprott_k_classical_3d.png")
print("  2. 02_sprott_k_classical_2d.png")
print("  3. 03_sprott_k_frac_3d.png")
print("  4. 04_sprott_k_frac_2d.png")
print("  5. 05_rossler_classical_3d.png")
print("  6. 06_rossler_classical_2d.png")
print("  7. 07_rossler_frac_3d.png")
print("  8. 08_rossler_frac_2d.png")
