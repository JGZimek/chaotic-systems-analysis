import numpy as np
from src.systems import generate_lorenz, generate_sprott_k, sprott_k_rhs, caputo_fractional_ode_solver
from src.signals import get_random_signal, get_periodic_signal
from src.plotting import plot_time_series_multi, plot_trajectory_3d

N = 10000
T_MAX = 100.0
T = np.linspace(0, T_MAX, N)
y0 = np.array([0.5, 0.5, 0.5])

# Random signal
random_signal = get_random_signal(N, seed=42)
plot_time_series_multi(
    signals=[random_signal],
    t=T,
    labels=["Random Signal"],
    title="Random Signal over Time",
    ylabel="Amplitude",
    xlabel="Time",
    save_path="data/random_signal.png"
)

# Periodic signal
periodic_signal = get_periodic_signal(N, freq=2.0, phase=0.0)
plot_time_series_multi(
    signals=[periodic_signal],
    t=T,
    labels=["Periodic Signal"],
    title="Periodic Signal over Time",
    ylabel="Amplitude",
    xlabel="Time",
    save_path="data/periodic_signal.png"
)

# Lorenz system trajectory
lorenz_trajectory = generate_lorenz((T[0], T[-1]), y0, T)
plot_time_series_multi(
    signals=[lorenz_trajectory[:, 0], lorenz_trajectory[:, 1], lorenz_trajectory[:, 2]],
    t=T,
    labels=["X", "Y", "Z"],
    title="Lorenz System Trajectory Components over Time",
    ylabel="Value",
    xlabel="Time",
    save_path="data/lorenz_trajectory.png"
)
plot_trajectory_3d(
    trajectory=lorenz_trajectory,
    title="Lorenz System 3D Trajectory",
    labels=["X", "Y", "Z"],
    save_path="data/lorenz_3d_trajectory.png"
)

# Sprott K system trajectory
sprott_k_trajectory = generate_sprott_k((T[0], T[-1]), y0, T)
plot_time_series_multi(
    signals=[sprott_k_trajectory[:, 0], sprott_k_trajectory[:, 1], sprott_k_trajectory[:, 2]],
    t=T,
    labels=["X", "Y", "Z"],
    title="Sprott K System Trajectory Components over Time",
    ylabel="Value",
    xlabel="Time",
    save_path="data/sprott_k_trajectory.png"
)
plot_trajectory_3d(
    trajectory=sprott_k_trajectory,
    title="Sprott K System 3D Trajectory",
    labels=["X", "Y", "Z"],
    save_path="data/sprott_k_3d_trajectory.png"
)

# Sprott K fractional system trajectory with alpha=0.85
sprott_k_frac_085 = caputo_fractional_ode_solver(0.85, lambda t, y: sprott_k_rhs(t, y, a=0.3), y0, T)
plot_time_series_multi(
    signals=[sprott_k_frac_085[:, 0], sprott_k_frac_085[:, 1], sprott_k_frac_085[:, 2]],
    t=T,
    labels=["X", "Y", "Z"],
    title="Sprott K Fractional System (α=0.85) Trajectory Components over Time",
    ylabel="Value",
    xlabel="Time",
    save_path="data/sprott_k_fractional_085_trajectory.png"
)
plot_trajectory_3d(
    trajectory=sprott_k_frac_085,
    title="Sprott K Fractional System (α=0.85) 3D Trajectory",
    labels=["X", "Y", "Z"],
    save_path="data/sprott_k_fractional_085_3d_trajectory.png"
)

# Sprott K fractional system trajectory with alpha=0.95
sprott_k_frac_095 = caputo_fractional_ode_solver(0.95, lambda t, y: sprott_k_rhs(t, y, a=0.3), y0, T)
plot_time_series_multi(
    signals=[sprott_k_frac_095[:, 0], sprott_k_frac_095[:, 1], sprott_k_frac_095[:, 2]],
    t=T,
    labels=["X", "Y", "Z"],
    title="Sprott K Fractional System (α=0.95) Trajectory Components over Time",
    ylabel="Value",
    xlabel="Time",
    save_path="data/sprott_k_fractional_095_trajectory.png"
)
plot_trajectory_3d(
    trajectory=sprott_k_frac_095,
    title="Sprott K Fractional System (α=0.95) 3D Trajectory",
    labels=["X", "Y", "Z"],
    save_path="data/sprott_k_fractional_095_3d_trajectory.png"
)