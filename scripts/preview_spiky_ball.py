"""Quiver-style preview plot for the spiky-ball scenario.

Draws N samples from the analytic spiky-ball DF, restricts to particles
within a narrow z-slice around the origin, and plots each as a dot with
a velocity arrow (vx, vy). This is what would appear as an inset in the
top row of the comparison plots - gives the reader intuition for what
each test scenario looks like geometrically.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, 'tests'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, 'src'))

from test_rate_sphere_analytic import make_spiky_ball_sampler


def main():
    # Parameters: more (25) and narrower (sigma_perp=0.3) spikes than the default
    # spiky_ball test (15 spikes at sigma_perp=1.0 pc).
    n_spikes = 25
    L = 100.0
    sigma_perp = 0.3
    sigma_v_long = 0.3
    sigma_v_perp = 0.05
    v_speed = 3.0

    sampler, n_hat = make_spiky_ball_sampler(n_spikes, L, sigma_perp, sigma_v_long, sigma_v_perp, v_speed)
    coords = sampler(np.random.default_rng(0), 10000)

    # z-slice around origin (= "Sun position")
    z_lo, z_hi = -2.0, 2.0   # pc - wider than sigma_perp=0.3 so we get a few particles per spike
    in_slice = (coords[:, 2] > z_lo) & (coords[:, 2] < z_hi)
    print(f"n particles in z-slice: {in_slice.sum()} / {len(coords)}")

    fig, ax = plt.subplots(figsize=(7, 7))
    sliced = coords[in_slice]
    ax.scatter(sliced[:, 0], sliced[:, 1], s=8, c="C0", alpha=0.6, lw=0)
    # Velocity arrows - colour by |v_z| so the reader can see which way
    # particles are coming out of the plane.
    ax.quiver(sliced[:, 0], sliced[:, 1], sliced[:, 3], sliced[:, 4], sliced[:, 5], cmap='RdBu_r', clim=(-v_speed, +v_speed), angles='xy', scale_units='xy', scale=1.0, width=0.003, alpha=0.7)
    # Mark the Sun
    ax.plot([0], [0], marker="*", ms=20, c="gold", mec="k", mew=1, zorder=5)
    ax.set_aspect("equal")
    ax.set_xlim(-50, 50); ax.set_ylim(-50, 50)
    ax.set_xlabel("x (pc)")
    ax.set_ylabel("y (pc)")
    ax.set_title(f"spiky ball: $N_{{\\rm spikes}}={n_spikes}$, "
                 f"$\\sigma_\\perp={sigma_perp}$ pc, "
                 f"$|z| < {z_hi:.1f}$ pc slice "
                 f"({in_slice.sum()} particles shown)\n"
                 r"arrow $=$ $(v_x, v_y)$; colour $=$ $v_z$")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig("preview_spiky_ball.pdf", dpi=120)
    print("wrote preview_spiky_ball.pdf")


if __name__ == "__main__":
    main()
