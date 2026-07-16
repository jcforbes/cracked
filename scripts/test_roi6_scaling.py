"""Test whether scaling roi6 with N reduces CV pick bimodality on isotropic
and / or improves rate fidelity on disk_stream.

Holds everything else at the production-canonical values and varies roi6
across {200, 800, 3200} at N=4000. Bimodality is measured by basin
classification (narrow vs wide); rate is measured by rate_sphere_importance.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, 'src'))

from cracked import (cvAdaptiveKDE, make_data_driven_is_proposal,
                     rate_sphere_importance)


# Data makers
def make_isotropic_data(N, sigma=1.0, dx=10.0, seed=0):
    rng = np.random.default_rng(seed)
    coords = np.empty((N, 6))
    coords[:, :3] = rng.uniform(-dx / 2, dx / 2, size=(N, 3))
    coords[:, 3:] = rng.normal(0, sigma, size=(N, 3))
    return coords


def make_disk_stream_data(N, seed=0, R_ring=8000.0, v_circ=220.0, sigma_R=1.0, sigma_z=1.0, sigma_t=0.1):
    rng = np.random.default_rng(seed)
    phi = rng.uniform(-np.pi, np.pi, size=N)
    R = R_ring + rng.normal(0, sigma_R, size=N)
    z = rng.normal(0, sigma_z, size=N)
    x = R * np.cos(phi) - R_ring
    y = R * np.sin(phi)
    vR = rng.normal(0, sigma_R, size=N)
    vz = rng.normal(0, sigma_z, size=N)
    vt = rng.normal(0, sigma_t, size=N)
    vx = vR * np.cos(phi) - (v_circ + vt) * np.sin(phi)
    vy = vR * np.sin(phi) + (v_circ + vt) * np.cos(phi)
    vy -= v_circ
    return np.column_stack([x, y, z, vx, vy, vz])


def build_neff_pts(v_mean, v_cov, r=0.1, n=200, seed=42):
    rng = np.random.default_rng(seed)
    n_hat = rng.normal(size=(n, 3))
    n_hat /= np.linalg.norm(n_hat, axis=1, keepdims=True)
    v = rng.multivariate_normal(v_mean, v_cov, size=n)
    pts = np.zeros((n, 6))
    pts[:, :3] = r * n_hat
    pts[:, 3:] = v
    return pts


def build_cv(coords, roi6, seed):
    """Match make_production_cv_kde except for the roi6 value (experiment knob)."""
    v_mean, v_cov = make_data_driven_is_proposal(coords, xloc=(0.0, 0.0, 0.0))
    neff_pts = build_neff_pts(v_mean, v_cov)
    cv = cvAdaptiveKDE(coords, nfolds=5, random_state=seed, scalings_grid=['auto', 'narrow', 'narrow_local'], shrinkage_target='local_pooled', shrinkage_grid=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0], covfac_range=(-3.0, 0.5), ncovfacs=15, covalpha_range=(-0.2, 0.6), ncovalphas=5, neff_floor=30.0, neff_eval_points=neff_pts, roi=None, rr_method='bootstrap', roi6=roi6, roiCenter6=[0.0] * 6, stability_lambda=2.0)
    cv.v_proposal_mean = v_mean
    cv.v_proposal_cov = v_cov
    return cv


def classify_basin(cf, sh, N, d=6):
    sil_sq = N ** (-2.0 / (d + 4))
    width = cf * ((1 - sh) + sh * sil_sq)
    return 'narrow' if width <= 0.1 else 'wide'


def measure_rate(cv, R_an, fac):
    out = rate_sphere_importance(cv.kde_rate, v_proposal_mean=cv.v_proposal_mean, v_proposal_cov=cv.v_proposal_cov, Nboot=30000, qmaxAU=5.0, r=0.1, fac=fac, xloc=(0, 0, 0), vloc=(0, 0, 0), rng=np.random.default_rng(99))
    rate = float(out[0].sum() / len(out[0]))
    return abs(rate / R_an - 1)


RESULTS_FILE = "test_roi6_scaling_results.txt"


def log_cell(line):
    """Append one line to the results file *and* print, so progress survives
    pipe-truncation / tail / interrupt."""
    print(line, flush=True)
    with open(RESULTS_FILE, "a") as f:
        f.write(line + "\n")


def main():
    # Wipe results file at start
    open(RESULTS_FILE, "w").close()
    log_cell(f"# roi6 scaling experiment - {os.path.basename(__file__)}")

    N = 4000
    seeds = [0, 1]   # 2 seeds per cell to keep runtime under control
    roi6_values = [200, 800, 3200]

    cfs_grid = np.logspace(-3.0, 0.5, 15)
    shs_grid = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    scalings = ['auto', 'narrow', 'narrow_local']

    scenarios = [
        ('isotropic', make_isotropic_data, 526.0, 10.0 ** 3 * 1e15),
        ('disk_stream', make_disk_stream_data, 0.10, None),
    ]

    for name, maker, R_an, fac in scenarios:
        log_cell(f"\n========= {name} (N={N}, R_an={R_an}) =========")
        for roi6 in roi6_values:
            log_cell(f"\n  --- roi6 = {roi6} ---")
            picks = []
            rates = []
            for seed in seeds:
                coords = maker(N, seed=seed)
                cv = build_cv(coords, roi6=roi6, seed=seed)
                pick = cv.rate_best
                cf, sh, sc = cfs_grid[pick[0]], shs_grid[pick[4]], scalings[pick[5]]
                basin = classify_basin(cf, sh, N)
                picks.append(basin)
                if fac is not None:
                    rate_err = measure_rate(cv, R_an, fac)
                    rates.append(rate_err)
                    rate_str = f" rate_err={rate_err:.3f}"
                else:
                    rate_str = ""
                log_cell(f"    seed={seed}: cf={cf:.3f} sh={sh:.2f} "
                         f"sc={sc:<22} -> {basin}{rate_str}")
            n_narrow = picks.count('narrow')
            summary = (f"    summary: {n_narrow}/{len(seeds)} narrow, "
                       f"{len(seeds) - n_narrow}/{len(seeds)} wide")
            if rates:
                summary += f"   median rate_err={np.median(rates):.3f}"
            log_cell(summary)


if __name__ == "__main__":
    main()
