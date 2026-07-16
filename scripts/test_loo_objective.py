"""Quick comparison of rr_method='loo' vs rr_method='bootstrap' for the CV
objective. We rebuild cvAdaptiveKDE on isotropic and disk_stream data at one
moderate N value, with three seeds each, and report whether LOO picks fall
in the narrow or wide basin.

Heuristic: if LOO consistently picks WIDE on isotropic (where wide is correct)
AND consistently picks NARROW on disk_stream (where narrow is correct),
it's a promising replacement for bootstrap-CV. If it fails either, it's not.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, 'tests'))

from cracked import cvAdaptiveKDE, make_data_driven_is_proposal


def make_isotropic_data(N, sigma=1.0, dx=10.0, seed=0):
    rng = np.random.default_rng(seed)
    coords = np.empty((N, 6))
    coords[:, :3] = rng.uniform(-dx / 2, dx / 2, size=(N, 3))
    coords[:, 3:] = rng.normal(0, sigma, size=(N, 3))
    return coords


def make_disk_stream_data(N, seed=0, R_ring=8000.0, v_circ=220.0, sigma_R=1.0, sigma_z=1.0, sigma_t=0.1):
    """Local-coordinate disk-stream around the Sun (origin)."""
    rng = np.random.default_rng(seed)
    # Stream runs along phi; Sun is at (R_ring, 0, 0), so locally the stream
    # axis is the y-direction (azimuthal at the Sun). Generate phi uniformly,
    # but in local coords we want a thin radial/vertical spread and a wide
    # azimuthal extent.
    phi = rng.uniform(-np.pi, np.pi, size=N)
    R = R_ring + rng.normal(0, sigma_R, size=N)
    z = rng.normal(0, sigma_z, size=N)
    # Sun at (R_ring, 0, 0); translate so the Sun is at origin
    x = R * np.cos(phi) - R_ring
    y = R * np.sin(phi)
    # Velocities: bulk v_circ along phihat (so locally along +y at sun);
    # subtract sun velocity which is also v_circ -> relative velocity is small
    vR = rng.normal(0, sigma_R, size=N)
    vz = rng.normal(0, sigma_z, size=N)
    vt = rng.normal(0, sigma_t, size=N)
    # Total v in cartesian: v_phi = v_circ + vt (along phi-hat)
    vx = vR * np.cos(phi) - (v_circ + vt) * np.sin(phi)
    vy = vR * np.sin(phi) + (v_circ + vt) * np.cos(phi)
    # subtract sun's circular motion at (R_ring, 0): vsun = (0, v_circ, 0)
    vy -= v_circ
    coords = np.column_stack([x, y, z, vx, vy, vz])
    return coords


def build_cv(coords, rr_method, seed):
    """Production-canonical CV grid with the chosen rr_method."""
    v_mean, v_cov = make_data_driven_is_proposal(coords, xloc=(0.0, 0.0, 0.0))
    rng_pts = np.random.default_rng(42)
    n_pts = 200
    n_hat = rng_pts.normal(size=(n_pts, 3))
    n_hat /= np.linalg.norm(n_hat, axis=1, keepdims=True)
    v_samples = rng_pts.multivariate_normal(v_mean, v_cov, size=n_pts)
    neff_pts = np.zeros((n_pts, 6))
    neff_pts[:, :3] = 0.1 * n_hat
    neff_pts[:, 3:] = v_samples

    return cvAdaptiveKDE(coords, nfolds=5, random_state=seed, scalings_grid=['auto', 'narrow', 'narrow_local'], shrinkage_target='local_pooled', shrinkage_grid=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0], covfac_range=(-3.0, 0.5), ncovfacs=15, covalpha_range=(-0.2, 0.6), ncovalphas=5, neff_floor=30.0, neff_eval_points=neff_pts, roi=None, rr_method=rr_method, roi6=200, roiCenter6=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], stability_lambda=2.0)


def classify_basin(cf_user, sh, N, d=6):
    sil_sq = N ** (-2.0 / (d + 4))
    width = cf_user * ((1 - sh) + sh * sil_sq)
    return 'narrow' if width <= 0.1 else 'wide'


RESULTS_FILE = "test_loo_objective_results.txt"


def log_cell(line):
    """Append + print - survives pipe truncation / interrupts."""
    print(line, flush=True)
    with open(RESULTS_FILE, "a") as f:
        f.write(line + "\n")


def main():
    open(RESULTS_FILE, "w").close()
    log_cell(f"# LOO vs bootstrap comparison - {os.path.basename(__file__)}")

    N = 4000        # moderate N to keep runtime tractable
    seeds = [0, 1, 2]

    cfs_grid = np.logspace(-3.0, 0.5, 15)
    shs_grid = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    scalings_list = ['auto', 'narrow', 'narrow_local']

    for scenario_name, data_maker in [('isotropic', make_isotropic_data),
                                       ('disk_stream', make_disk_stream_data)]:
        log_cell(f"\n========= {scenario_name} (N={N}) =========")
        for rr_method in ['bootstrap', 'loo']:
            log_cell(f"\n  --- rr_method={rr_method} ---")
            picks = []
            for seed in seeds:
                coords = data_maker(N, seed=seed)
                cv = build_cv(coords, rr_method=rr_method, seed=seed)
                pick = cv.rate_best
                cf_user = cfs_grid[pick[0]]
                sh = shs_grid[pick[4]]
                scaling = scalings_list[pick[5]]
                basin = classify_basin(cf_user, sh, N)
                log_cell(f"    seed={seed}: cf={cf_user:.3f} sh={sh:.2f} "
                         f"scaling={scaling}  ->  {basin}")
                picks.append(basin)
            log_cell(f"    summary: {picks.count('narrow')} narrow, "
                     f"{picks.count('wide')} wide  out of {len(seeds)}")


if __name__ == "__main__":
    main()
