"""Sweep stability_lambda post-hoc from a fresh cvAdaptiveKDE build.

For each (scenario, seed), build one CV at moderate N, then iterate
through lambda  in  [0, 0.5, 1, 2, 5, 10] reusing `cv.scores_per_fold` to
recompute the combined_score = avg - lambda*SEM and the resulting rate-best
and basin classification. Doesn't need re-running the inner CV loop;
the per-fold scores are stored on the cv object once.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, 'src'))
from cracked import cvAdaptiveKDE, make_data_driven_is_proposal


# Data makers (same as test_loo_objective.py)
def make_isotropic(N, seed=0, sigma=1.0, dx=10.0):
    rng = np.random.default_rng(seed)
    coords = np.empty((N, 6))
    coords[:, :3] = rng.uniform(-dx/2, dx/2, size=(N, 3))
    coords[:, 3:] = rng.normal(0, sigma, size=(N, 3))
    return coords


def make_disk_stream(N, seed=0):
    rng = np.random.default_rng(seed)
    R_ring, v_circ = 8000.0, 220.0
    phi = rng.uniform(-np.pi, np.pi, size=N)
    R = R_ring + rng.normal(0, 1.0, size=N)
    z = rng.normal(0, 1.0, size=N)
    x = R*np.cos(phi) - R_ring
    y = R*np.sin(phi)
    vt = rng.normal(0, 0.1, size=N)
    vR = rng.normal(0, 1.0, size=N)
    vz = rng.normal(0, 1.0, size=N)
    vx = vR*np.cos(phi) - (v_circ + vt)*np.sin(phi)
    vy = vR*np.sin(phi) + (v_circ + vt)*np.cos(phi) - v_circ
    return np.column_stack([x, y, z, vx, vy, vz])


def build_cv(coords, seed):
    v_mean, v_cov = make_data_driven_is_proposal(coords, xloc=(0.0, 0.0, 0.0))
    rng_pts = np.random.default_rng(42)
    n_hat = rng_pts.normal(size=(200, 3))
    n_hat /= np.linalg.norm(n_hat, axis=1, keepdims=True)
    v = rng_pts.multivariate_normal(v_mean, v_cov, size=200)
    neff_pts = np.zeros((200, 6)); neff_pts[:,:3] = 0.1*n_hat; neff_pts[:,3:] = v
    return cvAdaptiveKDE(
        coords, nfolds=5, random_state=seed,
        scalings_grid=['auto', 'narrow', 'narrow_local'],
        shrinkage_target='local_pooled',
        shrinkage_grid=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
        covfac_range=(-3.0, 0.5), ncovfacs=15,
        covalpha_range=(-0.2, 0.6), ncovalphas=5,
        neff_floor=30.0, neff_eval_points=neff_pts,
        roi=None, rr_method='bootstrap',
        roi6=200, roiCenter6=[0.0]*6,
        stability_lambda=0.0,  # base; we'll apply lambda post-hoc
    )


def classify(cf, sh, N, d=6):
    sil_sq = N ** (-2.0 / (d + 4))
    return 'narrow' if cf * ((1-sh) + sh*sil_sq) <= 0.1 else 'wide'


def pick_at_lambda(cv, lam, N):
    """Recompute rate-best / shape-best at given stability_lambda."""
    pfv = cv.scores_per_fold       # (cf, ca, nn, vf, sh, sc, fold)
    avg = pfv.mean(axis=-1)
    std = pfv.std(axis=-1, ddof=1)
    sem = std / np.sqrt(pfv.shape[-1])
    combined = avg - lam * sem
    # rate pick: unconstrained argmax
    rate_idx = np.unravel_index(np.nanargmax(combined), combined.shape)
    # shape pick: same but masked by neff floor
    neffs_med = np.nanmedian(cv.neffs, axis=-1)   # (cf, ca, nn, vf, sh, sc)
    eligible = neffs_med >= 30
    if np.any(eligible):
        masked = np.where(eligible, combined, -np.inf)
        be = float(np.nanmax(masked))
        bu = float(np.nanmax(combined))
        if bu > 0 and be < 0.1 * bu:
            shape_idx = rate_idx
        else:
            shape_idx = np.unravel_index(np.nanargmax(masked), masked.shape)
    else:
        shape_idx = rate_idx
    cfs = np.logspace(-3, 0.5, 15)
    cas = np.array([-0.2, 0, 0.2, 0.4, 0.6])
    shs = np.array([0, 0.1, 0.25, 0.5, 0.75, 1.0])
    scs = ['auto', 'narrow', 'narrow_local']
    def fmt(idx):
        cf, ca, sh, sc = cfs[idx[0]], cas[idx[1]], shs[idx[4]], scs[idx[5]]
        return cf, ca, sh, sc, classify(cf, sh, N)
    return fmt(rate_idx), fmt(shape_idx)


def main():
    N = 4000
    lambdas = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]

    for sc_name, maker in [('isotropic', make_isotropic),
                            ('disk_stream', make_disk_stream)]:
        print(f"\n========= {sc_name} (N={N}) =========\n")
        for seed in [0, 1]:
            coords = maker(N, seed=seed)
            print(f"  --- seed={seed} ---")
            print(f"    building CV ... ", end='', flush=True)
            cv = build_cv(coords, seed=seed)
            print("done")
            print(f"    {'lambda':>5}    {'rate cf':>7} {'rate sh':>7} {'rate sc':<15}"
                  f"  {'rate basin':<10}    {'shape cf':>8} {'shape sh':>8} {'shape basin':<10}")
            for lam in lambdas:
                rate_p, shape_p = pick_at_lambda(cv, lam, N)
                rc, _, rs, rsc, rb = rate_p
                sc, _, ss, ssc, sb = shape_p
                print(f"    {lam:>5.1f}    {rc:>7.3f} {rs:>7.2f} {rsc:<15}  {rb:<10}"
                      f"    {sc:>8.3f} {ss:>8.2f} {sb:<10}")
            print()


if __name__ == "__main__":
    main()
