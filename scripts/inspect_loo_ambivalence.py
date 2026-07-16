"""How strongly does LOO actually prefer the low-cf basin on disk_stream?
Compare the top-10 avg_scores from a bootstrap and a LOO CV on the same data."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, 'src'))
from cracked import cvAdaptiveKDE, make_data_driven_is_proposal

# Reuse the same data maker as test_loo_objective
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


def build_cv(coords, rr_method, seed):
    v_mean, v_cov = make_data_driven_is_proposal(coords, xloc=(0.0, 0.0, 0.0))
    rng_pts = np.random.default_rng(42)
    n_pts = 200
    n_hat = rng_pts.normal(size=(n_pts, 3))
    n_hat /= np.linalg.norm(n_hat, axis=1, keepdims=True)
    v_samples = rng_pts.multivariate_normal(v_mean, v_cov, size=n_pts)
    neff_pts = np.zeros((n_pts, 6))
    neff_pts[:, :3] = 0.1 * n_hat
    neff_pts[:, 3:] = v_samples
    return cvAdaptiveKDE(coords, nfolds=5, random_state=seed, scalings_grid=['auto', 'narrow', 'narrow_local'], shrinkage_target='local_pooled', shrinkage_grid=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0], covfac_range=(-3.0, 0.5), ncovfacs=15, covalpha_range=(-0.2, 0.6), ncovalphas=5, neff_floor=30.0, neff_eval_points=neff_pts, roi=None, rr_method=rr_method, roi6=200, roiCenter6=[0.0]*6, stability_lambda=2.0)


def top_k_picks(cv, k=10):
    """Top-k grid entries by avg_score, with (cf, sh, scaling) labels."""
    s = cv.avg_scores                # (n_cf, n_ca, n_nn, n_vf, n_sh, n_sc)
    flat_sorted = np.argsort(s, axis=None)[::-1][:k]
    cfs = np.logspace(-3.0, 0.5, 15)
    cas = np.array([-0.2, 0, 0.2, 0.4, 0.6])
    shs = np.array([0, 0.1, 0.25, 0.5, 0.75, 1.0])
    scalings = ['auto', 'narrow', 'narrow_local']
    out = []
    for idx in flat_sorted:
        i = np.unravel_index(idx, s.shape)
        out.append({'score': float(s[i]), 'cf': float(cfs[i[0]]), 'ca': float(cas[i[1]]), 'sh': float(shs[i[4]]), 'sc': scalings[i[5]]})
    return out


def main():
    N = 4000
    seed = 0
    coords = make_disk_stream_data(N, seed=seed)
    print(f"disk_stream N={N}, seed={seed}\n")

    for rr in ['bootstrap', 'loo']:
        print(f"=== rr_method = {rr} ===")
        cv = build_cv(coords, rr_method=rr, seed=seed)
        top = top_k_picks(cv, k=10)
        s_top = top[0]['score']
        print(f"  rank   score      score/top      cf       sh   scaling")
        print(f"  ----   --------   ---------    -------   ----  --------")
        for rank, p in enumerate(top):
            ratio = p['score'] / s_top if s_top != 0 else float('nan')
            print(f"  {rank:>4}   {p['score']:+.4e}    {ratio:>7.4f}  "
                  f"{p['cf']:>7.4f}  {p['sh']:>4.2f}  {p['sc']}")
        print()


if __name__ == "__main__":
    main()
