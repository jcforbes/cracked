"""Post-hoc test of an AIC-style smoothness penalty on the cached cv_grids.

Hypothesis: adding `combined = avg_score + lambda x log(kernel_vol)` would flip
the isotropic narrow-cf picks to wide (good for density scatter) while
leaving disk_stream picks alone (where narrow is genuinely correct).

Note: the cache stores raw avg_scores. The recorded picks were made with
stability_lambda=2.0 on top, so this post-hoc analysis is approximate.
What we care about is whether the AIC penalty meaningfully shifts the
argmax distribution toward wider kernels on isotropic without doing the
same on disk_stream.
"""
import os, sys
import pickle
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, 'src'))


def kernel_log_vol_grid(cg, d=6):
    """Compute log(kernel_vol_proxy) per grid cell.

    kernel_vol_proxy ~ (cf_user x natural_covfac x ((1-sh) + sh x silverman^2))^d
    where silverman^2 applies because the shrinkage formula mixes in
    silverman^2*Sigma_target. We pull silverman^2 from the data by inverting
    natural_covfac = silverman^2*geomean(diag(Sigma_scaled)) - but the geomean
    isn't stored. Approximate silverman^2 ~ N^(-2/(d+4)).
    """
    cfs = cg['covfacs_user']            # (n_cf,)
    shs = cg['shrinkages']              # (n_sh,)
    nat = cg['natural_covfacs_per_scaling']  # (n_sc,)
    n_cf, n_ca, _, _, n_sh, n_sc = cg['avg_scores'].shape

    # silverman^2 inferred from natural_covfac (the geomean is ~1 for auto-scaled
    # data; for narrow_local with tighter scales it can differ - but the relative
    # vol ranking within a scaling is dominated by cf x sh_factor).
    # For this experiment we use natural_covfac directly as the per-scaling
    # multiplier and a flat silverman^2 approximation for the sh factor.
    N_proxy = 1000   # placeholder - silverman^2 varies slowly with N, use a fixed value
    silverman_sq = 0.2   # ~ N^(-1/5) at N~3000; fine for ranking

    log_vol = np.zeros((n_cf, n_ca, 1, 1, n_sh, n_sc))
    for ic in range(n_cf):
        for ish in range(n_sh):
            sh_factor = (1 - shs[ish]) + shs[ish] * silverman_sq
            for isc in range(n_sc):
                eff = cfs[ic] * nat[isc] * sh_factor
                log_vol[ic, :, 0, 0, ish, isc] = d * np.log(max(eff, 1e-30))
    return log_vol


def repick_with_penalty(cg, lam):
    """Return new rate_best with combined = avg_score + lambda x log(kernel_vol)."""
    avg = cg['avg_scores']
    log_vol = kernel_log_vol_grid(cg)
    combined = avg + lam * log_vol
    flat_idx = int(np.argmax(combined))
    return np.unravel_index(flat_idx, combined.shape)


def kernel_width_proxy(cf_user, sh, N, d=6):
    sil_sq = N ** (-2.0 / (d + 4))
    return cf_user * ((1 - sh) + sh * sil_sq)


def main():
    with open('convergence_rate_vs_N_cache.pkl', 'rb') as f:
        cache = pickle.load(f)
    Ns = cache['Ns']

    for scenario in ['isotropic', 'disk_stream']:
        r = cache['results'][(scenario, 'cvAdaptive')]
        print(f"\n========= {scenario} =========")
        for lam in [0.0, 1e-7, 1e-6, 1e-5]:
            print(f"\n  lambda = {lam:g}")
            print(f"  {'N':>6}{'orig narrow':>14}{'new narrow':>14}{'narrow->wide':>14}{'wide->narrow':>14}")
            for Ni, N in enumerate(Ns):
                if N < 1000:
                    continue
                orig_picks = r['picks_rate'][Ni]
                cg_list = r['cv_grid'][Ni]
                orig_narrow = 0
                new_narrow = 0
                flipped_to_wide = 0
                flipped_to_narrow = 0
                for t, (pick_dict, cg) in enumerate(zip(orig_picks, cg_list)):
                    # Original narrow classification
                    orig_w = kernel_width_proxy(pick_dict['covfac_user'], pick_dict['shrinkage'], N)
                    orig_is_narrow = (orig_w <= 0.1)
                    orig_narrow += orig_is_narrow
                    # Repick with penalty
                    new_pick = repick_with_penalty(cg, lam)
                    cfs = cg['covfacs_user']
                    shs = cg['shrinkages']
                    new_w = kernel_width_proxy(cfs[new_pick[0]], shs[new_pick[4]], N)
                    new_is_narrow = (new_w <= 0.1)
                    new_narrow += new_is_narrow
                    if orig_is_narrow and not new_is_narrow:
                        flipped_to_wide += 1
                    elif not orig_is_narrow and new_is_narrow:
                        flipped_to_narrow += 1
                n_trials = len(orig_picks)
                print(f"  {N:>6}  {orig_narrow}/{n_trials:>10}    {new_narrow}/{n_trials:>10}    "
                      f"{flipped_to_wide:>11}    {flipped_to_narrow:>11}")


if __name__ == "__main__":
    main()
