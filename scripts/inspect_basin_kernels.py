"""Inspect kernel sizes at the two CV basins for the isotropic-Maxwellian
scenario: basin A = (sh=0.5, cf_user=1.0), basin B = (sh=0.0, cf_user=0.05).

All kernel covariances are converted back to ORIGINAL data coordinates by
unscaling with the 'auto' diag(scales) used in the adaptiveKDE metric,
so kernel sigma per axis can be compared directly to data std per axis.
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, 'src'))
from cracked import adaptiveKDE


def make_isotropic_data(N, sigma=1.0, dx=10.0, seed=0):
    rng = np.random.default_rng(seed)
    coords = np.empty((N, 6))
    coords[:, :3] = rng.uniform(-dx / 2, dx / 2, size=(N, 3))
    coords[:, 3:] = rng.normal(0, sigma, size=(N, 3))
    return coords


def kernel_at_point(kde, i, covfac_user, covalpha_user=0.0, natural_covfac=1.0):
    """Return Sigma_kernel at data point i in *scaled* coords (= adaptiveKDE's
    internal kdtree-metric coords), including all natural normalizations.

    Sigma_kernel = (natural_covfac x covfac_user) x Sigma_eff_i x exp(alpha*logdet_i/d)

    where natural_covfac is cvAdaptiveKDE's silverman^2*geomean(diag) so that
    cf_user=1 corresponds to "Silverman in the picked-scaling metric".
    """
    d = kde.data.shape[1]
    log_det = kde.logdets[i]
    cov_eff = kde.covariances[i]
    overall = natural_covfac * covfac_user
    alpha_term = (kde.covalpha_overall + covalpha_user) * log_det
    return overall * np.exp(alpha_term / d) * cov_eff


def unscale_to_orig(Sigma_scaled, scales):
    """Sigma_orig = D * Sigma_scaled * D, where D = diag(scales)."""
    D = np.diag(np.asarray(scales))
    return D @ Sigma_scaled @ D


def report(name, Sigma_orig, data_std):
    sigma_per_axis = np.sqrt(np.diag(Sigma_orig))
    print(f"\n=== {name} ===")
    print(f"  kernel sigma per axis  (x,y,z, vx,vy,vz):  "
          f"[{', '.join(f'{s:.3f}' for s in sigma_per_axis)}]")
    print(f"  data std per axis  (x,y,z, vx,vy,vz):  "
          f"[{', '.join(f'{s:.3f}' for s in data_std)}]")
    print(f"  kernel/data ratio:                     "
          f"[{', '.join(f'{s/d:.3f}' for s,d in zip(sigma_per_axis, data_std))}]")
    sign, logdet = np.linalg.slogdet(Sigma_orig)
    print(f"  log|Sigma_kernel|:                          {logdet:.2f}")
    print(f"  |Sigma_kernel|^(1/6) (geom-mean linear sigma):  {np.exp(logdet/6):.4f}")


def main():
    N = 8000
    coords = make_isotropic_data(N)
    data_std = coords.std(axis=0)
    print(f"N={N}, data shape={coords.shape}")
    print(f"data std per axis: {data_std}")

    # Build adaptiveKDE with scaling='auto', shrinkage_target='local_pooled',
    # K_pool = max(50, sqrt(N)) which is 89 at N=8000.
    K_pool = max(50, int(np.sqrt(N)))
    # adaptiveKDE doesn't resolve 'auto' itself (that's cvAdaptiveKDE's job),
    # so we hand-build the per-axis-std scalings.
    scalings_auto = list(coords.std(axis=0))
    kde = adaptiveKDE(coords, scalings=scalings_auto, nn=max(10, int(np.sqrt(N))), shrinkage_target='local_pooled', K_pool=K_pool)
    # cvAdaptiveKDE applies an additional natural_covfac normalization so that
    # cf_user=1 corresponds to "Silverman volume in the picked-scaling metric".
    # It's `silverman^2 x exp(mean(log(diag(cov_scaled))))`. We replicate it here.
    d = 6
    silverman_sq = N ** (-2.0 / (d + 4))
    data_scaled = coords / np.asarray(scalings_auto)[None, :]
    diag_scaled = np.diagonal(np.cov(data_scaled.T))
    natural_covfac = silverman_sq * np.exp(np.mean(np.log(np.maximum(diag_scaled, 1e-300))))
    print(f"scales (= data.std under auto): {kde.scales}")
    print(f"silverman^2 at N={N}, d={d}: {silverman_sq:.4f}")
    print(f"cv-equivalent natural_covfac (multiplies cf_user): {natural_covfac:.4f}")
    print(f"K_pool: {K_pool}, nn: {kde.nn}")

    # scipy.gaussian_kde reference: Scott's bandwidth = N^(-1/(d+4)) ~ sqrtsilverman^2
    # times data.std per axis. We report linear sigma per axis directly.
    scott_factor = N ** (-1.0 / (d + 4))
    print(f"\nscipy.gaussian_kde reference (Scott): sigma per axis = "
          f"{[f'{scott_factor * s:.3f}' for s in data_std]}")
    print(f"  (Scott factor x data_std per axis; covers ~{scott_factor:.1%} of data std)")

    # Pick a data point near the centre for inspection.
    center_dist = np.linalg.norm(coords, axis=1)
    i0 = int(np.argmin(center_dist))
    print(f"\nInspecting data point i={i0}, coords={coords[i0]}")

    # --- Basin A: sh=0.5, cf_user=1.0
    kde.apply_shrinkage(0.5)
    Sigma_scaled_A = kernel_at_point(kde, i0, covfac_user=1.0, covalpha_user=0.0, natural_covfac=natural_covfac)
    Sigma_orig_A = unscale_to_orig(Sigma_scaled_A, kde.scales)
    report("Basin A: sh=0.5, cf_user=1.0", Sigma_orig_A, data_std)

    # --- Basin B: sh=0.0, cf_user=0.05
    kde.apply_shrinkage(0.0)
    Sigma_scaled_B = kernel_at_point(kde, i0, covfac_user=0.05, covalpha_user=0.0, natural_covfac=natural_covfac)
    Sigma_orig_B = unscale_to_orig(Sigma_scaled_B, kde.scales)
    report("Basin B: sh=0.0, cf_user=0.05", Sigma_orig_B, data_std)

    # --- Aggregate across all data points so we don't get fooled by one point
    print("\n=== Aggregate kernel sigma across all N data points ===")
    for label, sh, cf in [("Basin A", 0.5, 1.0), ("Basin B", 0.0, 0.05)]:
        kde.apply_shrinkage(sh)
        sigmas_per_axis = np.zeros((N, 6))
        for i in range(N):
            S = unscale_to_orig(kernel_at_point(kde, i, cf, natural_covfac=natural_covfac), kde.scales)
            sigmas_per_axis[i] = np.sqrt(np.diag(S))
        med = np.median(sigmas_per_axis, axis=0)
        p5  = np.percentile(sigmas_per_axis, 5, axis=0)
        p95 = np.percentile(sigmas_per_axis, 95, axis=0)
        print(f"\n  {label} (sh={sh}, cf={cf}):")
        for ax in range(6):
            print(f"    axis {ax}: median sigma = {med[ax]:.3f}  "
                  f"(p5={p5[ax]:.3f}, p95={p95[ax]:.3f})  "
                  f"data_std={data_std[ax]:.3f}  "
                  f"ratio_med={med[ax]/data_std[ax]:.3f}")

    # --- Volume ratio
    print("\n=== Volume comparison ===")
    print(f"  basin A median |Sigma|^(1/6) (linear size proxy): ", end='')
    kde.apply_shrinkage(0.5)
    vols_A = []
    for i in range(N):
        S = unscale_to_orig(kernel_at_point(kde, i, 1.0, natural_covfac=natural_covfac), kde.scales)
        _, ld = np.linalg.slogdet(S)
        vols_A.append(ld)
    print(f"{np.exp(np.median(vols_A) / 6):.4f}")

    kde.apply_shrinkage(0.0)
    vols_B = []
    for i in range(N):
        S = unscale_to_orig(kernel_at_point(kde, i, 0.05, natural_covfac=natural_covfac), kde.scales)
        _, ld = np.linalg.slogdet(S)
        vols_B.append(ld)
    print(f"  basin B median |Sigma|^(1/6) (linear size proxy): "
          f"{np.exp(np.median(vols_B) / 6):.4f}")
    print(f"  ratio (A/B) in linear size: "
          f"{np.exp(np.median(vols_A)/6) / np.exp(np.median(vols_B)/6):.3f}")
    print(f"  ratio (A/B) in 6D volume:   "
          f"{np.exp((np.median(vols_A) - np.median(vols_B))):.1f}")


if __name__ == "__main__":
    main()
