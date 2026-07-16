"""Compare basin A (sh=0.5, cf_user=1.0) and basin B (sh=0.0, cf_user=0.05)
on the four convergence metrics: rate, density bias, density scatter, sky TV.

Isotropic Maxwellian only - that's where the two basins coexist in CV picks.
N=8000 to match the regime where the convergence plot's bimodality is clearest.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, 'src'))
from cracked import (adaptiveKDE, rate_sphere_importance,
                     make_data_driven_is_proposal)


# ---- scenario setup (matches test_rate_sphere_analytic constants)
DX = 10.0      # cube side length, pc
SIGMA = 1.0    # velocity dispersion, km/s
N0 = 1e15      # number density per pc^3 (sets units for rate)
R_SPHERE = 0.1
QMAX_AU = 5.0
MYR_TO_YR = 1e-6


def make_isotropic_data(N, seed=0):
    rng = np.random.default_rng(seed)
    coords = np.empty((N, 6))
    coords[:, :3] = rng.uniform(-DX / 2, DX / 2, size=(N, 3))
    coords[:, 3:] = rng.normal(0, SIGMA, size=(N, 3))
    return coords


def isotropic_truth_pdf(coords):
    """f(x,v) = (1/DX^3) * Maxwellian_3D(v; sigma). Joint PDF, not number density."""
    pos_pdf = (np.abs(coords[:, :3]) <= DX / 2).all(axis=1) / DX ** 3
    v_sq = (coords[:, 3:] ** 2).sum(axis=1)
    v_pdf = (1.0 / (2 * np.pi * SIGMA ** 2)) ** 1.5 * np.exp(-v_sq / (2 * SIGMA ** 2))
    return pos_pdf * v_pdf


def sample_isotropic_truth(M, seed=0):
    rng = np.random.default_rng(seed)
    s = np.empty((M, 6))
    s[:, :3] = rng.uniform(-DX / 2, DX / 2, size=(M, 3))
    s[:, 3:] = rng.normal(0, SIGMA, size=(M, 3))
    return s


def build_basin_kde(coords, sh, cf_user):
    """Build an adaptiveKDE with the cvAdaptiveKDE-equivalent kernel for
    (shrinkage=sh, cf_user). Returns a wrapped callable that exposes
    `kde(pts)` returning probability density (sums to 1 over the data domain)."""
    N, d = coords.shape
    scales = list(coords.std(axis=0))
    K_pool = max(50, int(np.sqrt(N)))
    nn = max(10, int(np.sqrt(N)))
    kde = adaptiveKDE(coords, scalings=scales, nn=nn, shrinkage_target='local_pooled', K_pool=K_pool)
    kde.apply_shrinkage(sh)
    # cvAdaptiveKDE's natural_covfac normalization (silverman^2*geomean(diag_scaled))
    silverman_sq = N ** (-2.0 / (d + 4))
    data_scaled = coords / np.asarray(scales)[None, :]
    diag_scaled = np.diagonal(np.cov(data_scaled.T))
    natural_covfac = silverman_sq * np.exp(np.mean(np.log(np.maximum(diag_scaled, 1e-300))))

    def callable_kde(pts):
        return kde(pts, covfac=natural_covfac * cf_user)
    callable_kde._kde = kde
    callable_kde._natural_covfac = natural_covfac
    callable_kde._effective_covfac = natural_covfac * cf_user
    return callable_kde


# ---- metrics
def metric_rate(kde_callable, fac):
    v_mean, v_cov = make_data_driven_is_proposal(kde_callable._kde.data * np.asarray(kde_callable._kde.scales))
    out = rate_sphere_importance(kde_callable, v_proposal_mean=v_mean, v_proposal_cov=v_cov, Nboot=30_000, qmaxAU=QMAX_AU, r=1.0, fac=fac, xloc=(0, 0, 0), vloc=(0, 0, 0), rng=np.random.default_rng(42))
    weights = np.asarray(out[0])
    rate = float(weights.sum() / len(weights))
    return rate, weights, out


def metric_density_bias_scatter(kde_callable, fac, M=20000, seed=1):
    samples = sample_isotropic_truth(M, seed=seed)
    f_kde = np.asarray(kde_callable(samples)) * fac
    f_truth = isotropic_truth_pdf(samples) * fac
    valid = (f_kde > 0) & (f_truth > 0)
    log_ratio = np.log10(f_kde[valid] / f_truth[valid])
    return float(np.median(log_ratio)), float(np.std(log_ratio))


def metric_sky_tv(weights, vee, costhetas, phis):
    """Sky-map TV for isotropic case: reference is uniform on (cos theta, phi).
    Bin 12x24 like the convergence diagnostic; compare normalized histograms."""
    # Filter unbound (weight > 0)
    mask = weights > 0
    if not mask.any():
        return float('nan')
    H, _, _ = np.histogram2d(costhetas[mask], phis[mask], bins=[12, 24], range=[[-1, 1], [-np.pi, np.pi]], weights=weights[mask])
    pdf = H / H.sum()
    ref = np.ones_like(pdf) / pdf.size  # uniform-sky reference
    return float(0.5 * np.abs(pdf - ref).sum())


# ---- main
def main():
    N = 8000
    coords = make_isotropic_data(N)
    fac = DX ** 3 * N0   # convert prob-density KDE into number density

    print(f"Scenario: isotropic Maxwellian (sigma=1, DX=10), N={N}, fac={fac:.2e}")
    print(f"Analytic rate R_an ~ 526 yr^-1 (per test_rate_sphere_analytic).")

    R_an = 526.0

    basins = [
        ('A', 0.5, 1.0),
        ('B', 0.0, 0.05),
    ]

    results = {}
    for label, sh, cf in basins:
        print(f"\n--- Basin {label}: sh={sh}, cf_user={cf} ---")
        kde_cb = build_basin_kde(coords, sh=sh, cf_user=cf)
        print(f"  natural_covfac x cf_user = {kde_cb._effective_covfac:.4f}")

        rate, weights, out = metric_rate(kde_cb, fac)
        print(f"  rate (yr^-1) = {rate:.1f}  ->  |R/R_an - 1| = {abs(rate/R_an - 1):.3f}")

        bias, scatter = metric_density_bias_scatter(kde_cb, fac)
        print(f"  density bias  (median log10 fhat/f_truth) = {bias:+.3f}")
        print(f"  density scatter (std log10 fhat/f_truth)  = {scatter:.3f}")

        vee, costhetas, phis = out[1], out[2], out[3]
        sky = metric_sky_tv(weights, vee, costhetas, phis)
        print(f"  sky TV (vs uniform reference)           = {sky:.3f}")

        results[label] = {
            'rate_err': abs(rate / R_an - 1),
            'bias': bias,
            'scatter': scatter,
            'sky_tv': sky,
        }

    print("\n\n=== Summary: which basin does worse? ===\n")
    rows = ['rate_err', 'bias', 'scatter', 'sky_tv']
    fmt = "{:<14} {:>10} {:>10} {:>8}"
    print(fmt.format('metric', 'basin A', 'basin B', 'worse'))
    print("-" * 46)
    for metric in rows:
        a = results['A'][metric]
        b = results['B'][metric]
        # For bias, "worse" = larger magnitude. For others, "worse" = larger value.
        if metric == 'bias':
            worse = 'A' if abs(a) > abs(b) else 'B'
        else:
            worse = 'A' if a > b else 'B'
        print(fmt.format(metric, f"{a:+.3f}", f"{b:+.3f}", worse))


if __name__ == "__main__":
    main()
