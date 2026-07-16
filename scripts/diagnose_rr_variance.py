"""Diagnose whether RR (bootstrap-RR bootstrap term) variance correlates with
CV picking near-delta kernels on flat-density isotropic data.

Hypothesis: on isotropic data with 'auto' scaling, CV occasionally picks
cf ~ 10^(-2) (much narrower than Silverman) - different trials make this
choice with substantial probability, producing the bimodal cf picks visible
in the convergence_cv_picks plot.

The CV score is `logsumexp([score, RR], b=[1.0, -0.5])`. If RR has high
fold-to-fold variance at low cf (because narrow kernels make the self-overlap
bootstrap noisy), some folds happen to land on a low-RR realisation -> low
penalty -> that (cf, fold) combo wins. Multiple folds with such realisations
in the same trial -> trial picks low cf.

This script builds one cvAdaptiveKDE on isotropic-Maxwellian data, then plots
across (cf, fold) the per-fold RR - looking specifically at whether the
across-fold std of RR explodes at low cf for scaling='auto'.

Usage:
    PYTHONPATH=src ipython3 scripts/diagnose_rr_variance.py

Output: diagnose_rr_variance.pdf in the current directory.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow running from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, 'src'))

from cracked import cvAdaptiveKDE


def make_isotropic_data(N, sigma=1.0, dx=10.0, seed=0):
    """Uniform-cube positions, isotropic Maxwellian velocities."""
    rng = np.random.default_rng(seed)
    coords = np.empty((N, 6))
    coords[:, :3] = rng.uniform(-dx / 2, dx / 2, size=(N, 3))
    coords[:, 3:] = rng.normal(0, sigma, size=(N, 3))
    return coords


def build_cv(coords, seed):
    """Build a tightly-scoped CV: only scaling=auto + sh=0.5 + a few covalphas,
    full cf grid. This isolates the cf-bimodality question on the path that
    the convergence test showed was 8/8 picks at large N (auto)."""
    return cvAdaptiveKDE(
        coords,
        nfolds=5,
        random_state=seed,
        scalings_grid=['auto'],
        shrinkage_target='local_pooled',
        shrinkage_grid=[0.5],          # single shrinkage to isolate the cf axis
        covfac_range=(-3.0, 0.5),
        ncovfacs=15,                   # match production-test grid
        covalpha_range=(0.0, 0.0),     # single covalpha to isolate the cf axis
        ncovalphas=1,
        neff_floor=None,
        roi=None,
        rr_method='bootstrap',
        stability_lambda=2.0,
    )


def plot_diagnostic(cv, ax_score, ax_rr_mean, ax_rr_std, label):
    """Slice scores_per_fold, RRs_per_fold at the singleton (ca, nn, vf, sh, sc)
    to get a (n_cf, n_folds) array; plot mean +/- fold-std vs cf, and also a
    panel showing fold-std alone."""
    # Shape: (n_cf, n_ca, n_nn, n_vf, n_sh, n_sc, n_folds) -> squeeze singletons
    rr  = cv.RRs_per_fold[:, 0, 0, 0, 0, 0, :]       # (n_cf, n_folds)
    sc  = cv.scores_per_fold[:, 0, 0, 0, 0, 0, :]    # (n_cf, n_folds)

    # cf_user values: log10 spacing per the grid.
    cfs_user = np.logspace(-3.0, 0.5, 15)

    rr_mean = rr.mean(axis=1)
    rr_std  = rr.std(axis=1, ddof=1)
    sc_mean = sc.mean(axis=1)
    sc_std  = sc.std(axis=1, ddof=1)

    ax_score.errorbar(cfs_user, sc_mean, yerr=sc_std, fmt='o-', label=label, capsize=2, alpha=0.7)
    ax_rr_mean.errorbar(cfs_user, rr_mean, yerr=rr_std, fmt='o-', label=label, capsize=2, alpha=0.7)
    ax_rr_std.plot(cfs_user, rr_std, 'o-', label=label, alpha=0.7)

    # Mark the picked cf (rate_best argmax over (cf,) since other axes singleton)
    pick_idx = int(cv.rate_best[0])
    ax_score.axvline(cfs_user[pick_idx], color='C0', ls='--', alpha=0.5)
    return cfs_user[pick_idx]


def main():
    Ns = [4000, 16000]
    seeds = [0, 1, 2]   # 3 trials to see across-trial pick variability

    fig, axes = plt.subplots(3, len(Ns), figsize=(6 * len(Ns), 12), sharex=True)
    if len(Ns) == 1:
        axes = axes[:, None]
    for i, N in enumerate(Ns):
        print(f"=== N={N} ===")
        for seed in seeds:
            print(f"  building CV (seed={seed}) ...", flush=True)
            coords = make_isotropic_data(N, seed=seed)
            cv = build_cv(coords, seed=seed)
            picked_cf = plot_diagnostic(cv, axes[0, i], axes[1, i], axes[2, i], label=f"seed={seed}")
            print(f"    picked cf = {picked_cf:.4f}")

        axes[0, i].set_title(f"isotropic N={N}, auto scaling, sh=0.5, ca=0")
        axes[0, i].set_ylabel("CV score (sign*exp(score))\nmean +/- fold std")
        axes[0, i].set_xscale("log")
        axes[0, i].grid(alpha=0.3); axes[0, i].legend(fontsize=8)

        axes[1, i].set_ylabel("RR\nmean +/- fold std")
        axes[1, i].set_xscale("log")
        axes[1, i].grid(alpha=0.3); axes[1, i].legend(fontsize=8)

        axes[2, i].set_ylabel("fold std of RR\n(the diagnostic)")
        axes[2, i].set_xscale("log"); axes[2, i].set_yscale("log")
        axes[2, i].set_xlabel("covfac_user")
        axes[2, i].grid(alpha=0.3); axes[2, i].legend(fontsize=8)

    fig.tight_layout()
    out = "diagnose_rr_variance.pdf"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
