"""
test_convergence.py

Convergence, reproducibility, and edge-case tests for the KDE/rate-sphere
pipeline. Complements:
    test_adaptive_kde.py            (correctness/regression)
    test_rate_sphere_analytic.py    (per-scenario validation)

What's here:
    1. Rate convergence vs N for an isotropic Maxwellian KDE.
    2. RATE_Sphere MC convergence vs Nboot (variance ~ 1/Nboot).
    3. Importance-sampling variance reduction vs uniform RATE_Sphere on a
       narrow distribution.
    4. cvAdaptiveKDE / cvGaussianKDE reproducibility under fixed seed.
    5. pickle round-trip for adaptiveKDE.
    6. make_data_driven_is_proposal with K > N.
    7. make_data_driven_is_proposal with a zero-variance velocity axis
       (sigma_floor_factor activation).

Run:
    ipython3 test_convergence.py                # all tests
    ipython3 test_convergence.py --no-slow      # fast subset
    ipython3 test_convergence.py -k pickle      # filter

Pytest-compatible too:
    ipython3 -m pytest -- test_convergence.py
"""
import os
import pickle
import sys
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cracked import (adaptiveKDE, cvAdaptiveKDE, cvGaussianKDE,
                     gaussianKDEWrapper,
                     make_data_driven_is_proposal, rate_sphere_importance,
                     make_production_cv_kde)
from test_rate_sphere_analytic import (
    DX, N0, SIGMA, NBOOT, R_SPHERE, V0, QMAX_AU, MYR_TO_YR,
    analytic_total_rate, isotropic_fv, drifting_fv,
    make_isotropic_df, make_drifting_df,
    sample_uniform_pos_gaussian_v, run_rate_sphere,
    make_cold_hot_df, make_cold_hot_sampler, cold_hot_fv,
    make_disk_stream_df, make_disk_stream_sampler,
    _suppress_stdout, _make_neff_eval_points,
)


def slow(fn):
    fn._slow = True
    return fn


try:
    import pytest as _pytest  # noqa: F401
    slow = _pytest.mark.slow  # type: ignore
except ImportError:
    pass


# --- shared fixtures (lazy)

_fv_iso = None
_R_an_iso = None


def isotropic_truth():
    """(f_v callable, analytic R) for the default isotropic Maxwellian."""
    global _fv_iso, _R_an_iso
    if _fv_iso is None:
        _fv_iso = isotropic_fv(N0, SIGMA)
        _R_an_iso = analytic_total_rate(_fv_iso)
    return _fv_iso, _R_an_iso


def build_kde(rng, n_samp, sigma=SIGMA):
    """Build a default adaptiveKDE on isotropic Maxwellian samples."""
    coords = sample_uniform_pos_gaussian_v(rng, n_samp, DX, np.zeros(3), sigma)
    return adaptiveKDE(coords, scalings=np.ones(6), nn=50, use_multiprocessing=False), coords


# 1. Rate convergence vs N

CACHE_PATH = "convergence_rate_vs_N_cache.pkl"

# Sky histogram bin scheme - fixed across all trials so TV is comparable.
N_CT_BINS = 12
N_PHI_BINS = 24
CT_EDGES = np.linspace(-1.0, 1.0, N_CT_BINS + 1)
PHI_EDGES = np.linspace(-np.pi, np.pi, N_PHI_BINS + 1)
REF_NBOOT = 200000  # high-precision reference sky


def _sampler_isotropic(rng, N):
    return sample_uniform_pos_gaussian_v(rng, N, DX, np.zeros(3), SIGMA)


def _setup_isotropic():
    """Returns scenario dict for isotropic Maxwellian sigma=1, uniform position."""
    return dict(
        label=r"isotropic Maxwellian ($\sigma=1$)",
        short="isotropic",
        sampler=_sampler_isotropic,
        df=make_isotropic_df(N0, SIGMA),
        fac=DX ** 3 * N0,
        R_an=analytic_total_rate(isotropic_fv(N0, SIGMA)),
        xloc=(0.0, 0.0, 0.0),
        vloc=(0.0, 0.0, 0.0),
        # For sky reference: use data-driven proposal from a big truth-sample.
        ref_proposal_mode='data_driven',
    )


def _setup_disk_stream():
    """Returns scenario dict for the production disk-stream scenario."""
    R_ring = 8000.0
    v_circ = 220.0
    sigma_R = 1.0
    sigma_z = 1.0
    sigma_t = 0.1
    G_pc3_msun_myr2 = 0.00449987
    rho_0 = 0.1
    kappa = np.sqrt(2.0) * v_circ / R_ring
    nu = np.sqrt(4.0 * np.pi * G_pc3_msun_myr2 * rho_0)
    width = sigma_R / kappa
    height = sigma_z / nu
    v_sun_peculiar = (-5.0, +5.0, 0.0)
    N_total = 1.0e20
    truth = make_disk_stream_df(N_total, R_ring, sigma_R, sigma_z, sigma_t, v_circ, width, height, v_sun_peculiar)
    sampler = make_disk_stream_sampler(R_ring, sigma_R, sigma_z, sigma_t, v_circ, width, height, v_sun_peculiar)
    # High-precision reference rate via IS on truth with the geometry-aware
    # proposal (stream's local bulk velocity in Sun's frame + scenario sigma).
    v_bulk_sun = -np.asarray(v_sun_peculiar)
    proposal_cov = np.diag([sigma_R ** 2, sigma_t ** 2, sigma_z ** 2]) * 4.0
    out_ref = rate_sphere_importance(truth, v_bulk_sun, proposal_cov, Nboot=NBOOT, fac=1.0, rng=np.random.default_rng(101))
    R_an = float(np.mean(np.asarray(out_ref[0])))
    return dict(
        label=r"disk_stream ($R=8$ kpc, $v_{\rm circ}=220$, $\sigma_t=0.1$)",
        short="disk_stream",
        sampler=sampler,
        df=truth,
        fac=N_total,
        R_an=R_an,
        xloc=(0.0, 0.0, 0.0),
        vloc=(0.0, 0.0, 0.0),
        # Sky reference: use the geometry-aware fixed proposal that gave us
        # the high-precision R_an. Data-driven would also work but require an
        # extra big sample.
        ref_proposal_mode='fixed',
        ref_proposal_mean=v_bulk_sun,
        ref_proposal_cov=proposal_cov,
    )


def _scenario_by_name(name):
    """Picklable scenario lookup so workers can reconstruct callables."""
    if name == "isotropic":
        return _setup_isotropic()
    if name == "disk_stream":
        return _setup_disk_stream()
    raise ValueError(f"unknown scenario {name!r}")


def _compute_reference_sky(sc, ref_seed=12345, n_ref=20000):
    """High-precision binned reference sky histogram for total-variation
    distance. Uses the scenario's ref_proposal config:
      - 'data_driven': draw n_ref samples from truth, build proposal from
                       those, run rate_sphere_importance on truth.
      - 'fixed': use the precomputed geometry-aware proposal (e.g.
                 disk_stream's v_bulk_sun, diag(sigma^2)*4).
    Returns dict {'sky': (N_CT_BINS, N_PHI_BINS) PDF, 'R_an_ref': float}.
    """
    if sc['ref_proposal_mode'] == 'fixed':
        v_mean = sc['ref_proposal_mean']
        v_cov = sc['ref_proposal_cov']
    else:
        rng = np.random.default_rng(ref_seed)
        ref_coords = sc['sampler'](rng, n_ref)
        v_mean, v_cov = make_data_driven_is_proposal(ref_coords, xloc=sc['xloc'])
    out = rate_sphere_importance(sc['df'], v_mean, v_cov, xloc=sc['xloc'], vloc=sc['vloc'], Nboot=REF_NBOOT, fac=1.0, qmaxAU=QMAX_AU, r=R_SPHERE, rng=np.random.default_rng(ref_seed + 1))
    weights, _vee, ct, phi, _ecc, _tc, _vinf = out
    weights = np.asarray(weights)
    ct = np.asarray(ct)
    phi = np.asarray(phi)
    H, _, _ = np.histogram2d(ct, phi, bins=[CT_EDGES, PHI_EDGES], weights=weights)
    total = float(H.sum())
    sky_pdf = H / total if total > 0 else H
    return {'sky': sky_pdf, 'R_an_ref': float(np.mean(weights))}


def _sky_tv_from_rate_output(rate_out, ref_sky):
    """Compute total-variation distance between observed sky PDF (from the
    rate_sphere_importance output) and the reference sky PDF.

    Returns (tv, obs_sky_pdf) - caller may discard the second element if not
    needed for diagnostic plots.
    """
    _w, _vee, ct, phi, _ecc, _tc, _vinf = rate_out
    H, _, _ = np.histogram2d(np.asarray(ct), np.asarray(phi), bins=[CT_EDGES, PHI_EDGES], weights=np.asarray(_w))
    total = float(H.sum())
    obs = H / total if total > 0 else H
    tv = 0.5 * float(np.sum(np.abs(obs - ref_sky)))
    return tv, obs


def _build_cv_adaptive_production(coords, seed, rr_method='ISE'):
    """Production cvAdaptiveKDE - thin wrapper around `cracked.make_production_cv_kde`.

    All grid/hyperparameter choices live in the factory; this wrapper exists
    only to fix `ncovfacs=15` (the convergence test runs a finer covfac grid
    than the deployed `quasi_one_sim` does) and to forward the test-harness
    `seed` and stdout-suppression preference. Anything else (covalpha grid,
    shrinkage grid, scalings, ROI throttle, stability_lambda) must NOT be
    overridden here - drift between this and production was the failure mode
    that motivated the factory in the first place.
    """
    # R_SPHERE matches the encounter radius used by run_rate_sphere in
    # test_rate_sphere_analytic - must match here or the floor mechanism
    # silently degrades (see make_production_cv_kde docstring).
    from test_rate_sphere_analytic import R_SPHERE
    return make_production_cv_kde(
        coords,
        xloc=(0.0, 0.0, 0.0),
        vloc=(0.0, 0.0, 0.0),
        encounter_radius_pc=R_SPHERE,
        ncovfacs=15,             # finer grid than the deployed default of 11
        rr_method=rr_method,
        random_state=seed,
        suppress_stdout=True,
    )


def _extract_cv_grid(cv):
    """Snapshot the full CV grid scores + axes + anchors needed to re-pick
    under alternative covfac/covalpha/shrinkage restrictions post-hoc. Kept
    lightweight (~15 KB per trial) so cache size stays manageable.
    """
    # Aggregate per-fold N_eff to median across folds; matches the floor logic.
    if cv.neffs is not None and np.any(np.isfinite(cv.neffs)):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            neff_med_grid = np.nanmedian(cv.neffs, axis=-1)
    else:
        neff_med_grid = None
    return {
        'avg_scores': cv.avg_scores.copy(),     # (ncf, nca, nn, nvf, nsh, nsc)
        'neff_med_grid': (neff_med_grid.copy()
                          if neff_med_grid is not None else None),
        'covfacs_user': cv.covfacs_user.copy(),
        'covalphas_user': cv.covalphas_user.copy(),
        'shrinkages': np.asarray(cv._shrinkages).copy(),
        'scalings_labels': list(cv.scalings_labels),
        'natural_covfacs_per_scaling': cv.natural_covfacs_per_scaling.copy(),
        'natural_covalpha': float(cv.natural_covalpha),
        'asymcorr': float(cv._asymcorr),
        'neff_floor': cv.neff_floor,
        'rate_best': tuple(int(x) for x in cv.rate_best),
        'shape_best': tuple(int(x) for x in cv.shape_best),
    }


def repick_from_grid(cv_grid, covfac_range=None, covalpha_range=None, shrinkage_range=None, neff_floor=None):
    """Re-pick (covfac_idx, covalpha_idx, nn_idx, vfac_idx, shrinkage_idx,
    scaling_idx) from a saved cv_grid under arbitrary axis restrictions.
    Returns (rate_pick_dict, shape_pick_dict) - each with idx tuple,
    user-facing covfac/covalpha/shrinkage/scaling, and avg_score at the pick.
    Use None for any range to leave that axis unrestricted.
    """
    scores = cv_grid['avg_scores'].copy()
    covfacs_user = cv_grid['covfacs_user']
    covalphas_user = cv_grid['covalphas_user']
    shrinkages = cv_grid['shrinkages']
    neff_med = cv_grid['neff_med_grid']
    floor = neff_floor if neff_floor is not None else cv_grid['neff_floor']

    def _mask_axis(arr, axis_vals, lo_hi, axis):
        if lo_hi is None:
            return arr
        lo, hi = lo_hi
        keep = (axis_vals >= lo) & (axis_vals <= hi)
        if not np.any(keep):
            return arr
        kill = ~keep
        # Broadcast kill mask along the right axis.
        idx = [slice(None)] * arr.ndim
        idx[axis] = kill
        arr[tuple(idx)] = -np.inf
        return arr

    scores = _mask_axis(scores, np.log10(covfacs_user), covfac_range, 0)
    scores = _mask_axis(scores, covalphas_user, covalpha_range, 1)
    scores = _mask_axis(scores, shrinkages, shrinkage_range, 4)
    rate_best_idx = np.unravel_index(np.nanargmax(scores), scores.shape)

    if neff_med is not None and floor is not None and floor > 0:
        eligible = neff_med >= floor
        shape_scores = np.where(eligible, scores, -np.inf)
        if np.any(np.isfinite(shape_scores)):
            shape_best_idx = np.unravel_index(np.nanargmax(shape_scores), shape_scores.shape)
        else:
            shape_best_idx = rate_best_idx
    else:
        shape_best_idx = rate_best_idx

    def _describe(idx):
        return {
            'idx': tuple(int(x) for x in idx),
            'covfac_user': float(covfacs_user[idx[0]]),
            'covalpha_user': float(covalphas_user[idx[1]]),
            'shrinkage': float(shrinkages[idx[4]]),
            'scaling': str(cv_grid['scalings_labels'][idx[5]]),
            'avg_score': float(scores[idx]),
            'natural_covfac_at_pick':
                float(cv_grid['natural_covfacs_per_scaling'][idx[5]]),
        }

    return _describe(rate_best_idx), _describe(shape_best_idx)


def _eval_rate_is(kde, coords, fac, xloc, vloc, rate_seed):
    """Rate via importance-sampled rate_sphere_importance with data-driven
    proposal. Returns (R_hat, full_rate_output_tuple). The full output is
    used downstream for sky-TV computation.
    """
    v_mean, v_cov = make_data_driven_is_proposal(coords, xloc=xloc)
    out = rate_sphere_importance(kde, v_mean, v_cov, xloc=xloc, vloc=vloc, Nboot=NBOOT, fac=fac, qmaxAU=QMAX_AU, r=R_SPHERE, rng=np.random.default_rng(rate_seed))
    R = float(np.mean(np.asarray(out[0])))
    return R, out


def _eval_density_bias(kde, df, sampler, fac, n_eval, bias_seed):
    """Returns (median, std) of log10(fhat*fac / f_truth) at points drawn from
    truth - median is the "where's the peak" bias measure; std is the
    "how scattered are the log ratios" diagnostic.
    """
    rng = np.random.default_rng(bias_seed)
    eval_coords = sampler(rng, n_eval)
    fhat = np.atleast_1d(kde(eval_coords)) * fac
    f_true = df(eval_coords)
    log_ratio = np.log10(np.maximum(fhat, 1e-30) / np.maximum(f_true, 1e-30))
    return float(np.median(log_ratio)), float(np.std(log_ratio, ddof=1))


def _kde_pick_metadata(kde, cv=None, which='rate'):
    """Extract pick attributes from a fitted KDE (or its parent cvAdaptive).
    Returns a flat dict with whatever attrs are present. For cvAdaptive picks,
    also stores user-facing covfac/covalpha (covfac=1 ~ silverman*|Sigma|^(1/d),
    covalpha=0 ~ Abramson volume-equalize -1/d).
    """
    meta = {}
    for attr in ('covfac_overall', 'covalpha_overall', 'shrinkage'):
        if hasattr(kde, attr):
            try:
                meta[attr] = float(getattr(kde, attr))
            except (TypeError, ValueError):
                pass
    if hasattr(kde, 'bw'):
        try:
            meta['bw'] = float(kde.bw)
        except (TypeError, ValueError):
            pass
    if cv is not None:
        # Per-scaling natural anchor + asymcorr backout (2026-05-20 fix).
        # 2026-05-21 fix: look up natural via the kde's actual `scales`,
        # NOT via cv.rate_best / cv.shape_best - pick_for_dim can route to
        # a different idx when the floor masks the "best" out, in which case
        # rate_best's natural would be wrong.
        asymcorr = getattr(cv, '_asymcorr', 1.0)
        actual_sc_idx = None
        if (hasattr(cv, 'natural_covfacs_per_scaling') and
                hasattr(kde, 'scales') and
                hasattr(cv, '_resolved_scalings_list')):
            # Identify which scaling option this kde was built on by
            # matching its scale vector.
            for sc_i, sc_vec in enumerate(cv._resolved_scalings_list):
                if np.allclose(np.asarray(kde.scales), np.asarray(sc_vec), atol=0.0, rtol=1e-10):
                    actual_sc_idx = sc_i
                    break
        if actual_sc_idx is not None:
            nat_cf = float(cv.natural_covfacs_per_scaling[actual_sc_idx])
            scaling_label = str(cv.scalings_labels[actual_sc_idx])
        else:
            # Fall back to rate_best/shape_best lookup (back-compat with old
            # caches and any kde whose scales don't match any grid entry).
            if which == 'rate':
                nat_cf = getattr(cv, 'natural_covfac_rate', getattr(cv, 'natural_covfac', None))
            else:
                nat_cf = getattr(cv, 'natural_covfac_shape', getattr(cv, 'natural_covfac', None))
            scaling_label = None
            if hasattr(cv, 'scalings_labels'):
                try:
                    best_idx = (cv.rate_best if which == 'rate'
                                else cv.shape_best)
                    scaling_label = str(cv.scalings_labels[best_idx[5]])
                except (AttributeError, IndexError):
                    pass
        if nat_cf is not None and 'covfac_overall' in meta:
            meta['covfac_user'] = meta['covfac_overall'] / (asymcorr * nat_cf)
            meta['natural_covfac'] = float(nat_cf)
        if hasattr(cv, 'natural_covalpha') and 'covalpha_overall' in meta:
            meta['covalpha_user'] = meta['covalpha_overall'] - cv.natural_covalpha
            meta['natural_covalpha'] = float(cv.natural_covalpha)
        if scaling_label is not None:
            meta['scaling'] = scaling_label
    return meta


def _save_results_cache(results, scenarios, Ns, n_trials, refs=None, path=CACHE_PATH):
    """Pickle the per-trial results so we can re-plot without re-running."""
    cache = {
        'results': results,
        'scenarios': [{'label': sc['label'], 'short': sc['short'],
                       'R_an': sc['R_an']} for sc in scenarios],
        'Ns': list(Ns),
        'n_trials': n_trials,
        'refs': refs or {},
    }
    with open(path, 'wb') as f:
        pickle.dump(cache, f)
    print(f"  cached results to {path}")


def _plot_convergence_from_results(results, scenarios_meta, Ns, n_trials, out_path="convergence_rate_vs_N.pdf"):
    """4 rows x n_scenarios cols: rate, bias-median, bias-std, sky-TV.
    Each panel overlays scipy_kde and cvAdaptive [rate-pick].
    """
    n_sc = len(scenarios_meta)
    fig, axes = plt.subplots(4, n_sc, figsize=(5.5 * n_sc, 14.0), squeeze=False)
    Ns_arr = np.asarray(Ns, dtype=float)
    xlim = (Ns_arr[0] * 0.8, Ns_arr[-1] * 1.25)

    def _stats(arr_2d):
        return arr_2d.mean(axis=1), arr_2d.std(axis=1, ddof=1)

    def _scatter_and_line(ax, arr, label, color, log_y=False):
        for ni, N in enumerate(Ns):
            (ax.loglog if log_y else ax.plot)([N] * n_trials, arr[ni], "o", color=color, ms=3, alpha=0.25)
        m, s = _stats(arr)
        ax.errorbar(Ns_arr, m, yerr=s, fmt="o-", capsize=4, color=color, label=label)
        return m, s

    # Pretty scenario titles - only set on the top row; downstream rows
    # share the column so the title applies via vertical alignment.
    _short_title = {'isotropic': 'Isotropic Maxwellian',
                    'disk_stream': 'Disk Stream'}
    for i, sc_meta in enumerate(scenarios_meta):
        short = sc_meta['short']
        title = _short_title.get(short, short.replace('_', ' ').title())
        ax_r = axes[0, i]
        ax_bm = axes[1, i]
        ax_bs = axes[2, i]
        ax_tv = axes[3, i]
        scipy_res = results[(short, "scipy_kde")]
        cv_res = results[(short, "cvAdaptive")]
        nf_res = results.get((short, "NormalizingFlow"), None)

        # -- row 0: rate --
        m_sr, _ = _scatter_and_line(ax_r, scipy_res['rel_err'], "scipy_kde", "C1", log_y=True)
        m_cr, _ = _scatter_and_line(ax_r, cv_res['rel_err'], "cvAdaptive [rate-pick]", "C0", log_y=True)
        rel_all = [m_sr, m_cr]
        if nf_res is not None:
            m_nf, _ = _scatter_and_line(ax_r, nf_res['rel_err'], "NormalizingFlow (MAF)", "C3", log_y=True)
            rel_all.append(m_nf)
        rel_all = np.concatenate(rel_all)
        ax_r.set_xlim(xlim)
        ax_r.set_ylim(max(1e-3, rel_all.min() / 1.5), min(5.0, rel_all.max() * 2.0))
        ax_r.set_xlabel("N"); ax_r.set_ylabel(r"$|R/R_{\rm an} - 1|$")
        ax_r.set_title(title)
        ax_r.grid(True, which="both", alpha=0.3)
        ax_r.legend(fontsize=9)

        # -- row 1: bias median --
        # Plot |median log10(fhat/f_truth)| on a log y-axis so "lower is better"
        # is consistent with the other rows. (median bias can sign-flip and
        # asymmetry isn't the diagnostic here - magnitude of departure from 0 is.)
        _scatter_and_line(ax_bm, np.abs(scipy_res['bias_median_rate_pick']), "scipy_kde", "C1", log_y=True)
        _scatter_and_line(ax_bm, np.abs(cv_res['bias_median_rate_pick']), "cvAdaptive [rate-pick]", "C0", log_y=True)
        if nf_res is not None:
            _scatter_and_line(ax_bm, np.abs(nf_res['bias_median_rate_pick']), "NormalizingFlow (MAF)", "C3", log_y=True)
        ax_bm.set_xlim(xlim)
        ax_bm.set_xlabel("N")
        ax_bm.set_ylabel(r"$|\,\mathrm{median}\,\log_{10}(\hat f / f_{\rm truth})\,|$")
        ax_bm.grid(True, which="both", alpha=0.3)
        ax_bm.legend(fontsize=9)

        # -- row 2: bias std --
        _scatter_and_line(ax_bs, scipy_res['bias_std_rate_pick'], "scipy_kde", "C1", log_y=True)
        _scatter_and_line(ax_bs, cv_res['bias_std_rate_pick'], "cvAdaptive [rate-pick]", "C0", log_y=True)
        if nf_res is not None:
            _scatter_and_line(ax_bs, nf_res['bias_std_rate_pick'], "NormalizingFlow (MAF)", "C3", log_y=True)
        ax_bs.set_xlim(xlim)
        ax_bs.set_xlabel("N")
        ax_bs.set_ylabel(r"std $\log_{10}(\hat f / f_{\rm truth})$")
        ax_bs.grid(True, which="both", alpha=0.3)
        ax_bs.legend(fontsize=9)

        # -- row 3: sky TV --
        _scatter_and_line(ax_tv, scipy_res['sky_tv_rate_pick'], "scipy_kde", "C1", log_y=True)
        _scatter_and_line(ax_tv, cv_res['sky_tv_rate_pick'], "cvAdaptive [rate-pick]", "C0", log_y=True)
        # d_m=5 pick - the paper-aligned sky-map pick (smoother kernel for
        # shape diagnostics). For scipy_kde this equals the rate-pick line.
        if 'sky_tv_d5_pick' in cv_res:
            _scatter_and_line(ax_tv, cv_res['sky_tv_d5_pick'], r"cvAdaptive [d$_m$=5 sky-pick]", "C2", log_y=True)
        if nf_res is not None:
            _scatter_and_line(ax_tv, nf_res['sky_tv_rate_pick'], "NormalizingFlow (MAF)", "C3", log_y=True)
        ax_tv.set_xlim(xlim)
        ax_tv.set_xlabel("N")
        ax_tv.set_ylabel(r"$\mathrm{TV}(\mathrm{sky}_{\rm KDE}, \mathrm{sky}_{\rm ref})$")
        ax_tv.grid(True, which="both", alpha=0.3)
        ax_tv.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _plot_sky_maps_per_N_from_results(results, scenarios_meta, refs, Ns, out_path_template="convergence_sky_maps_{short}.pdf"):
    """Per scenario: one figure with rows = N values, cols = 4 estimators:
    (analytic reference, scipy_kde, cvAdaptive [rate-pick], cvAdaptive [d_m=5]).
    Each cell shows the **median-TV trial's** rate-weighted sky PDF as a
    (costheta, phi) heatmap - picking the trial whose sky_tv is closest to the
    median for that (N, estimator) cell. Trial-AVERAGING was misleading
    because trials pick wildly different hyperparameters (some `unit`, some
    `narrow`, etc.) and the average smears their distinct per-trial
    pathologies into a smooth-looking artifact.

    Color scale is shared across all panels in a given scenario (LogNorm),
    using the reference PDF's range. Skips gracefully if the cache doesn't
    have `sky_hist_*` arrays (older runs).
    """
    from matplotlib.colors import LogNorm

    col_specs = [
        ("analytic", None,           None,                  None,                "analytic DF"),
        ("scipy",    "scipy_kde",    "sky_hist_rate_pick",  "sky_tv_rate_pick",  "scipy_kde"),
        ("cv_rate",  "cvAdaptive",   "sky_hist_rate_pick",  "sky_tv_rate_pick",  "cvAdaptive [rate-pick]"),
        ("cv_d5",    "cvAdaptive",   "sky_hist_d5_pick",    "sky_tv_d5_pick",    r"cvAdaptive [d$_m$=5]"),
    ]

    for sc_meta in scenarios_meta:
        short = sc_meta['short']
        ref_sky = (refs.get(short, {}) or {}).get('sky') if refs else None
        if ref_sky is None:
            print(f"  (skipping sky-map figure for {short}: no ref_sky in cache)")
            continue
        # Make sure the cv_rate block has sky_hist arrays - if it doesn't, the
        # cache is from a pre-2026-05-21 run that didn't capture histograms.
        cv_block = results.get((short, "cvAdaptive"), {})
        if 'sky_hist_rate_pick' not in cv_block:
            print(f"  (skipping sky-map figure for {short}: cache has no "
                  f"sky_hist arrays - re-run to capture them)")
            return

        n_rows = len(Ns)
        n_cols = len(col_specs)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 1.9 * n_rows), squeeze=False)

        # Build shared color norm from the reference sky's range (positive
        # values only, since LogNorm).
        pos_vals = ref_sky[ref_sky > 0]
        if pos_vals.size == 0:
            vmin = 1e-6; vmax = 1.0
        else:
            vmin = max(pos_vals.min() / 5.0, 1e-6)
            vmax = pos_vals.max() * 5.0
        norm = LogNorm(vmin=vmin, vmax=vmax)

        for ni, N in enumerate(Ns):
            for ci, (key, est, hist_key, tv_key, label) in enumerate(col_specs):
                ax = axes[ni, ci]
                if key == "analytic":
                    img = ref_sky
                    cell_label = label
                else:
                    blk = results.get((short, est), {})
                    arr = blk.get(hist_key, None)
                    tv_arr = blk.get(tv_key, None)
                    if arr is None or arr.shape[0] <= ni:
                        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", fontsize=8)
                        ax.set_xticks([]); ax.set_yticks([])
                        continue
                    # Pick the median-TV trial: argmin |tv_t - median(tv)|.
                    # Robust to NaNs (any all-NaN trial gets +inf, so it's
                    # not selected unless every trial is NaN).
                    tv_vals = (tv_arr[ni] if tv_arr is not None
                               else np.zeros(arr.shape[1]))
                    finite = np.isfinite(tv_vals)
                    if finite.any():
                        med_tv = float(np.nanmedian(tv_vals))
                        dist = np.where(finite, np.abs(tv_vals - med_tv), np.inf)
                        t_pick = int(np.nanargmin(dist))
                    else:
                        t_pick = 0
                    img = arr[ni, t_pick]
                    if tv_arr is not None and finite.any():
                        cell_label = f"t={t_pick}, TV={tv_vals[t_pick]:.2f}"
                    else:
                        cell_label = f"t={t_pick}"
                im = ax.imshow(img, origin="lower", aspect="auto", extent=[-np.pi, np.pi, -1, 1], cmap="viridis", norm=norm)
                # Per-cell mini-annotation (top-right) showing which trial.
                if key != "analytic":
                    ax.text(0.98, 0.97, cell_label, transform=ax.transAxes, ha="right", va="top", fontsize=6, color="white", bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.4, edgecolor="none"))
                if ni == 0:
                    ax.set_title(label, fontsize=9)
                if ci == 0:
                    ax.set_ylabel(f"N={N}\n" + r"$\cos\theta$", fontsize=8)
                else:
                    ax.set_yticklabels([])
                if ni == n_rows - 1:
                    ax.set_xlabel(r"$\phi$", fontsize=8)
                else:
                    ax.set_xticklabels([])
                ax.tick_params(labelsize=7)
        fig.suptitle(f"{sc_meta['label']}: sky-map dr/dOmega "
                      f"(median-TV trial per cell)",
                      fontsize=11)
        fig.tight_layout(rect=[0, 0, 0.97, 0.96])
        cbar_ax = fig.add_axes([0.97, 0.05, 0.012, 0.85])
        fig.colorbar(im, cax=cbar_ax)
        out_path = out_path_template.format(short=short)
        fig.savefig(out_path)
        plt.close(fig)
        print(f"  wrote {out_path}")


def _plot_cv_picks_from_results(results, scenarios_meta, Ns, n_trials, out_path="convergence_cv_picks.pdf"):
    """Auxiliary plot: cvAdaptive's per-trial CV picks vs N
    (covfac, shrinkage, scaling-label distribution).
    """
    n_sc = len(scenarios_meta)
    fig, axes = plt.subplots(2, n_sc, figsize=(5.5 * n_sc, 7.0), squeeze=False)
    Ns_arr = np.asarray(Ns, dtype=float)
    xlim = (Ns_arr[0] * 0.8, Ns_arr[-1] * 1.25)
    for i, sc_meta in enumerate(scenarios_meta):
        short = sc_meta['short']
        cv_res = results[(short, "cvAdaptive")]
        picks = cv_res.get('picks_rate', None)
        if picks is None:
            continue
        # picks is list[N_index][trial] -> dict. Prefer user-facing
        # covfac_user (covfac=1 = silverman*|Sigma|^(1/d)) when present.
        has_user = picks[0][0] is not None and 'covfac_user' in picks[0][0]
        cf_key = 'covfac_user' if has_user else 'covfac_overall'
        covfacs = np.array([[p.get(cf_key, np.nan) if p else np.nan
                              for p in picks[ni]]
                             for ni, _ in enumerate(Ns)])
        shrinks = np.array([[p.get('shrinkage', np.nan) if p else np.nan
                              for p in picks[ni]]
                             for ni, _ in enumerate(Ns)])
        ax_cf = axes[0, i]
        ax_sh = axes[1, i]
        for ni, N in enumerate(Ns):
            ax_cf.semilogx([N] * n_trials, covfacs[ni], "o", color="C0", alpha=0.4)
            ax_sh.semilogx([N] * n_trials, shrinks[ni], "o", color="C0", alpha=0.4)
        cf_med = np.nanmedian(covfacs, axis=1)
        sh_med = np.nanmedian(shrinks, axis=1)
        ax_cf.semilogx(Ns_arr, cf_med, "-", color="C0", label="median across trials")
        ax_sh.semilogx(Ns_arr, sh_med, "-", color="C0", label="median across trials")
        ax_cf.axhline(1.0, color="k", lw=0.6, alpha=0.4, label="natural (covfac=1)" if has_user else None)
        ax_cf.set_yscale("log")
        ax_cf.set_xlim(xlim); ax_sh.set_xlim(xlim)
        ax_cf.set_xlabel("N")
        ax_cf.set_ylabel("covfac (user; 1 = silverman*|Sigma|^(1/d))" if has_user else "covfac (raw)")
        ax_sh.set_xlabel("N"); ax_sh.set_ylabel("shrinkage")
        ax_cf.set_title(f"{sc_meta['label']}: CV picks - covfac")
        ax_sh.set_title(f"{sc_meta['label']}: CV picks - shrinkage")
        ax_cf.grid(True, which="both", alpha=0.3)
        ax_sh.grid(True, which="both", alpha=0.3)
        ax_cf.legend(fontsize=9); ax_sh.legend(fontsize=9)

        # Scaling-label distribution
        scalings = [[p.get('scaling', 'n/a') for p in picks[ni]]
                    for ni, _ in enumerate(Ns)]
        # Add a small text annotation per-N inside the covfac panel
        for ni, N in enumerate(Ns):
            unique, counts = np.unique(scalings[ni], return_counts=True)
            tag = "/".join(f"{u}:{c}" for u, c in zip(unique, counts))
            ax_cf.annotate(tag, xy=(N, ax_cf.get_ylim()[1] * 0.9), xytext=(0, 0), textcoords='offset points', ha='center', va='top', fontsize=6, rotation=90, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def replot_convergence_from_cache(cache_path=CACHE_PATH, out_path="convergence_rate_vs_N.pdf", picks_path="convergence_cv_picks.pdf"):
    """Utility for iterating on the plot without re-running the sims."""
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    _plot_convergence_from_results(cache['results'], cache['scenarios'], cache['Ns'], cache['n_trials'], out_path=out_path)
    try:
        _plot_cv_picks_from_results(cache['results'], cache['scenarios'], cache['Ns'], cache['n_trials'], out_path=picks_path)
    except Exception as e:
        print(f"  (skipping CV picks plot: {type(e).__name__}: {e})")
    try:
        _plot_sky_maps_per_N_from_results(cache['results'], cache['scenarios'], cache.get('refs', {}), cache['Ns'])
    except Exception as e:
        print(f"  (skipping per-N sky-map plot: {type(e).__name__}: {e})")


def _run_one_trial(task):
    """One (scenario, estimator, N, trial) trial. Picklable; called from
    workers in run_parallel. Returns a result dict suitable for accumulating
    into the cache structure.

    task keys: scenario, est, N, t, seed_base, R_an (precomputed),
               ref_sky (precomputed dict).
    """
    scenario_name = task['scenario']
    est_name = task['est']
    N = task['N']
    t = task['t']
    seed_base = task['seed_base']
    R_an = task['R_an']
    ref_sky = task['ref_sky']

    sc = _scenario_by_name(scenario_name)
    # Derive deterministic seeds from (scenario, N, t).
    ni_hash = (hash(scenario_name) & 0xFFFF) ^ (N * 31)
    data_seed = seed_base + (ni_hash + t) * 7
    rate_seed = seed_base + 1000 + (ni_hash + t) * 11
    bias_seed = seed_base + 2000 + (ni_hash + t) * 13
    cv_seed = seed_base + 5000 + ni_hash + t

    rng_data = np.random.default_rng(data_seed)
    coords = sc['sampler'](rng_data, N)

    if est_name == "scipy_kde":
        kde = gaussianKDEWrapper(coords)
        kde_rate = kde_shape = kde_d5 = kde
        picks_rate = {'bw': float(getattr(kde, 'bw', np.nan))
                      if hasattr(kde, 'bw') else None}
        picks_shape = picks_rate
        cv_grid = None
    elif est_name == "cvAdaptive":
        cv = _build_cv_adaptive_production(coords, seed=cv_seed, rr_method=task.get('rr_method', 'ISE'))
        kde_rate = cv.pick_for_dim(0)
        kde_shape = cv.pick_for_dim(6)
        kde_d5 = cv.pick_for_dim(5)   # paper-aligned sky-map pick
        picks_rate = _kde_pick_metadata(kde_rate, cv=cv, which='rate')
        picks_shape = _kde_pick_metadata(kde_shape, cv=cv, which='shape')
        cv_grid = _extract_cv_grid(cv)
    elif est_name == "NormalizingFlow":
        # Buckley et al. 2022-aligned MAF, defaults from cracked.normalizing_flow.
        # No CV - architecture/training are fixed by the literature recipe.
        # kde_rate / kde_shape / kde_d5 all the same since NF has no per-pick
        # bandwidth knob (the flow IS the density estimator).
        from cracked.normalizing_flow import NormalizingFlowKDE
        kde = NormalizingFlowKDE(coords, random_state=cv_seed)
        kde_rate = kde_shape = kde_d5 = kde
        picks_rate = {'best_val_nll': float(kde.best_val_nll),
                      'n_layers': 5, 'hidden_units': 48, 'activation': 'gelu'}
        picks_shape = picks_rate
        cv_grid = None
    else:
        raise ValueError(f"unknown estimator {est_name!r}")

    R, rate_out = _eval_rate_is(kde_rate, coords, sc['fac'], sc['xloc'], sc['vloc'], rate_seed)
    bias_m_rp, bias_s_rp = _eval_density_bias(kde_rate, sc['df'], sc['sampler'], sc['fac'], 1000, bias_seed)
    if kde_shape is kde_rate:
        bias_m_sp, bias_s_sp = bias_m_rp, bias_s_rp
    else:
        bias_m_sp, bias_s_sp = _eval_density_bias(kde_shape, sc['df'], sc['sampler'], sc['fac'], 1000, bias_seed)
    sky_tv_rate, sky_hist_rate = _sky_tv_from_rate_output(rate_out, ref_sky)
    # Sky-TV from the paper-aligned d_m=5 pick. For scipy_kde, kde_d5 == kde_rate
    # -> identical sky_tv. For cvAdaptive, this is a separate rate_sphere_importance
    # pass on the smoother shape pick - the diagnostic that the paper's multi-pick
    # scheme assigns to sky-map outputs.
    if kde_d5 is kde_rate:
        sky_tv_d5 = sky_tv_rate
        sky_hist_d5 = sky_hist_rate
    else:
        _, rate_out_d5 = _eval_rate_is(kde_d5, coords, sc['fac'], sc['xloc'], sc['vloc'], rate_seed + 10001)
        sky_tv_d5, sky_hist_d5 = _sky_tv_from_rate_output(rate_out_d5, ref_sky)

    return {
        'scenario': scenario_name,
        'est': est_name,
        'N': N,
        't': t,
        'rel_err': abs(R - R_an) / R_an,
        'bias_median_rate_pick': bias_m_rp,
        'bias_std_rate_pick': bias_s_rp,
        'bias_median_shape_pick': bias_m_sp,
        'bias_std_shape_pick': bias_s_sp,
        'sky_tv_rate_pick': sky_tv_rate,
        'sky_tv_d5_pick': sky_tv_d5,
        'sky_hist_rate_pick': sky_hist_rate,
        'sky_hist_d5_pick': sky_hist_d5,
        'picks_rate': picks_rate,
        'picks_shape': picks_shape,
        'cv_grid': cv_grid,
        'R_kde': R,
    }


def _empty_result_block(n_Ns, n_trials):
    """Allocate the per-(scenario, estimator) result arrays."""
    return {
        'rel_err': np.full((n_Ns, n_trials), np.nan),
        'bias_median_rate_pick': np.full((n_Ns, n_trials), np.nan),
        'bias_std_rate_pick': np.full((n_Ns, n_trials), np.nan),
        'bias_median_shape_pick': np.full((n_Ns, n_trials), np.nan),
        'bias_std_shape_pick': np.full((n_Ns, n_trials), np.nan),
        'sky_tv_rate_pick': np.full((n_Ns, n_trials), np.nan),
        'sky_tv_d5_pick': np.full((n_Ns, n_trials), np.nan),
        'sky_hist_rate_pick': np.full((n_Ns, n_trials, N_CT_BINS, N_PHI_BINS), np.nan),
        'sky_hist_d5_pick': np.full((n_Ns, n_trials, N_CT_BINS, N_PHI_BINS), np.nan),
        'picks_rate': [[None] * n_trials for _ in range(n_Ns)],
        'picks_shape': [[None] * n_trials for _ in range(n_Ns)],
        'cv_grid': [[None] * n_trials for _ in range(n_Ns)],
    }


def _accumulate_result(results, scenarios_meta, Ns, result):
    """Insert a single _run_one_trial output into the results dict.
    Caller must have pre-allocated via _initialise_results.
    """
    key = (result['scenario'], result['est'])
    ni = Ns.index(result['N'])
    t = result['t']
    blk = results[key]
    blk['rel_err'][ni, t] = result['rel_err']
    blk['bias_median_rate_pick'][ni, t] = result['bias_median_rate_pick']
    blk['bias_std_rate_pick'][ni, t] = result['bias_std_rate_pick']
    blk['bias_median_shape_pick'][ni, t] = result['bias_median_shape_pick']
    blk['bias_std_shape_pick'][ni, t] = result['bias_std_shape_pick']
    blk['sky_tv_rate_pick'][ni, t] = result['sky_tv_rate_pick']
    blk['sky_tv_d5_pick'][ni, t] = result['sky_tv_d5_pick']
    if 'sky_hist_rate_pick' in result and result['sky_hist_rate_pick'] is not None:
        blk['sky_hist_rate_pick'][ni, t] = result['sky_hist_rate_pick']
    if 'sky_hist_d5_pick' in result and result['sky_hist_d5_pick'] is not None:
        blk['sky_hist_d5_pick'][ni, t] = result['sky_hist_d5_pick']
    blk['picks_rate'][ni][t] = result['picks_rate']
    blk['picks_shape'][ni][t] = result['picks_shape']
    blk['cv_grid'][ni][t] = result.get('cv_grid', None)
    # R_an is per-scenario, set from scenarios_meta when finalising
    for sc in scenarios_meta:
        if sc['short'] == result['scenario']:
            blk['R_an'] = sc['R_an']
            break


def _initialise_results(scenarios_meta, estimators, Ns, n_trials):
    results = {}
    for sc in scenarios_meta:
        for est in estimators:
            results[(sc['short'], est)] = _empty_result_block(len(Ns), n_trials)
            results[(sc['short'], est)]['R_an'] = sc['R_an']
    return results


def _precompute_refs(scenario_names):
    """For each scenario, build the scenario dict and compute the reference
    sky histogram + R_an. Heavy, do once in main process.
    """
    refs = {}
    scenarios = []
    for name in scenario_names:
        print(f"  building scenario {name!r}...")
        sc = _scenario_by_name(name)
        print(f"    R_an = {sc['R_an']:.4g}; computing reference sky "
              f"({REF_NBOOT} IS samples)...")
        sky = _compute_reference_sky(sc)
        print(f"    R_an_ref (sky-eval) = {sky['R_an_ref']:.4g}")
        refs[name] = sky
        scenarios.append(sc)
    return scenarios, refs


def run_serial(scenario_names=("isotropic", "disk_stream"), Ns=(125, 250, 500, 1000, 2000, 4000, 8000), n_trials=3, seed_base=7919, estimators=("scipy_kde", "cvAdaptive", "NormalizingFlow"), cache_path=CACHE_PATH, plot=True, rr_method='ISE'):
    """Serial driver. Used by the pytest entry point.

    `rr_method`  in  {'ISE', 'LOO'} is forwarded to the production CV
    factory - bootstrap is the historical default; LOO is selectively useful
    on narrow-feature data (see scripts/test_loo_objective.py results)."""
    Ns = list(Ns)
    estimators = list(estimators)
    scenarios, refs = _precompute_refs(scenario_names)
    scenarios_meta = [{'label': sc['label'], 'short': sc['short'],
                        'R_an': sc['R_an']} for sc in scenarios]
    results = _initialise_results(scenarios_meta, estimators, Ns, n_trials)
    for sc in scenarios:
        print(f"\n  ===== {sc['label']} =====")
        for est in estimators:
            for N in Ns:
                for t in range(n_trials):
                    task = {
                        'scenario': sc['short'], 'est': est, 'N': N, 't': t,
                        'seed_base': seed_base, 'R_an': sc['R_an'],
                        'ref_sky': refs[sc['short']]['sky'],
                        'rr_method': rr_method,
                    }
                    r = _run_one_trial(task)
                    _accumulate_result(results, scenarios_meta, Ns, r)
                # progress summary per (est, N)
                blk = results[(sc['short'], est)]
                ni = Ns.index(N)
                r = blk['rel_err'][ni]
                bm = blk['bias_median_rate_pick'][ni]
                tv = blk['sky_tv_rate_pick'][ni]
                print(f"    [{est}] N={N:>5d}: rel_err "
                      f"{np.nanmean(r):.3f}+/-{np.nanstd(r, ddof=1):.3f}  "
                      f"bias_m {np.nanmean(bm):.3f}  "
                      f"sky_TV {np.nanmean(tv):.3f}")
                # Save incrementally so a long run is resumable-by-inspection.
                _save_results_cache(results, scenarios, Ns, n_trials, refs=refs, path=cache_path)
    if plot:
        _plot_convergence_from_results(results, scenarios_meta, Ns, n_trials)
        _plot_cv_picks_from_results(results, scenarios_meta, Ns, n_trials)
        try:
            _plot_sky_maps_per_N_from_results(results, scenarios_meta, refs, Ns)
        except Exception as e:
            print(f"  (skipping per-N sky-map plot: {type(e).__name__}: {e})")
    return results, scenarios_meta


def run_parallel(scenario_names=("isotropic", "disk_stream"), Ns=(125, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000), n_trials=8, seed_base=7919, max_workers=16, estimators=("scipy_kde", "cvAdaptive", "NormalizingFlow"), cache_path=CACHE_PATH, plot=True, rr_method='ISE'):
    """Parallel driver for the server run. Spawns up to max_workers worker
    processes via ProcessPoolExecutor. Each worker runs one trial. Results
    are accumulated in the main process and the cache is rewritten after
    every completed trial so an interruption keeps partial work.

    Inner KDE parallelism (numpy/BLAS threads etc.) is unaffected - keep
    max_workers low enough to leave cores for it (the user's 50-core box
    with max_workers=16 leaves ~34 cores per worker).

    `rr_method`  in  {'ISE', 'LOO'} forwarded to the production CV
    factory. LOO unanimously prefers the deep-narrow basin on disk_stream
    (3/3 trials in scripts/test_loo_objective.py) but breaks isotropic in
    the opposite direction - opt in only if the run is dominated by
    narrow-feature data.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    Ns = list(Ns)
    estimators = list(estimators)
    scenarios, refs = _precompute_refs(scenario_names)
    scenarios_meta = [{'label': sc['label'], 'short': sc['short'],
                        'R_an': sc['R_an']} for sc in scenarios]
    results = _initialise_results(scenarios_meta, estimators, Ns, n_trials)

    tasks = []
    for sc in scenarios:
        for est in estimators:
            for N in Ns:
                for t in range(n_trials):
                    tasks.append({
                        'scenario': sc['short'], 'est': est, 'N': N, 't': t,
                        'seed_base': seed_base, 'R_an': sc['R_an'],
                        'ref_sky': refs[sc['short']]['sky'],
                        'rr_method': rr_method,
                    })
    total = len(tasks)
    print(f"\n  scheduling {total} trials across {max_workers} workers")
    t_start = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_run_one_trial, t): t for t in tasks}
        for fut in as_completed(futures):
            try:
                r = fut.result()
            except Exception as exc:
                t = futures[fut]
                print(f"  TRIAL FAILED  scenario={t['scenario']} "
                      f"est={t['est']} N={t['N']} t={t['t']}: "
                      f"{type(exc).__name__}: {exc}")
                continue
            _accumulate_result(results, scenarios_meta, Ns, r)
            done += 1
            if done % max(1, total // 50) == 0 or done == total:
                elapsed = time.time() - t_start
                eta = elapsed * (total - done) / max(done, 1)
                print(f"  [{done:>4d}/{total}] {elapsed/60:.1f}m elapsed, "
                      f"~{eta/60:.1f}m remaining")
                _save_results_cache(results, scenarios, Ns, n_trials, refs=refs, path=cache_path)
    # final save + plot
    _save_results_cache(results, scenarios, Ns, n_trials, refs=refs, path=cache_path)
    if plot:
        _plot_convergence_from_results(results, scenarios_meta, Ns, n_trials)
        _plot_cv_picks_from_results(results, scenarios_meta, Ns, n_trials)
        try:
            _plot_sky_maps_per_N_from_results(results, scenarios_meta, refs, Ns)
        except Exception as e:
            print(f"  (skipping per-N sky-map plot: {type(e).__name__}: {e})")
    return results, scenarios_meta


@slow
def test_rate_convergence_vs_N():
    """Production-recipe convergence sweep (serial pytest entry).

    Scenarios: isotropic Maxwellian, disk_stream.
    Estimators: scipy_kde, cvAdaptive (production CV grid + local_pooled).
    Diagnostics: rate error, bias median + std at truth samples, sky-map TV.
    CV picks recorded per trial for diagnostic plotting.

    Cache: convergence_rate_vs_N_cache.pkl (resumable / replottable).
    For the bigger production sweep with 5-10 seeds and N up to 32k, run
    `run_parallel(...)` on a multi-core machine instead.
    """
    results, scenarios_meta = run_serial(scenario_names=("isotropic", "disk_stream"), Ns=(125, 250, 500, 1000, 2000, 4000, 8000), n_trials=3, seed_base=7919)
    # ---- assertions
    for sc in scenarios_meta:
        cv_res = results[(sc['short'], "cvAdaptive")]
        rel_mean = np.nanmean(cv_res['rel_err'], axis=1)
        bias_rp_mean = np.nanmean(cv_res['bias_median_rate_pick'], axis=1)
        assert np.all(np.isfinite(rel_mean)), (
            f"[{sc['short']}] cvAdaptive rate has non-finite values")
        assert abs(bias_rp_mean[-1]) < abs(bias_rp_mean[0]) + 0.5, (
            f"[{sc['short']}] cvAdaptive rate-pick density-bias diverged: "
            f"{bias_rp_mean[0]:.3f} -> {bias_rp_mean[-1]:.3f}")


# 2. RATE_Sphere MC convergence vs Nboot

@slow
def test_rate_sphere_mc_variance_vs_Nboot():
    """Hold the KDE fixed and reseed RATE_Sphere with progressively larger
    Nboot. The estimator std over reseeds should drop as 1/sqrt(Nboot). We
    sweep 4 Nboot values x 10 trials each, plot std vs Nboot on log-log with
    the reference slope (figure: convergence_mc_vs_Nboot.pdf), and assert
    the endpoint-ratio falls in a band around the theoretical value.
    """
    from cracked import RATE_Sphere
    fac = DX ** 3 * N0
    # Use a smaller KDE here than the rate-vs-N test to keep per-call cost
    # low; the assertion is about MC variance not KDE bias.
    rng = np.random.default_rng(7)
    kde, _ = build_kde(rng, 500)
    Nboots = [1000, 3000, 10000, 30000]
    n_trials = 10
    means = []
    stds = []
    all_rs = []
    for Nboot in Nboots:
        rs = []
        for trial in range(n_trials):
            np.random.seed(1000 + Nboot + trial)
            out = RATE_Sphere(kde, [0, 0, 0], [0, 0, 0], qmaxAU=QMAX_AU, Nboot=Nboot, fac=fac, r=R_SPHERE, v0=V0)
            rs.append(float(np.mean(out[0])))
        rs = np.asarray(rs)
        all_rs.append(rs)
        means.append(float(rs.mean()))
        stds.append(float(rs.std(ddof=1)))
        print(f"  [info] Nboot={Nboot:>6d}: mean={means[-1]:.4g} "
              f"std={stds[-1]:.4g}")
    Nb_arr = np.asarray(Nboots, dtype=float)
    stds_arr = np.asarray(stds)

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.loglog(Nb_arr, stds_arr, "o-", color="C0", label="std over reseeds")
    ref = stds_arr[0] * np.sqrt(Nb_arr[0] / Nb_arr)
    ax.loglog(Nb_arr, ref, "k--", lw=1, label=r"$1/\sqrt{N_{\rm boot}}$")
    ax.set_xlabel(r"$N_{\rm boot}$")
    ax.set_ylabel(r"std of $\hat R$ over reseeds")
    ax.set_title(f"RATE_Sphere MC convergence (isotropic, KDE N=500, "
                 f"{n_trials} trials)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("convergence_mc_vs_Nboot.pdf")
    plt.close(fig)

    ratio = stds_arr[-1] / max(stds_arr[0], 1e-300)
    expected = np.sqrt(Nb_arr[0] / Nb_arr[-1])
    print(f"  [info] std ratio (Nboot={Nboots[-1]}/Nboot={Nboots[0]})"
          f" = {ratio:.3f}  (expected 1/sqrt = {expected:.3f})")
    # Allow generous band around the theoretical 0.183 ratio for 30x.
    assert 0.07 < ratio < 0.6, (
        f"MC variance scaling ratio {ratio:.3f} outside [0.07, 0.6]")


# 3. Importance-sampling variance reduction

@slow
def test_importance_sampling_reduces_variance():
    """On a narrow drifting Maxwellian, the data-driven IS proposal should
    have substantially lower variance than uniform RATE_Sphere over reseeds.
    Both should agree on the mean. We sweep 30 trials each and plot the two
    Rhat distributions side-by-side (figure: convergence_is_vs_uniform.pdf).
    """
    from cracked import RATE_Sphere
    sigma_narrow = 0.5
    v_drift = np.array([2.0, 0.0, 0.0])
    rng = np.random.default_rng(33)
    n_samp = 2000
    coords = np.zeros((n_samp, 6))
    coords[:, :3] = rng.uniform(-DX / 2.0, DX / 2.0, size=(n_samp, 3))
    coords[:, 3:] = v_drift[None, :] + rng.standard_normal((n_samp, 3)) * sigma_narrow
    kde = adaptiveKDE(coords, scalings=np.ones(6), nn=50, use_multiprocessing=False)
    fac = DX ** 3 * N0
    v_mean, v_cov = make_data_driven_is_proposal(coords, xloc=(0, 0, 0))
    n_trials = 30
    Nboot = 5000
    rs_uniform, rs_is = [], []
    for trial in range(n_trials):
        np.random.seed(500 + trial)
        out = RATE_Sphere(kde, [0, 0, 0], [0, 0, 0], qmaxAU=QMAX_AU, Nboot=Nboot, fac=fac, r=R_SPHERE, v0=V0)
        rs_uniform.append(float(np.mean(out[0])))
        out_is = rate_sphere_importance(kde, v_mean, v_cov, xloc=(0, 0, 0), vloc=(0, 0, 0), Nboot=Nboot, fac=fac, qmaxAU=QMAX_AU, r=R_SPHERE, rng=np.random.default_rng(600 + trial))
        rs_is.append(float(np.mean(out_is[0])))
    rs_uniform = np.asarray(rs_uniform)
    rs_is = np.asarray(rs_is)
    m_uni, s_uni = float(rs_uniform.mean()), float(rs_uniform.std(ddof=1))
    m_is, s_is = float(rs_is.mean()), float(rs_is.std(ddof=1))
    print(f"  [info] uniform: mean={m_uni:.4g}  std={s_uni:.4g}  "
          f"(n_trials={n_trials})")
    print(f"  [info] IS     : mean={m_is:.4g}  std={s_is:.4g}  "
          f"reduction={s_uni / max(s_is, 1e-30):.2f}x")

    fig, (ax_h, ax_s) = plt.subplots(1, 2, figsize=(9.0, 4.0))
    lo = min(rs_uniform.min(), rs_is.min())
    hi = max(rs_uniform.max(), rs_is.max())
    bins = np.linspace(lo, hi, 14)
    ax_h.hist(rs_uniform, bins=bins, alpha=0.55, color="C0", label=f"uniform (std={s_uni:.2g})")
    ax_h.hist(rs_is, bins=bins, alpha=0.55, color="C1", label=f"IS (std={s_is:.2g})")
    ax_h.axvline(m_uni, color="C0", lw=1, ls="--")
    ax_h.axvline(m_is, color="C1", lw=1, ls="--")
    ax_h.set_xlabel(r"$\hat R$ (per trial)")
    ax_h.set_ylabel("count")
    ax_h.legend()
    ax_h.set_title(f"Rhat over {n_trials} reseeds at Nboot={Nboot}")

    ax_s.errorbar([0, 1], [m_uni, m_is], yerr=[s_uni, s_is], fmt="o", capsize=6, color="k")
    ax_s.set_xticks([0, 1])
    ax_s.set_xticklabels(["uniform", "IS"])
    ax_s.set_ylabel(r"$\hat R$ (mean +/- std)")
    ax_s.set_title(f"variance reduction: {s_uni / max(s_is, 1e-30):.2f}x")
    ax_s.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("convergence_is_vs_uniform.pdf")
    plt.close(fig)

    assert s_is < s_uni, (
        f"IS std ({s_is:.3g}) not smaller than uniform std ({s_uni:.3g})")
    se = np.sqrt(s_uni ** 2 + s_is ** 2) / np.sqrt(n_trials)
    bias = abs(m_uni - m_is)
    assert bias < max(3.0 * se, 0.5 * min(m_uni, m_is)), (
        f"uniform mean {m_uni:.3g} and IS mean {m_is:.3g} disagree by "
        f"{bias:.3g} (>3 SE = {3*se:.3g})")


# 4. cvAdaptiveKDE / cvGaussianKDE seeded reproducibility

def test_cv_adaptive_reproducibility_seeded():
    """Two cvAdaptiveKDE builds with the same data and random_state must pick
    the same (covfac, covalpha, shrinkage, scaling) tuple.
    """
    rng = np.random.default_rng(99)
    coords = sample_uniform_pos_gaussian_v(rng, 800, DX, np.zeros(3), SIGMA)
    cv_a = cvAdaptiveKDE(coords.copy(), nfolds=3, ncovfacs=3, ncovalphas=2, nshrinkages=2, scalings_grid=[None, 'auto'], random_state=2024)
    cv_b = cvAdaptiveKDE(coords.copy(), nfolds=3, ncovfacs=3, ncovalphas=2, nshrinkages=2, scalings_grid=[None, 'auto'], random_state=2024)
    assert cv_a.best == cv_b.best, (
        f"reproducibility: cv_a.best={cv_a.best} cv_b.best={cv_b.best}")
    assert cv_a.rate_best == cv_b.rate_best
    assert cv_a.shape_best == cv_b.shape_best
    pt = coords[0:1]
    assert np.allclose(cv_a.kde(pt), cv_b.kde(pt))


def test_cv_gaussian_reproducibility_seeded():
    """Same expectation for cvGaussianKDE."""
    rng = np.random.default_rng(101)
    coords = sample_uniform_pos_gaussian_v(rng, 800, DX, np.zeros(3), SIGMA)
    cv_a = cvGaussianKDE(coords.copy(), nfolds=3, nbw=5, scalings_grid=[None, 'auto'], random_state=77)
    cv_b = cvGaussianKDE(coords.copy(), nfolds=3, nbw=5, scalings_grid=[None, 'auto'], random_state=77)
    assert cv_a.best == cv_b.best
    pt = coords[0:1]
    assert np.allclose(cv_a(pt), cv_b(pt))


# 5. Pickle round-trip

def test_pickle_roundtrip_adaptive_kde():
    """An adaptiveKDE that has been pickled and unpickled must return
    bit-identical densities at the same evaluation points."""
    rng = np.random.default_rng(17)
    coords = sample_uniform_pos_gaussian_v(rng, 400, DX, np.zeros(3), SIGMA)
    kde = adaptiveKDE(coords, scalings=np.ones(6), nn=30, use_multiprocessing=False)
    pts = rng.standard_normal((20, 6)) * 0.3
    p_before = kde(pts)
    blob = pickle.dumps(kde)
    kde2 = pickle.loads(blob)
    p_after = kde2(pts)
    assert p_before.shape == p_after.shape
    assert np.array_equal(p_before, p_after), (
        f"pickled-then-loaded KDE returns different densities; max abs "
        f"diff {np.max(np.abs(p_before - p_after)):.3g}")


# 6. make_data_driven_is_proposal with K > N

def test_proposal_K_exceeds_N():
    """When K > N, K_eff should clamp to N-1 and the proposal must remain a
    valid (3-vec mean, 3x3 PSD cov)."""
    rng = np.random.default_rng(41)
    n_samp = 50
    coords = sample_uniform_pos_gaussian_v(rng, n_samp, DX, np.zeros(3), SIGMA)
    v_mean, v_cov = make_data_driven_is_proposal(coords, xloc=(0, 0, 0), K=300)
    assert v_mean.shape == (3,)
    assert v_cov.shape == (3, 3)
    # Symmetric and positive-definite (cholesky succeeds).
    assert np.allclose(v_cov, v_cov.T, atol=1e-10)
    np.linalg.cholesky(v_cov)  # raises if not PSD


# 7. Zero-variance axis -> sigma_floor_factor must rescue the proposal

def test_proposal_local_zero_variance_axis_is_invertible():
    """The sigma_floor_factor protects against the *local* (K-NN) subset
    having near-zero spread on some axis, with the global data still
    providing a finite std to set the floor magnitude. We construct a tight
    spatial core of K particles all with v_z=0, plus a wider bulk with
    nonzero v_z, then build the proposal at the origin. The K-NN subset has
    Cov[v_z,v_z]=0; the global std is nonzero; the floor must lift the
    diagonal so the cov is invertible.
    """
    rng = np.random.default_rng(53)
    K = 50
    coords = np.zeros((600, 6))
    # Core: K spatially-near-origin particles with v_z = 0
    coords[:K, :3] = rng.standard_normal((K, 3)) * 0.05
    coords[:K, 3:5] = rng.standard_normal((K, 2)) * SIGMA
    coords[:K, 5] = 0.0
    # Bulk: 550 farther-out particles with all 3 velocity axes nonzero
    coords[K:, :3] = rng.uniform(-DX / 2.0, DX / 2.0, size=(550, 3)) + 3.0
    coords[K:, 3:] = rng.standard_normal((550, 3)) * SIGMA
    v_mean, v_cov = make_data_driven_is_proposal(coords, xloc=(0, 0, 0), K=K)
    assert v_cov[2, 2] > 0.0, (
        f"sigma_floor_factor did not lift the local zero-variance axis; "
        f"v_cov[2,2]={v_cov[2,2]:.3g}")
    np.linalg.cholesky(v_cov)
    # And the proposal must be usable as a sampling distribution.
    draws = np.random.default_rng(0).multivariate_normal(v_mean, v_cov, size=10)
    assert draws.shape == (10, 3)
    assert np.all(np.isfinite(draws))


# Standalone runner

def _is_test(name, fn):
    return name.startswith("test_") and callable(fn)


def main(argv):
    skip_slow = "--no-slow" in argv
    name_filter = None
    if "-k" in argv:
        name_filter = argv[argv.index("-k") + 1]
    tests = [(n, f) for n, f in sorted(globals().items()) if _is_test(n, f)]
    if name_filter:
        tests = [(n, f) for n, f in tests if name_filter in n]
    passed = failed = skipped = 0
    failures = []
    t0 = time.time()
    for name, fn in tests:
        if skip_slow and getattr(fn, "_slow", False):
            print(f"SKIP  {name}")
            skipped += 1
            continue
        t1 = time.time()
        try:
            fn()
            print(f"PASS  {name}  ({time.time() - t1:.2f}s)")
            passed += 1
        except Exception as e:
            print(f"FAIL  {name}  ({time.time() - t1:.2f}s): "
                  f"{type(e).__name__}: {e}")
            failures.append((name, traceback.format_exc()))
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped "
          f"({time.time() - t0:.1f}s)")
    if failures:
        print("\n--- failure tracebacks ---")
        for name, tb in failures:
            print(f"\n[{name}]")
            print(tb)
    return failed


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1:]) == 0 else 1)
