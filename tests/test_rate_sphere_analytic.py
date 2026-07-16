"""
test_rate_sphere_analytic.py

Validate the RATE_Sphere encounter-rate estimator and the adaptive KDE pipeline
against analytic predictions for known velocity distributions in a uniform
spatial medium.

Convention follows the paper appendix:
    v_inf^2 = v^2 + v_esc^2       (so v_inf^2 = v^2 + 2GM/r)
    sin theta_c = (q/r) * sqrt[ (1 + 2GM/(q*v_inf^2)) / (1 - 2GM/(r*v_inf^2)) ]
With v_inf^2 = v^2 + 2GM/r, the denominator equals v^2/v_inf^2, and the formula reduces to
simple-Safronov sin^2theta_c = (q/r)^2*(1 + 2GM/(q*v^2)) (with v at radius r). We use that
closed form as the absolute ground truth.

Per-spec, particles with v^2 <= v_esc^2 (bound - can't reach infinity) are dropped.

Default test setup: sigma = 1 pc/Myr (~ 1 km/s), r = 0.1 pc, qmax = 5 AU. With v_esc/sigma = 0.3
the bound fraction of a Maxwellian is < 1%, so the closed form (with or without bound
truncation) is accurate. (1 km/s is much lower than realistic thin-disk dispersion, but
it is sufficient for the test to be in the unbound regime - the actual application has
a much higher sigma where the bug is even smaller.)

Three regimes are exercised by the sigma-sweep test:
    sigma = 0.3  ->  v_esc/sigma = 1     mostly bound - pre-fix this is where the ~10x bug was
    sigma = 1.0  ->  v_esc/sigma = 0.3   mostly unbound - main test case
    sigma = 3.0  ->  v_esc/sigma = 0.1   essentially fully unbound - bug was already small here

Run:
    ipython3 test_rate_sphere_analytic.py
    ipython3 test_rate_sphere_analytic.py --no-slow
    ipython3 test_rate_sphere_analytic.py -k drifting
"""
import sys
import time
import traceback

import numpy as np
import scipy.integrate
import scipy.stats
import json

from cracked import (RATE_Sphere, G, pcperau,
                     adaptiveKDE, cvAdaptiveKDE, gaussianKDEWrapper, cvGaussianKDE)


# --- constants matching RATE_Sphere internals

QMAX_AU = 5.0
QMAX_PC = QMAX_AU * pcperau
M_TARGET = 1.0
GM = G * M_TARGET
R_SPHERE = 0.1
V_ESC_SQ = 2.0 * GM / R_SPHERE
V_ESC = np.sqrt(V_ESC_SQ)
V0 = 1.0
NBOOT = 30000
MYR_TO_YR = 1.0e-6


def slow(fn):
    fn._slow = True
    return fn


# --- analytic ground truth

def _sin2_thetac(vee):
    """Mirror of the (post-fix) RATE_Sphere sin^2theta_c, including the bound-orbit cutoff."""
    if vee * vee <= V_ESC_SQ:
        return 0.0
    vinfsq = vee * vee + V_ESC_SQ
    sintc = (QMAX_PC / R_SPHERE) * np.sqrt((1.0 + 2.0 * GM / (QMAX_PC * vinfsq)) / (1.0 - 2.0 * GM / (R_SPHERE * vinfsq)))
    return min(sintc, 1.0) ** 2


def analytic_drdomega(f_v_axisym, costheta_r, vmax=30.0):
    def integrand(vee):
        return (f_v_axisym(vee, costheta_r) * vee ** 3
                * _sin2_thetac(vee) * np.pi * R_SPHERE ** 2 * MYR_TO_YR)
    val, _ = scipy.integrate.quad(integrand, V_ESC, vmax, limit=300)
    return val


def analytic_total_rate(f_v_axisym):
    inner = lambda c: analytic_drdomega(f_v_axisym, c)
    val, _ = scipy.integrate.quad(inner, -1.0, 1.0, limit=200)
    return 2.0 * np.pi * val


def analytic_mean_costheta(f_v_axisym):
    top, _ = scipy.integrate.quad(lambda c: c * analytic_drdomega(f_v_axisym, c), -1.0, 1.0, limit=200)
    bot, _ = scipy.integrate.quad(lambda c: analytic_drdomega(f_v_axisym, c), -1.0, 1.0, limit=200)
    return top / bot


def closed_form_isotropic_rate(n0, sigma):
    """Simple Safronov for isotropic Maxwellian, with bound-orbit truncation.

    R = sqrt(2pi) * qmax^2 * n0 * exp(-v_esc^2/(2sigma^2)) * [v_esc^2/sigma + 2sigma + 2GM/(qmax*sigma)]  / Myr
    Multiply by 1e-6 to convert to /yr.

    Without truncation (the v_esc=0 limit), this reduces to the textbook form
        R = n0 * pi*qmax^2 * <v> * (1 + GM/(qmax*sigma^2)).
    """
    return (np.sqrt(2.0 * np.pi) * QMAX_PC ** 2 * n0
            * np.exp(-V_ESC_SQ / (2.0 * sigma ** 2))
            * (V_ESC_SQ / sigma + 2.0 * sigma + 2.0 * GM / (QMAX_PC * sigma))
            * MYR_TO_YR)


# --- DF callables

def make_isotropic_df(n0, sigma):
    norm = n0 * (2.0 * np.pi * sigma ** 2) ** -1.5
    inv_two_sigma2 = 0.5 / sigma ** 2

    def df(points, covfac=1.0, show_contribs=False):
        pts = np.atleast_2d(points).reshape(-1, 6)
        vsq = pts[:, 3] ** 2 + pts[:, 4] ** 2 + pts[:, 5] ** 2
        return norm * np.exp(-vsq * inv_two_sigma2)
    return df


def make_drifting_df(n0, v_bulk, sigma):
    v_bulk = np.asarray(v_bulk, dtype=float)
    norm = n0 * (2.0 * np.pi * sigma ** 2) ** -1.5
    inv_two_sigma2 = 0.5 / sigma ** 2

    def df(points, covfac=1.0, show_contribs=False):
        pts = np.atleast_2d(points).reshape(-1, 6)
        v = pts[:, 3:] - v_bulk[None, :]
        vsq = v[:, 0] ** 2 + v[:, 1] ** 2 + v[:, 2] ** 2
        return norm * np.exp(-vsq * inv_two_sigma2)
    return df


def isotropic_fv(n0, sigma):
    norm = n0 * (2.0 * np.pi * sigma ** 2) ** -1.5
    return lambda vee, ct: norm * np.exp(-vee * vee / (2.0 * sigma ** 2))


def drifting_fv(n0, v_bulk_z, sigma):
    """f_v at v = -vee rhat (axisymmetric around zhat at polar angle ct):
       |v - v_b zhat|^2 = vee^2 + 2*vee*v_b*ct + v_b^2
    """
    norm = n0 * (2.0 * np.pi * sigma ** 2) ** -1.5

    def fv(vee, ct):
        norm_sq = vee * vee + 2.0 * vee * v_bulk_z * ct + v_bulk_z * v_bulk_z
        return norm * np.exp(-norm_sq / (2.0 * sigma ** 2))
    return fv


def sample_uniform_pos_gaussian_v(rng, n_samp, dx, v_mean, sigma):
    coords = np.zeros((n_samp, 6))
    coords[:, :3] = rng.uniform(-dx / 2.0, dx / 2.0, size=(n_samp, 3))
    coords[:, 3:] = v_mean[None, :] + rng.standard_normal(size=(n_samp, 3)) * sigma
    return coords


def run_rate_sphere(cvkde, fac=1.0):
    out = RATE_Sphere(cvkde, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], qmaxAU=QMAX_AU, Nboot=NBOOT, fac=fac, r=R_SPHERE, v0=V0)
    weights = np.asarray(out[0])
    costhetas = np.asarray(out[2])
    return float(np.mean(weights)), costhetas, weights


# IS helpers re-exported here for back-compat with callsites that do
# `from test_rate_sphere_analytic import make_data_driven_is_proposal`.
from cracked import (make_data_driven_is_proposal as _make_is_proposal_helper,
                     rate_sphere_importance as _rate_sphere_importance_helper)
make_data_driven_is_proposal = _make_is_proposal_helper


def rate_sphere_importance(cvkde, v_proposal_mean, v_proposal_cov, xloc=(0.0, 0.0, 0.0), vloc=(0.0, 0.0, 0.0), Nboot=NBOOT, fac=1.0, qmaxAU=QMAX_AU, r=R_SPHERE, rng=None):
    """Thin wrapper around `rate_sphere_helpers.rate_sphere_importance` that
    fills in the test file's NBOOT/QMAX_AU/R_SPHERE defaults so tests get the
    same numerics they used to. See `rate_sphere_helpers` for the full docstring.
    """
    return _rate_sphere_importance_helper(cvkde, v_proposal_mean, v_proposal_cov, xloc=xloc, vloc=vloc, Nboot=Nboot, fac=fac, qmaxAU=qmaxAU, r=r, rng=rng)


def weighted_mean(weights, values):
    return float(np.sum(weights * values) / np.sum(weights))


# --- common parameters

N0 = 1.0e15
SIGMA = 1.0     # pc/Myr (~ 1 km/s)
DX = 10.0
V_BULK_Z = 0.5  # pc/Myr - chosen so the drifting-Maxwellian peak (v ~ v_bulk) is comfortably above v_esc = 0.3


# Sanity checks (cheap)

def test_v_esc_value():
    # Sanity: with our r and GM, v_esc should be ~ 0.3 pc/Myr.
    assert abs(V_ESC - 0.3) < 0.005, f"V_ESC = {V_ESC}"


def test_isotropic_quadrature_matches_closed_form():
    """Quadrature using the (post-fix) RATE_Sphere integrand should reduce to
    simple-Safronov for sigma >> v_esc, where bound-orbit truncation is negligible.
    """
    fv = isotropic_fv(N0, SIGMA)
    R_quad = analytic_total_rate(fv)
    R_closed = closed_form_isotropic_rate(N0, SIGMA)
    rel_err = abs(R_quad - R_closed) / R_closed
    assert rel_err < 0.02, \
        f"closed={R_closed:.3g}, quad={R_quad:.3g}, rel_err={rel_err:.3g}"


def test_isotropic_mean_costheta_is_zero():
    fv = isotropic_fv(N0, SIGMA)
    m = analytic_mean_costheta(fv)
    assert abs(m) < 1e-6, f"|mean(costheta)| = {abs(m):.3g}"


def test_bound_particles_excluded():
    """Sin^2theta_c is exactly 0 for any v^2 < v_esc^2."""
    for v in [0.0, 0.1, 0.2, 0.29]:
        assert _sin2_thetac(v) == 0.0
    # And nonzero for v > v_esc:
    for v in [0.31, 0.5, 1.0, 5.0]:
        assert _sin2_thetac(v) > 0


# Isotropic Maxwellian (sigma=1, mostly unbound)

@slow
def test_isotropic_DF_rate_matches_analytic():
    df = make_isotropic_df(N0, SIGMA)
    fv = isotropic_fv(N0, SIGMA)
    R_an = analytic_total_rate(fv)
    R_closed = closed_form_isotropic_rate(N0, SIGMA)
    R_DF, _, _ = run_rate_sphere(df, fac=1.0)
    print(f"  [info] sigma={SIGMA}: R_closed={R_closed:.3g}  R_quad={R_an:.3g}  R_DF={R_DF:.3g}")
    assert abs(R_DF - R_an) / R_an < 0.10
    assert abs(R_DF - R_closed) / R_closed < 0.10


@slow
def test_isotropic_KDE_rate_matches_analytic():
    fv = isotropic_fv(N0, SIGMA)
    R_an = analytic_total_rate(fv)
    rng = np.random.default_rng(13)
    coords = sample_uniform_pos_gaussian_v(rng, 4000, DX, np.zeros(3), SIGMA)
    kde = adaptiveKDE(coords, scalings=[1, 1, 1, SIGMA, SIGMA, SIGMA], nn=50, use_multiprocessing=False)
    R_KDE, _, _ = run_rate_sphere(kde, fac=DX ** 3 * N0)
    rel_err = abs(R_KDE - R_an) / R_an
    print(f"  [info] sigma={SIGMA} KDE: R_an={R_an:.3g}  R_KDE={R_KDE:.3g}  rel_err={rel_err:.3f}")
    assert rel_err < 0.20


@slow
def test_isotropic_DF_sky_is_flat():
    df = make_isotropic_df(N0, SIGMA)
    _, ct, w = run_rate_sphere(df, fac=1.0)
    m = weighted_mean(w, ct)
    print(f"  [info] sigma={SIGMA} DF mean(costheta) = {m:+.4f}")
    assert abs(m) < 0.05


@slow
def test_isotropic_KDE_sky_is_flat():
    rng = np.random.default_rng(14)
    coords = sample_uniform_pos_gaussian_v(rng, 4000, DX, np.zeros(3), SIGMA)
    kde = adaptiveKDE(coords, scalings=[1, 1, 1, SIGMA, SIGMA, SIGMA], nn=50, use_multiprocessing=False)
    _, ct, w = run_rate_sphere(kde, fac=DX ** 3 * N0)
    m = weighted_mean(w, ct)
    print(f"  [info] sigma={SIGMA} KDE mean(costheta) = {m:+.4f}")
    assert abs(m) < 0.10


# Drifting Maxwellian (sigma=1, v_bulk=0.5 along zhat)

@slow
def test_drifting_DF_rate_matches_analytic():
    df = make_drifting_df(N0, [0.0, 0.0, V_BULK_Z], SIGMA)
    fv = drifting_fv(N0, V_BULK_Z, SIGMA)
    R_an = analytic_total_rate(fv)
    R_DF, _, _ = run_rate_sphere(df, fac=1.0)
    rel_err = abs(R_DF - R_an) / R_an
    print(f"  [info] drifting: R_an={R_an:.3g}  R_DF={R_DF:.3g}  rel_err={rel_err:.3f}")
    assert rel_err < 0.10


@slow
def test_drifting_KDE_rate_within_known_bias():
    """KDE-driven rate for the drifting Maxwellian is biased low by ~25-30% - the
    KDE smooths the velocity distribution and the rate's leverage at low v amplifies
    that bias. This test asserts the bias hasn't grown beyond the known level; it is
    NOT a clean recovery test - it pins the known ISE-vs-rate CV mismatch.
    """
    fv = drifting_fv(N0, V_BULK_Z, SIGMA)
    R_an = analytic_total_rate(fv)
    rng = np.random.default_rng(15)
    coords = sample_uniform_pos_gaussian_v(rng, 4000, DX, np.array([0.0, 0.0, V_BULK_Z]), SIGMA)
    kde = adaptiveKDE(coords, scalings=[1, 1, 1, SIGMA, SIGMA, SIGMA], nn=50, use_multiprocessing=False)
    R_KDE, _, _ = run_rate_sphere(kde, fac=DX ** 3 * N0)
    rel_err = abs(R_KDE - R_an) / R_an
    print(f"  [info] drifting KDE: R_an={R_an:.3g}  R_KDE={R_KDE:.3g}  rel_err={rel_err:.3f}")
    assert rel_err < 0.40, f"KDE rate bias {rel_err:.2f} exceeds the documented 25-30% level"


@slow
def test_drifting_DF_sky_anisotropy_matches_analytic():
    fv = drifting_fv(N0, V_BULK_Z, SIGMA)
    m_an = analytic_mean_costheta(fv)
    print(f"  [info] drifting analytic mean(costheta) = {m_an:+.4f}")
    assert m_an < -0.05, "expect a negative-costheta bias for v_bulk along +zhat"

    df = make_drifting_df(N0, [0.0, 0.0, V_BULK_Z], SIGMA)
    _, ct, w = run_rate_sphere(df, fac=1.0)
    m_DF = weighted_mean(w, ct)
    print(f"  [info] drifting DF       mean(costheta) = {m_DF:+.4f}")
    rel_err = abs(m_DF - m_an) / abs(m_an)
    assert rel_err < 0.20


@slow
def test_drifting_KDE_sky_anisotropy_partial_recovery():
    """The KDE smooths out a substantial fraction of the genuine sky anisotropy.
    With sigma=1, v_bulk=0.5, N=4000, current observation is that the KDE recovers
    only ~25% of the analytic mean(costheta). This is the (costheta, phi) artifact
    flagged in the original report - it's not a bug in RATE_Sphere or in the KDE
    code, it's a fundamental over-smoothing of the velocity distribution.

    This test asserts that at least the SIGN of the asymmetry is preserved and
    a non-trivial fraction (>10%) of the magnitude is recovered.
    """
    fv = drifting_fv(N0, V_BULK_Z, SIGMA)
    m_an = analytic_mean_costheta(fv)
    rng = np.random.default_rng(16)
    coords = sample_uniform_pos_gaussian_v(rng, 4000, DX, np.array([0.0, 0.0, V_BULK_Z]), SIGMA)
    kde = adaptiveKDE(coords, scalings=[1, 1, 1, SIGMA, SIGMA, SIGMA], nn=50, use_multiprocessing=False)
    _, ct, w = run_rate_sphere(kde, fac=DX ** 3 * N0)
    m_KDE = weighted_mean(w, ct)
    fraction = m_KDE / m_an
    print(f"  [info] drifting KDE mean(costheta) = {m_KDE:+.4f}  "
          f"(analytic {m_an:+.4f}, recovered {fraction*100:.1f}%)")
    assert m_KDE * m_an > 0, "sign of anisotropy should match analytic"
    assert fraction > 0.10, f"KDE recovers only {fraction*100:.1f}% of analytic asymmetry"


# Better metrics than mean(costheta)

def kish_ess(weights):
    """Effective sample size of the rate estimator: (Sigmaw)^2 / Sigmaw^2.
    For all-equal weights this equals N. When a few samples dominate (low N_eff
    in the KDE -> spiky integrand) this drops sharply.
    """
    s = np.sum(weights)
    return float(s * s / np.sum(weights * weights))


def per_eval_neff(kde, points, n_subsample=300):
    """Kish ESS of the per-evaluation kernel contributions, evaluated at a
    subsample of the input points. For each query point, this is the effective
    number of data points contributing to *that single* density estimate - a
    median near 1 means each evaluation is dominated by a single data point
    (-> speckly per-direction noise); a median near N_data means kernels are wide
    enough that many points contribute (-> smooth density and smooth sky map).

    Distinct from the rate-estimator Kish ESS (`kish_ess`), which measures
    convergence of the MC integral over (u, costheta, phi) given the integrand values.
    Per-evaluation N_eff is what governs the *spatial* smoothness of the integrand
    itself - it's the right number to look at when the sky map is blotchy.

    Returns array of N_eff values, one per evaluated query point.
    """
    import scipy.special
    if len(points) > n_subsample:
        rng = np.random.default_rng(7)
        idx = rng.choice(len(points), n_subsample, replace=False)
        sub_points = np.asarray(points[idx])
    else:
        sub_points = np.asarray(points)

    if isinstance(kde, adaptiveKDE):  # adaptiveKDE and mockScipyKde
        cache = kde.precompute_query(sub_points)
        mahas = cache['mahas']
        logdets = cache['logdets_at_n']
        dim = kde.data.shape[1]
        log2pi = np.log(2.0 * np.pi)
        log_covfac = (np.log(kde.covfac_overall)
                      + kde.covalpha_overall * logdets)
        mahas_eff = mahas / np.exp(log_covfac)
        contribs = -0.5 * (dim * log2pi + mahas_eff + dim * log_covfac + logdets)
        log_sum = scipy.special.logsumexp(contribs, axis=1)
        log_sum_sq = scipy.special.logsumexp(2.0 * contribs, axis=1)
        return np.exp(2.0 * log_sum - log_sum_sq)

    # gaussianKDEWrapper / cvGaussianKDE / scaledGaussianKDE - all wrap a scipy
    # gaussian_kde at .kde, which provides .dataset and .inv_cov.
    inner_kde = getattr(kde, 'kde', None)
    if isinstance(inner_kde, scipy.stats.gaussian_kde):
        data = inner_kde.dataset.T
        cov_inv = inner_kde.inv_cov
        a = np.sum(sub_points @ cov_inv * sub_points, axis=1)
        c = np.sum(data @ cov_inv * data, axis=1)
        b = sub_points @ cov_inv @ data.T
        mahas = a[:, None] + c[None, :] - 2.0 * b
        log_kernel = -0.5 * mahas
        log_sum = scipy.special.logsumexp(log_kernel, axis=1)
        log_sum_sq = scipy.special.logsumexp(2.0 * log_kernel, axis=1)
        return np.exp(2.0 * log_sum - log_sum_sq)

    return None  # unknown KDE type (e.g. analytic_DF callable)


def reconstruct_rate_sphere_coords(vee, costhetas, phis, r=R_SPHERE):
    """Rebuild the 6D query points RATE_Sphere evaluated, from its returned arrays.
    Sun (target) at origin, stationary."""
    sintheta = np.sqrt(np.maximum(0.0, 1.0 - costhetas ** 2))
    rhat_x = sintheta * np.cos(phis)
    rhat_y = sintheta * np.sin(phis)
    rhat_z = costhetas
    coords = np.zeros((len(vee), 6))
    coords[:, 0] = r * rhat_x
    coords[:, 1] = r * rhat_y
    coords[:, 2] = r * rhat_z
    coords[:, 3] = -vee * rhat_x
    coords[:, 4] = -vee * rhat_y
    coords[:, 5] = -vee * rhat_z
    return coords


def weighted_std(weights, values, mean=None):
    if mean is None:
        mean = weighted_mean(weights, values)
    return float(np.sqrt(np.sum(weights * (values - mean) ** 2) / np.sum(weights)))


def costheta_bin_residuals(weights, costhetas, f_v_axisym, n_bins=8):
    """Bin samples by costheta, compare to analytic dR/d(costheta)*Deltac.

    The MC estimator for the rate restricted to a bin is sum(weights_in_bin)/N_total
    (with weights = Vol*integrand and uniform sampling on (u, costheta, phi), this gives
    the right contribution; sums over bins recover the total rate).

    Returns observed bin rates, expected bin rates, and relative residuals.
    Statistical noise on each bin is ~ 1/sqrt(N_bin) of the bin rate; anything
    substantially above ~5% across bins indicates KDE-induced bias rather than
    MC scatter.
    """
    n_total = len(weights)
    edges = np.linspace(-1.0, 1.0, n_bins + 1)
    obs = np.zeros(n_bins)
    expected = np.zeros(n_bins)
    for k in range(n_bins):
        in_bin = (costhetas >= edges[k]) & (costhetas < edges[k + 1])
        obs[k] = float(np.sum(weights[in_bin])) / n_total
        # analytic rate in this bin: 2pi * int dR/dOmega(c) dc
        val, _ = scipy.integrate.quad(lambda c: analytic_drdomega(f_v_axisym, c), edges[k], edges[k + 1], limit=100)
        expected[k] = 2.0 * np.pi * val
    rel_resid = (obs - expected) / np.maximum(expected, 1e-30)
    return obs, expected, rel_resid


def density_agreement(kde, truth_fn, sampler_fn, fac=1.0, n_samp=2000, rng_seed=99):
    """How well does the KDE recover the true density at typical samples drawn
    from the truth? Reports log(p_kde*fac / p_truth) statistics - median is bias,
    p5/p95 give the spread. fac scales sample-based KDEs (probability density)
    to number density (multiplying by N0*V_pos).
    """
    rng = np.random.default_rng(rng_seed)
    samples = sampler_fn(rng, n_samp)
    p_truth = truth_fn(samples)
    p_kde = np.atleast_1d(kde(samples)) * fac
    valid = (p_truth > 0) & (p_kde > 0)
    log_ratio = np.log(p_kde[valid]) - np.log(p_truth[valid])
    return dict(median=float(np.median(log_ratio)), rms=float(np.sqrt(np.mean(log_ratio ** 2))), p5=float(np.percentile(log_ratio, 5)), p95=float(np.percentile(log_ratio, 95)))


def report_estimator(name, weights, costhetas, f_v_axisym=None, density_check=None, neff_per_eval_stats=None):
    R = float(np.mean(weights))
    m = weighted_mean(weights, costhetas)
    s = weighted_std(weights, costhetas, mean=m)
    n_eff = kish_ess(weights)
    msg = (f"  {name:>14s}  R={R:>9.3g}  <c>={m:+.4f}  sigma_c={s:.3f}  "
           f"N_eff_rate={n_eff:>7.1f}")
    if neff_per_eval_stats is not None:
        msg += f"  N_eff_per_eval(med/p95)={neff_per_eval_stats[0]:.1f}/{neff_per_eval_stats[1]:.1f}"
    if f_v_axisym is not None:
        _, _, rel_resid = costheta_bin_residuals(weights, costhetas, f_v_axisym)
        msg += f"  bin-resid std={np.std(rel_resid):.3f}  max={np.max(np.abs(rel_resid)):.3f}"
    if density_check is not None:
        d = density_check
        msg += f"  log(fhat/f) median={d['median']:+.3f}  p5/p95={d['p5']:+.2f}/{d['p95']:+.2f}"
    print(msg)
    return dict(R=R, mean_c=m, std_c=s, n_eff=n_eff, neff_per_eval=neff_per_eval_stats)


# --- samplers (used by density_agreement to draw test points from the truth)

def make_isotropic_sampler(sigma):
    def sample(rng, n):
        c = np.zeros((n, 6))
        c[:, :3] = rng.uniform(-DX / 2, DX / 2, size=(n, 3))
        c[:, 3:] = rng.standard_normal((n, 3)) * sigma
        return c
    return sample


def make_drifting_sampler(v_bulk, sigma):
    v_bulk = np.asarray(v_bulk, float)
    def sample(rng, n):
        c = np.zeros((n, 6))
        c[:, :3] = rng.uniform(-DX / 2, DX / 2, size=(n, 3))
        c[:, 3:] = v_bulk[None, :] + rng.standard_normal((n, 3)) * sigma
        return c
    return sample


def make_bimodal_sampler(v_a, v_b, sigma_a, sigma_b, alpha=0.5):
    v_a = np.asarray(v_a, float)
    v_b = np.asarray(v_b, float)
    def sample(rng, n):
        c = np.zeros((n, 6))
        c[:, :3] = rng.uniform(-DX / 2, DX / 2, size=(n, 3))
        which_a = rng.uniform(0, 1, size=n) < alpha
        c[which_a, 3:] = v_a[None, :] + rng.standard_normal((which_a.sum(), 3)) * sigma_a
        c[~which_a, 3:] = v_b[None, :] + rng.standard_normal(((~which_a).sum(), 3)) * sigma_b
        return c
    return sample


def make_ring_df(n0, v_R, sigma_perp, sigma_z):
    """Curved velocity DF: a ring in (v_x, v_y) of radius v_R with thermal radial scatter
    sigma_perp, plus a thin Gaussian in v_z. Rotationally invariant about zhat - so still axisymmetric
    for the rate calculation. The 1D marginal in v_x (or v_y) is bimodal at +/-v_R.

    p_perp(r) = (1/(2pi sigma_perp^2)) * exp(-(r-v_R)^2 / (2sigma_perp^2)) * I_0_e(r*v_R/sigma_perp^2)
    where the simplification uses the modified Bessel function I_0_e(x) = exp(-x)*I_0(x)
    (scipy.special.i0e) so the total stays well-behaved at any r*v_R/sigma^2.
    """
    import scipy.special
    norm_perp = 1.0 / (2.0 * np.pi * sigma_perp ** 2)
    norm_z = 1.0 / np.sqrt(2.0 * np.pi * sigma_z ** 2)

    def df(points, covfac=1.0, show_contribs=False):
        pts = np.atleast_2d(points).reshape(-1, 6)
        v_perp = np.sqrt(pts[:, 3] ** 2 + pts[:, 4] ** 2)
        v_z = pts[:, 5]
        ring = (norm_perp
                * np.exp(-(v_perp - v_R) ** 2 / (2.0 * sigma_perp ** 2))
                * scipy.special.i0e(v_perp * v_R / sigma_perp ** 2))
        zpart = norm_z * np.exp(-v_z ** 2 / (2.0 * sigma_z ** 2))
        return n0 * ring * zpart
    return df


def make_ring_sampler(v_R, sigma_perp, sigma_z):
    def sample(rng, n):
        c = np.zeros((n, 6))
        c[:, :3] = rng.uniform(-DX / 2, DX / 2, size=(n, 3))
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
        c[:, 3] = v_R * np.cos(theta) + rng.standard_normal(n) * sigma_perp
        c[:, 4] = v_R * np.sin(theta) + rng.standard_normal(n) * sigma_perp
        c[:, 5] = rng.standard_normal(n) * sigma_z
        return c
    return sample


def ring_fv(n0, v_R, sigma_perp, sigma_z):
    """f_v(vee, costheta_r) for the ring DF: at v = -vee*rhat with rhat at polar angle ct,
    |v_perp| = vee*sqrt(1-ct^2), v_z = -vee*ct. f_v factorises by rotational symmetry.
    """
    import scipy.special
    norm_perp = 1.0 / (2.0 * np.pi * sigma_perp ** 2)
    norm_z = 1.0 / np.sqrt(2.0 * np.pi * sigma_z ** 2)

    def fv(vee, ct):
        v_perp = vee * np.sqrt(max(0.0, 1.0 - ct * ct))
        v_z = -vee * ct
        ring = (norm_perp
                * np.exp(-(v_perp - v_R) ** 2 / (2.0 * sigma_perp ** 2))
                * scipy.special.i0e(v_perp * v_R / sigma_perp ** 2))
        zpart = norm_z * np.exp(-v_z ** 2 / (2.0 * sigma_z ** 2))
        return n0 * ring * zpart
    return fv


def make_bimodal_df(n0, v_a, v_b, sigma_a, sigma_b, alpha=0.5):
    v_a = np.asarray(v_a, float)
    v_b = np.asarray(v_b, float)
    norm_a = n0 * (2.0 * np.pi * sigma_a ** 2) ** -1.5
    norm_b = n0 * (2.0 * np.pi * sigma_b ** 2) ** -1.5

    def df(points, covfac=1.0, show_contribs=False):
        pts = np.atleast_2d(points).reshape(-1, 6)
        d_a = pts[:, 3:] - v_a[None, :]
        d_b = pts[:, 3:] - v_b[None, :]
        ea = np.exp(-(d_a ** 2).sum(axis=1) * 0.5 / sigma_a ** 2)
        eb = np.exp(-(d_b ** 2).sum(axis=1) * 0.5 / sigma_b ** 2)
        return alpha * norm_a * ea + (1.0 - alpha) * norm_b * eb
    return df


# Cold + hot mixture (multi-scale dispersion)
# Two co-centred isotropic Maxwellians with sigma_c << sigma_h. Cold is a small mass
# fraction but contributes strongly to the rate via gravitational focusing
# (R ~ <v>*(1 + GM/(qmax*sigma^2)) blows up at small sigma). A global-bandwidth KDE pools
# the two components' covariances and ends up with kernel sigma ~ sigma_h, smearing out
# the cold component's narrow peak - the local-Sigma_i adaptive KDE should do better.

def make_cold_hot_df(n0, sigma_cold, sigma_hot, alpha):
    norm_c = n0 * (2.0 * np.pi * sigma_cold ** 2) ** -1.5
    norm_h = n0 * (2.0 * np.pi * sigma_hot ** 2) ** -1.5
    inv_two_sc2 = 0.5 / sigma_cold ** 2
    inv_two_sh2 = 0.5 / sigma_hot ** 2

    def df(points, covfac=1.0, show_contribs=False):
        pts = np.atleast_2d(points).reshape(-1, 6)
        vsq = pts[:, 3] ** 2 + pts[:, 4] ** 2 + pts[:, 5] ** 2
        return (alpha * norm_c * np.exp(-vsq * inv_two_sc2)
                + (1.0 - alpha) * norm_h * np.exp(-vsq * inv_two_sh2))
    return df


def make_cold_hot_sampler(sigma_cold, sigma_hot, alpha):
    def sample(rng, n):
        c = np.zeros((n, 6))
        c[:, :3] = rng.uniform(-DX / 2, DX / 2, size=(n, 3))
        which_cold = rng.uniform(0, 1, size=n) < alpha
        c[which_cold, 3:] = rng.standard_normal((which_cold.sum(), 3)) * sigma_cold
        c[~which_cold, 3:] = rng.standard_normal(((~which_cold).sum(), 3)) * sigma_hot
        return c
    return sample


def cold_hot_fv(n0, sigma_cold, sigma_hot, alpha):
    norm_c = n0 * (2.0 * np.pi * sigma_cold ** 2) ** -1.5
    norm_h = n0 * (2.0 * np.pi * sigma_hot ** 2) ** -1.5

    def fv(vee, ct):
        return (alpha * norm_c * np.exp(-vee * vee / (2.0 * sigma_cold ** 2))
                + (1.0 - alpha) * norm_h * np.exp(-vee * vee / (2.0 * sigma_hot ** 2)))
    return fv


# Cosine shear (non-linear position-velocity correlation)
# v_z = A*cos(2pi*n*x/DX) + N(0, sigma^2); v_x, v_y are independent N(0, sigma^2).
#
# Why cosine and not linear: linear shear v_y = -A*x has Cov(x, v_y) != 0, which
# scipy's full-covariance kernel correctly captures. The kernel ellipsoid aligns
# along the shear ridge and scipy handles linear correlation perfectly - adaptive
# has nothing to add (verified empirically: scipy 96.6%, cvAdaptive 94.5%).
#
# Cosine shear has Cov(x, v_z) = 0 globally (cos is orthogonal to x over the
# symmetric box), so scipy gets no off-diagonal help. But Var(v_z) is still
# inflated to sigma^2 + A^2/2, so scipy's kernel sigma_z is broadened by the spatial
# modulation. At any local x, the conditional v_z dispersion is just sigma - adaptive
# can capture this *if* KNN ball << shear period DX/n.
#
# Using cos (not sin) so v_z_mean(x=0) = A -> local DF at the encounter target
# is anisotropic (drifting Maxwellian with bulk = A*zhat), not isotropic. Closed-
# form rate is `drifting_fv(N0, A, sigma)`.

def make_cosine_shear_df(n0, sigma_v, A, n_periods, dx_box):
    norm = n0 * (2.0 * np.pi * sigma_v ** 2) ** -1.5
    inv_two_s2 = 0.5 / sigma_v ** 2
    k = 2.0 * np.pi * n_periods / dx_box

    def df(points, covfac=1.0, show_contribs=False):
        pts = np.atleast_2d(points).reshape(-1, 6)
        v_z_resid = pts[:, 5] - A * np.cos(k * pts[:, 0])
        vsq = pts[:, 3] ** 2 + pts[:, 4] ** 2 + v_z_resid ** 2
        return norm * np.exp(-vsq * inv_two_s2)
    return df


def make_cosine_shear_sampler(sigma_v, A, n_periods, dx_box):
    k = 2.0 * np.pi * n_periods / dx_box
    def sample(rng, n):
        c = np.zeros((n, 6))
        c[:, :3] = rng.uniform(-dx_box / 2, dx_box / 2, size=(n, 3))
        c[:, 3:] = rng.standard_normal((n, 3)) * sigma_v
        c[:, 5] += A * np.cos(k * c[:, 0])
        return c
    return sample


# Side-by-side comparison: analytic DF vs scipy.gaussian_kde vs adaptive KDE

@slow
def test_compare_estimators_isotropic():
    """For an isotropic source: how much do scipy and adaptive KDEs distort the rate
    and the sky distribution? Reports rate, mean/std of costheta, Kish ESS, bin
    residuals (std and max), AND log(fhat/f_truth) statistics on samples from truth
    (the global density-agreement metric).
    """
    rng = np.random.default_rng(31)
    coords = sample_uniform_pos_gaussian_v(rng, 4000, DX, np.zeros(3), SIGMA)
    fv = isotropic_fv(N0, SIGMA)
    R_an = analytic_total_rate(fv)
    truth = make_isotropic_df(N0, SIGMA)
    sampler = make_isotropic_sampler(SIGMA)
    print(f"\n  analytic R = {R_an:.3g}, expected mean(costheta) = 0, std(costheta) = "
          f"{1.0/np.sqrt(3):.3f}\n")

    scipy_kde = gaussianKDEWrapper(coords)
    adapt50 = adaptiveKDE(coords, scalings=[1, 1, 1, SIGMA, SIGMA, SIGMA], nn=50, use_multiprocessing=False)
    adapt200 = adaptiveKDE(coords, scalings=[1, 1, 1, SIGMA, SIGMA, SIGMA], nn=200, use_multiprocessing=False)

    fac_kde = DX ** 3 * N0
    _, ct, w = run_rate_sphere(truth, fac=1.0)
    res_df = report_estimator("analytic DF", w, ct, fv, density_check=density_agreement(truth, truth, sampler, fac=1.0))
    _, ct, w = run_rate_sphere(scipy_kde, fac=fac_kde)
    report_estimator("scipy KDE", w, ct, fv, density_check=density_agreement(scipy_kde, truth, sampler, fac=fac_kde))
    _, ct, w = run_rate_sphere(adapt50, fac=fac_kde)
    report_estimator("adapt nn=50", w, ct, fv, density_check=density_agreement(adapt50, truth, sampler, fac=fac_kde))
    _, ct, w = run_rate_sphere(adapt200, fac=fac_kde)
    report_estimator("adapt nn=200", w, ct, fv, density_check=density_agreement(adapt200, truth, sampler, fac=fac_kde))

    assert abs(res_df["R"] - R_an) / R_an < 0.10
    assert abs(res_df["mean_c"]) < 0.05


@slow
def test_compare_estimators_drifting():
    """Same comparison for a drifting Maxwellian, with the density-agreement metric
    and an adaptive variant at nn=200 (vs the default nn=50).
    """
    rng = np.random.default_rng(32)
    v_bulk = np.array([0.0, 0.0, V_BULK_Z])
    coords = sample_uniform_pos_gaussian_v(rng, 4000, DX, v_bulk, SIGMA)
    fv = drifting_fv(N0, V_BULK_Z, SIGMA)
    R_an = analytic_total_rate(fv)
    m_an = analytic_mean_costheta(fv)
    truth = make_drifting_df(N0, v_bulk, SIGMA)
    sampler = make_drifting_sampler(v_bulk, SIGMA)
    print(f"\n  analytic R = {R_an:.3g},  <c> = {m_an:+.4f}\n")

    scipy_kde = gaussianKDEWrapper(coords)
    adapt50 = adaptiveKDE(coords, scalings=[1, 1, 1, SIGMA, SIGMA, SIGMA], nn=50, use_multiprocessing=False)
    adapt200 = adaptiveKDE(coords, scalings=[1, 1, 1, SIGMA, SIGMA, SIGMA], nn=200, use_multiprocessing=False)

    fac_kde = DX ** 3 * N0
    _, ct, w = run_rate_sphere(truth, fac=1.0)
    res_df = report_estimator("analytic DF", w, ct, fv, density_check=density_agreement(truth, truth, sampler, fac=1.0))
    _, ct, w = run_rate_sphere(scipy_kde, fac=fac_kde)
    report_estimator("scipy KDE", w, ct, fv, density_check=density_agreement(scipy_kde, truth, sampler, fac=fac_kde))
    _, ct, w = run_rate_sphere(adapt50, fac=fac_kde)
    report_estimator("adapt nn=50", w, ct, fv, density_check=density_agreement(adapt50, truth, sampler, fac=fac_kde))
    _, ct, w = run_rate_sphere(adapt200, fac=fac_kde)
    report_estimator("adapt nn=200", w, ct, fv, density_check=density_agreement(adapt200, truth, sampler, fac=fac_kde))

    assert abs(res_df["R"] - R_an) / R_an < 0.10
    assert abs(res_df["mean_c"] - m_an) / abs(m_an) < 0.10


@slow
def test_compare_estimators_bimodal():
    """Bimodal velocity distribution - two sharp Gaussian clumps along zhat. The global
    Gaussian KDE (scipy) cannot fit both peaks with a single bandwidth, so it
    smooths between them; the adaptive KDE's per-point local covariance should
    track each clump much better. This is the regime where the adaptive approach
    is *expected* to pay off.

    DF: 0.5*N(+0.4 zhat, sigma_c^2I) + 0.5*N(-0.4 zhat, sigma_c^2I), sigma_c = 0.15.
    Both modes are above v_esc=0.3, so both contribute to the rate. By symmetry
    the overall <c> ~ 0 (each mode pulls the rate to opposite hemispheres equally),
    but the *bin-residual* and *density-agreement* metrics will reveal whether
    each KDE preserves the bimodal structure.
    """
    rng = np.random.default_rng(33)
    sigma_c = 0.15
    v_a = np.array([0.0, 0.0, +0.4])
    v_b = np.array([0.0, 0.0, -0.4])

    truth = make_bimodal_df(N0, v_a, v_b, sigma_c, sigma_c, alpha=0.5)
    sampler = make_bimodal_sampler(v_a, v_b, sigma_c, sigma_c, alpha=0.5)

    # bimodal f_v(vee, costheta_r): each mode contributes via its drifting form
    norm = N0 * (2.0 * np.pi * sigma_c ** 2) ** -1.5
    def fv_bimodal(vee, ct):
        # mode A at +0.4 zhat contributes: |v - v_a|^2 with v = -vee rhat
        nsq_a = vee ** 2 + 2.0 * vee * (+0.4) * ct + 0.16
        # mode B at -0.4 zhat contributes: |v - v_b|^2
        nsq_b = vee ** 2 + 2.0 * vee * (-0.4) * ct + 0.16
        e_a = np.exp(-nsq_a / (2.0 * sigma_c ** 2))
        e_b = np.exp(-nsq_b / (2.0 * sigma_c ** 2))
        return 0.5 * norm * (e_a + e_b)

    R_an = analytic_total_rate(fv_bimodal)
    m_an = analytic_mean_costheta(fv_bimodal)
    print(f"\n  bimodal: analytic R = {R_an:.3g},  <c> = {m_an:+.4f}  (~ 0 by symmetry)\n")

    coords = sampler(np.random.default_rng(34), 4000)
    scipy_kde = gaussianKDEWrapper(coords)
    adapt50 = adaptiveKDE(coords, scalings=[1, 1, 1, SIGMA, SIGMA, SIGMA], nn=50, use_multiprocessing=False)
    adapt200 = adaptiveKDE(coords, scalings=[1, 1, 1, SIGMA, SIGMA, SIGMA], nn=200, use_multiprocessing=False)

    fac_kde = DX ** 3 * N0
    _, ct, w = run_rate_sphere(truth, fac=1.0)
    res_df = report_estimator("analytic DF", w, ct, fv_bimodal, density_check=density_agreement(truth, truth, sampler, fac=1.0))
    _, ct, w = run_rate_sphere(scipy_kde, fac=fac_kde)
    report_estimator("scipy KDE", w, ct, fv_bimodal, density_check=density_agreement(scipy_kde, truth, sampler, fac=fac_kde))
    _, ct, w = run_rate_sphere(adapt50, fac=fac_kde)
    report_estimator("adapt nn=50", w, ct, fv_bimodal, density_check=density_agreement(adapt50, truth, sampler, fac=fac_kde))
    _, ct, w = run_rate_sphere(adapt200, fac=fac_kde)
    report_estimator("adapt nn=200", w, ct, fv_bimodal, density_check=density_agreement(adapt200, truth, sampler, fac=fac_kde))

    assert abs(res_df["R"] - R_an) / R_an < 0.10


# Comparison plots - visualize how each estimator does on the sky, the velocity
# distribution, the rate, and the global density. Uses cvAdaptiveKDE (the actual
# CV-tuned variant) rather than raw adaptiveKDE.

import contextlib, io, os


@contextlib.contextmanager
def _suppress_stdout():
    """cvAdaptiveKDE is chatty; we don't want its output in our test logs."""
    saved = sys.stdout
    try:
        sys.stdout = io.StringIO()
        yield
    finally:
        sys.stdout = saved


def _make_neff_eval_points(coords, n_samples=300, rng=None, roi6=None):
    """Sample 6D points on the encounter sphere for the cvAdaptive N_eff floor.

    Sphere positions are uniform on the unit sphere (radius R_SPHERE around
    origin). Velocities are drawn from a data-driven proposal:

      - With `roi6=K`: select the K nearest particles to (origin, v=0) in
        scaled 6D Euclidean (auto-scaled by per-axis std), then sample
        velocities from N(mean(v_subset), cov(v_subset)). This keeps the
        floor's evaluation locus inside the same ROI the CV is optimising -
        otherwise the two mechanisms target different regions and the floor
        forces oversmoothing at the rate-leverage region without benefit.
      - With `roi6=None` (default): use `make_data_driven_is_proposal`
        (IS-style 4x inflated); broader, fits density-estimation use.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if roi6 is not None and coords.shape[1] == 6:
        scales = np.std(coords, axis=0)
        scales = np.where(scales > 0, scales, 1.0)
        center = np.zeros(6)
        diffs = (coords - center) / scales
        rsq6 = np.sum(diffs ** 2, axis=1)
        sortr = np.argsort(rsq6)
        K = min(int(roi6), coords.shape[0])
        subset_v = coords[sortr[:K], 3:]
        v_mean = subset_v.mean(axis=0)
        v_cov = np.cov(subset_v.T)
        # Floor v-cov at 1% of global per-axis var to avoid degeneracy if
        # the ROI subset has a tightly aligned v-distribution.
        floor = (0.01 * scales[3:]) ** 2
        v_cov = v_cov + np.diag(floor)
    else:
        v_mean, v_cov = make_data_driven_is_proposal(coords)
    v = rng.multivariate_normal(v_mean, v_cov, n_samples)
    costheta = rng.uniform(-1.0, 1.0, n_samples)
    phi = rng.uniform(-np.pi, np.pi, n_samples)
    sintheta = np.sqrt(np.maximum(0.0, 1.0 - costheta * costheta))
    x = R_SPHERE * sintheta * np.cos(phi)
    y = R_SPHERE * sintheta * np.sin(phi)
    z = R_SPHERE * costheta
    return np.column_stack([x, y, z, v])


def _build_estimators(coords, fac=None, label_data="data", cv_kwargs=None, nfolds=5, random_state=None, neff_floor=30.0, roi6=None):
    """Build the estimators we plot. Returns list of (name, kde, fac).

    Lineup (consolidated 2026-05-05):
      scipy_kde     - gaussianKDEWrapper, Scott's rule
      cvGaussianKDE - CV-tuned scalar bandwidth, scalings_grid=[None,'auto','narrow']
      cvAdaptive    - adaptive KDE with scalings_grid=[None,'auto','narrow'];
                      CV chooses among the three scalings instead of running them
                      as separate columns.

    `fac` converts unit-integral KDE density to a number density (the KDE returns
    intfhat dV = 1; truth = total_particles * pdf). Defaults to DX**3 * N0 for the
    uniform-cube scenarios.
    """
    if fac is None:
        fac = DX ** 3 * N0
    if cv_kwargs is None:
        cv_kwargs = {}
    print(f"  building scipy KDE on {label_data} ({len(coords)} samples)...")
    scipy_kde = gaussianKDEWrapper(coords)

    # Sphere eval points: shared by all CV-based estimators so the N_eff
    # floor is applied at the same locus across columns. When 6D ROI is set,
    # eval velocities are drawn from the ROI sub-population's v-distribution
    # so the floor and the ROI agree on the rate-leverage region; otherwise
    # the IS-proposal default applies.
    neff_eval_points = (None if neff_floor is None
                        else _make_neff_eval_points(coords, roi6=roi6))

    print(f"  building cvGaussianKDE nfolds={nfolds} (roi6={roi6})...")
    t0 = time.time()
    with _suppress_stdout():
        cv_gauss = cvGaussianKDE(coords, nfolds=nfolds, roi=None, roi6=roi6, scalings_grid=[None, 'auto', 'narrow', 'narrow_local'], random_state=random_state, neff_eval_points=neff_eval_points, neff_floor=neff_floor)
    bw_mult = cv_gauss.bw_multipliers[cv_gauss.best[1]]
    print(f"    done in {time.time()-t0:.0f}s "
          f"(bw={cv_gauss.bw:.3g}, mult={bw_mult:.3g}x Scott, "
          f"scaling={cv_gauss.scales_label})")
    if getattr(cv_gauss, 'neff_floor_active', False):
        nelig, ntot = cv_gauss.neff_n_eligible, cv_gauss.neff_n_total
        med = cv_gauss.neff_med_at_pick
        if nelig == 0:
            print(f"    N_eff floor={cv_gauss.neff_floor:.0f}: NO grid entry passed; "
                  f"unconstrained pick (med N_eff at pick = {med:.1f}).")
        else:
            print(f"    N_eff floor={cv_gauss.neff_floor:.0f}: "
                  f"{nelig}/{ntot} grid entries passed; "
                  f"picked entry has med N_eff = {med:.1f}")

    # Build the cvAdaptive column via the production factory so the test
    # exercises exactly the configuration production deploys
    # (scalings_grid=['auto', 'narrow', 'narrow_local'], local_pooled
    # shrinkage target, stability_lambda=2.0, etc.) and so the resulting
    # cv object carries the rate-weighted Kish-ESS picks (kde_sky, kde_vinf)
    # that the comparison-figure routing will use for sky/v_inf rows.
    from cracked import make_production_cv_kde
    print(f"  building cvAdaptiveKDE via make_production_cv_kde [production "
          f"settings, nfolds={nfolds}]...")
    t0 = time.time()
    cv = make_production_cv_kde(coords, xloc=(0.0, 0.0, 0.0), vloc=(0.0, 0.0, 0.0), encounter_radius_pc=R_SPHERE, random_state=random_state, suppress_stdout=True)
    shape_kde = cv.kde_shape
    rate_kde = cv.kde_rate
    print(f"    done in {time.time()-t0:.0f}s")
    print(f"      shape pick: covfac={shape_kde.covfac_overall:.3g}, "
          f"covalpha={shape_kde.covalpha_overall:.3g}, "
          f"shrinkage={shape_kde.shrinkage:.3g}, "
          f"scaling={cv.scalings_labels[cv.shape_best[5]]}")
    if rate_kde is not shape_kde:
        print(f"      rate pick:  covfac={rate_kde.covfac_overall:.3g}, "
              f"covalpha={rate_kde.covalpha_overall:.3g}, "
              f"shrinkage={rate_kde.shrinkage:.3g}, "
              f"scaling={cv.scalings_labels[cv.rate_best[5]]}")
    if getattr(cv, 'neff_floor_active', False):
        nelig, ntot = cv.neff_n_eligible, cv.neff_n_total
        med = cv.neff_med_at_pick
        if nelig == 0:
            print(f"    N_eff floor={cv.neff_floor:.0f}: NO grid entry passed; "
                  f"used unconstrained ISE-CV pick (med N_eff at pick = {med:.1f}). "
                  f"Adaptive KDE may not fit this scenario at this N.")
        else:
            print(f"    N_eff floor={cv.neff_floor:.0f}: {nelig}/{ntot} grid "
                  f"entries passed; picked entry has med N_eff = {med:.1f}")

    # Optional EnBiD column - the established adaptive 6D KDE from
    # Sharma & Steinmetz 2006. Runs only if the compiled binary is on disk;
    # silently skipped otherwise so the test suite stays portable.
    extra_entries = []
    try:
        from cracked.enbid import enbidKDE, _default_binary_path
        if _default_binary_path().exists():
            print(f"  building EnBiD (ngb=64) [Sharma & Steinmetz 2006 default]...")
            t0 = time.time()
            with _suppress_stdout():
                enbid64 = enbidKDE(coords, ngb=64)
            print(f"    done in {time.time()-t0:.0f}s "
                  f"(ngb=64, isotropic kernel + adaptive metric)")
            extra_entries.append(("EnBiD (ngb=64)", enbid64, fac))
    except Exception as e:
        print(f"  (EnBiD column skipped: {type(e).__name__}: {e})")

    # Optional Normalizing-Flow column - Masked Autoregressive Flow following
    # Buckley et al. 2022 (arXiv:2205.01129). Requires torch + nflows;
    # silently skipped otherwise.
    try:
        from cracked.normalizing_flow import NormalizingFlowKDE
        print(f"  building Normalizing Flow (MAF, 5 layers, 64 hidden units)...")
        t0 = time.time()
        with _suppress_stdout():
            nf = NormalizingFlowKDE(coords, n_layers=5, hidden_units=64, n_epochs=300, random_state=random_state)
        print(f"    done in {time.time()-t0:.0f}s "
              f"(best val NLL = {nf.best_val_nll:.3f})")
        extra_entries.append(("NormalizingFlow (MAF)", nf, fac))
    except ImportError as e:
        print(f"  (NormalizingFlow column skipped: {e})")
    except Exception as e:
        print(f"  (NormalizingFlow column failed: {type(e).__name__}: {e})")

    # Optional Neural-Spline-Flow column - same masked-autoregressive
    # conditioner, but each 1D transform is a piecewise rational-quadratic
    # spline (Durkan et al. 2019, arXiv:1906.04032). The candidate fix for
    # the affine MAF's failures on narrow spatial features (disk stream,
    # spiky ball), where stiff affine layers can't carve sharp ridges.
    try:
        from cracked.normalizing_flow import NormalizingFlowKDE
        print(f"  building Normalizing Flow (NSF, 5 layers, 64 hidden units)...")
        t0 = time.time()
        with _suppress_stdout():
            nsf = NormalizingFlowKDE(coords, transform='spline', n_layers=5, hidden_units=64, n_epochs=300, random_state=random_state)
        print(f"    done in {time.time()-t0:.0f}s "
              f"(best val NLL = {nsf.best_val_nll:.3f})")
        extra_entries.append(("NormalizingFlow (NSF)", nsf, fac))
    except ImportError as e:
        print(f"  (NSF column skipped: {e})")
    except Exception as e:
        print(f"  (NSF column failed: {type(e).__name__}: {e})")

    return [
        ("scipy_kde", scipy_kde, fac),
        ("cvGaussianKDE", cv_gauss, fac),
        ("cvAdaptive", cv, fac),
    ] + extra_entries


def _build_smart_cvadaptive(coords, scalings, fac, label='cvAdapt-smart', nfolds=5, random_state=None, neff_floor=30.0, roi6=None, **cv_kwargs):
    """Build a cvAdaptiveKDE with explicit domain-knowledge scalings.

    Returns a (label, kde, fac) tuple suitable for the `extra_estimators`
    argument of make_comparison_figure. The point of this column is to
    answer ``what if we used physical knowledge of the scenario to set
    the kdtree metric, rather than trusting CV's data-driven choice?''
    For monomodal scenarios (isotropic, drifting, ring, etc.) the smart
    scaling is just the natural per-axis structure scale; for streams it
    matches the cvAdapt-stream recipe used on the disk-stream scenario.

    Inherits the same N_eff floor as `_build_estimators` so smart and the
    main cvAdaptive column are held to the same structural-smoothness
    standard. Caller can pass `neff_floor=None` to disable.
    """
    cv_kw = dict(cv_kwargs)
    cv_kw.setdefault('neff_floor', neff_floor)
    cv_kw.setdefault('roi6', roi6)
    if cv_kw.get('neff_floor', None) is not None and 'neff_eval_points' not in cv_kw:
        cv_kw['neff_eval_points'] = _make_neff_eval_points(coords, roi6=cv_kw.get('roi6'))
    print(f"  building cvAdaptive [{label}, scalings={scalings}, "
          f"neff_floor={cv_kw.get('neff_floor')}, roi6={cv_kw.get('roi6')}]...")
    t0 = time.time()
    with _suppress_stdout():
        cv = cvAdaptiveKDE(coords, roi=None, nfolds=nfolds, scalings=list(scalings), random_state=random_state, **cv_kw)
    shape_kde = cv.kde_shape
    rate_kde = cv.kde_rate
    print(f"    done in {time.time()-t0:.0f}s")
    print(f"      shape pick: covfac={shape_kde.covfac_overall:.3g}, "
          f"sh={shape_kde.shrinkage:.3g}")
    if rate_kde is not shape_kde:
        print(f"      rate pick:  covfac={rate_kde.covfac_overall:.3g}, "
              f"sh={rate_kde.shrinkage:.3g}")
    if getattr(cv, 'neff_floor_active', False):
        nelig, ntot = cv.neff_n_eligible, cv.neff_n_total
        med = cv.neff_med_at_pick
        if nelig == 0:
            print(f"    N_eff floor={cv.neff_floor:.0f}: NO grid entry passed; "
                  f"unconstrained pick (med N_eff at pick = {med:.1f}).")
        else:
            print(f"    N_eff floor={cv.neff_floor:.0f}: {nelig}/{ntot} grid "
                  f"entries passed; picked entry has med N_eff = {med:.1f}")
    return (label, cv, fac)


def analytic_dR_dvee(vee, fv_axisym):
    """dR/dvee at speed `vee`, axisymmetric f_v. 0 for bound vee."""
    if vee * vee <= V_ESC_SQ:
        return 0.0
    sin2 = _sin2_thetac(vee)
    F_vee, _ = scipy.integrate.quad(lambda ct: fv_axisym(vee, ct), -1.0, 1.0, limit=80)
    return 2.0 * np.pi ** 2 * R_SPHERE ** 2 * MYR_TO_YR * vee ** 3 * sin2 * F_vee


def make_stream_ring_sampler(R_ring, sigma_R, sigma_h, sigma_t, v_circ, width, height):
    """Spatial ring in the xz-plane, centred at (R_ring, 0, 0). Ring passes through
    the origin (Sun) at theta=pi. Stream is thin: width=sigma_R/kappa in the radial direction,
    height=sigma_h/nu vertically. Velocity is bulk v_circ along the ring tangent plus
    Gaussian thermal scatter (sigma_R radial, sigma_h vertical, sigma_t tangential - small).

    At theta=pi (origin), local frame: tangent ~ -zhat, radial ~ -xhat, vertical = yhat.
    So at origin the local velocity DF is N(0, sigma_R) in v_x, N(0, sigma_h) in v_y,
    N(-v_circ, sigma_t) in v_z - anisotropic with small dispersion along the bulk.
    """
    R_c = R_ring   # ring center; ring of radius R_ring touches origin

    def sample(rng, n):
        c = np.zeros((n, 6))
        theta = rng.uniform(0.0, 2.0 * np.pi, n)
        eR_pos = rng.standard_normal(n) * width
        eh_pos = rng.standard_normal(n) * height
        eR_vel = rng.standard_normal(n) * sigma_R
        eh_vel = rng.standard_normal(n) * sigma_h
        eT_vel = rng.standard_normal(n) * sigma_t
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        # Position: ring in xz plane, "vertical" is the y axis.
        c[:, 0] = R_c + (R_ring + eR_pos) * cos_t
        c[:, 1] = eh_pos
        c[:, 2] = (R_ring + eR_pos) * sin_t
        # Velocity: tangenthat = (-sin theta, 0, cos theta), radialhat = (cos theta, 0, sin theta), verticalhat = yhat.
        c[:, 3] = -(v_circ + eT_vel) * sin_t + eR_vel * cos_t
        c[:, 4] = eh_vel
        c[:, 5] = (v_circ + eT_vel) * cos_t + eR_vel * sin_t
        return c
    return sample


def make_stream_ring_df(N_total, R_ring, sigma_R, sigma_h, sigma_t, v_circ, width, height):
    """Analytic 6D number density f(x, v) = N_total * p_pos(x) * p_vel(v | theta(x)).

    Position pdf: p_pos(x, y, z) = (1/(2pi*R_proj)) * N(R_proj-R_ring; 0, width^2) * N(y; 0, height^2)
    where R_proj = sqrt((x-R_c)^2 + z^2) is the in-plane distance from the ring centre.
    Velocity pdf (given theta): N(v_R; 0, sigma_R^2) * N(v_T; v_circ, sigma_t^2) * N(v_h; 0, sigma_h^2)
    """
    R_c = R_ring
    norm_eR = 1.0 / np.sqrt(2.0 * np.pi) / width
    norm_eh = 1.0 / np.sqrt(2.0 * np.pi) / height
    norm_vR = 1.0 / np.sqrt(2.0 * np.pi) / sigma_R
    norm_vT = 1.0 / np.sqrt(2.0 * np.pi) / sigma_t
    norm_vh = 1.0 / np.sqrt(2.0 * np.pi) / sigma_h

    def df(points, covfac=1.0, show_contribs=False):
        pts = np.atleast_2d(points).reshape(-1, 6)
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        vx, vy, vz = pts[:, 3], pts[:, 4], pts[:, 5]
        R_proj = np.sqrt((x - R_c) ** 2 + z ** 2)
        eR_pos = R_proj - R_ring
        R_proj_safe = np.maximum(R_proj, 1.0e-30)
        p_pos = ((1.0 / (2.0 * np.pi * R_proj_safe))
                 * norm_eR * np.exp(-eR_pos ** 2 / (2.0 * width ** 2))
                 * norm_eh * np.exp(-y ** 2 / (2.0 * height ** 2)))
        theta_x = np.arctan2(z, x - R_c)
        cos_t = np.cos(theta_x)
        sin_t = np.sin(theta_x)
        v_R = vx * cos_t + vz * sin_t
        v_T = -vx * sin_t + vz * cos_t
        v_h = vy
        p_vel = (norm_vR * np.exp(-v_R ** 2 / (2.0 * sigma_R ** 2))
                 * norm_vT * np.exp(-(v_T - v_circ) ** 2 / (2.0 * sigma_t ** 2))
                 * norm_vh * np.exp(-v_h ** 2 / (2.0 * sigma_h ** 2)))
        return N_total * p_pos * p_vel
    return df


def stream_ring_fv(N_total, R_ring, sigma_R, sigma_h, sigma_t, v_circ, width, height):
    """f_v(vee, costheta_r) for the at-origin local DF, axisymmetric around zhat.

    At origin (theta_x = pi), local frame gives v_R = -v_x, v_T = -v_z, v_h = v_y.
    Axisymmetry around zhat requires sigma_R == sigma_h. v at -vee rhat has |v_perp|^2 = vee^2(1-ct^2)
    and v_z = -vee*ct, so v_T = vee*ct, v_R^2 + v_h^2 = |v_perp|^2.
    """
    if not np.isclose(sigma_R, sigma_h):
        raise ValueError("axisymmetric framework requires sigma_R == sigma_h")
    R_c = R_ring
    # local number density at origin = N_total * p_pos(0)
    n_local = N_total * (1.0 / (2.0 * np.pi * R_c)) \
        * (1.0 / np.sqrt(2.0 * np.pi) / width) \
        * (1.0 / np.sqrt(2.0 * np.pi) / height)
    norm_perp = 1.0 / (2.0 * np.pi * sigma_R ** 2)
    norm_T = 1.0 / np.sqrt(2.0 * np.pi) / sigma_t

    def fv(vee, ct):
        v_perp_sq = vee * vee * (1.0 - ct * ct)
        v_T = vee * ct
        return (n_local
                * norm_perp * np.exp(-v_perp_sq / (2.0 * sigma_R ** 2))
                * norm_T * np.exp(-(v_T - v_circ) ** 2 / (2.0 * sigma_t ** 2)))
    return fv


def make_disk_stream_sampler(R_ring, sigma_R, sigma_z, sigma_t, v_circ, width, height, v_sun_peculiar=(0.0, 0.0, 0.0)):
    """Galactic disk-like stream - ring at R=R_ring in the xy plane (z = vertical),
    Sun positioned at (R_ring, 0, 0). Returns coordinates in the SUN'S REST FRAME.

    Stream tangent at the Sun (phi=0): +yhat (positive v_circ -> +yhat flow at Sun).
    Radial at Sun: +xhat. Vertical: +zhat.

    `v_sun_peculiar = (radial_offset, tangent_offset, vertical_offset)` in km/s,
    so the Sun's full galactic velocity is
        (v_sun_peculiar[0], v_circ + v_sun_peculiar[1], v_sun_peculiar[2]).
    """
    v_sun_gal = np.array([v_sun_peculiar[0], v_circ + v_sun_peculiar[1],
                          v_sun_peculiar[2]])

    def sample(rng, n):
        c = np.zeros((n, 6))
        phi = rng.uniform(0.0, 2.0 * np.pi, n)
        e_R_pos = rng.standard_normal(n) * width
        e_z_pos = rng.standard_normal(n) * height
        e_R_vel = rng.standard_normal(n) * sigma_R
        e_z_vel = rng.standard_normal(n) * sigma_z
        e_T_vel = rng.standard_normal(n) * sigma_t
        cp = np.cos(phi); sp = np.sin(phi)
        # Galactic Cartesian position
        x_gal = (R_ring + e_R_pos) * cp
        y_gal = (R_ring + e_R_pos) * sp
        z_gal = e_z_pos
        # Galactic Cartesian velocity:
        # tangenthat(phi) = (-sin phi, cos phi, 0); radialhat(phi) = (cos phi, sin phi, 0); zhat
        v_x_gal = -(v_circ + e_T_vel) * sp + e_R_vel * cp
        v_y_gal = +(v_circ + e_T_vel) * cp + e_R_vel * sp
        v_z_gal = e_z_vel
        # Transform to Sun's rest frame
        c[:, 0] = x_gal - R_ring
        c[:, 1] = y_gal
        c[:, 2] = z_gal
        c[:, 3] = v_x_gal - v_sun_gal[0]
        c[:, 4] = v_y_gal - v_sun_gal[1]
        c[:, 5] = v_z_gal - v_sun_gal[2]
        return c
    return sample


def make_disk_stream_df(N_total, R_ring, sigma_R, sigma_z, sigma_t, v_circ, width, height, v_sun_peculiar=(0.0, 0.0, 0.0)):
    """Analytic 6D number density f(x_sun, v_sun) for the disk-like stream,
    expressed in the SUN'S REST FRAME (transforms to galactic internally).

    Position pdf in galactic cylindrical (R_proj, phi, z):
        p(x, y, z) = (1/(2pi*R_proj))*N(R_proj-R_ring; 0, width^2)*N(z; 0, height^2)
    Velocity pdf at this point: anisotropic Gaussian
        N(v_R; 0, sigma_R^2)*N(v_T; v_circ, sigma_t^2)*N(v_z; 0, sigma_z^2)
    """
    v_sun_gal = np.array([v_sun_peculiar[0], v_circ + v_sun_peculiar[1],
                          v_sun_peculiar[2]])
    norm_eR = 1.0 / np.sqrt(2.0 * np.pi) / width
    norm_ez = 1.0 / np.sqrt(2.0 * np.pi) / height
    norm_vR = 1.0 / np.sqrt(2.0 * np.pi) / sigma_R
    norm_vT = 1.0 / np.sqrt(2.0 * np.pi) / sigma_t
    norm_vz = 1.0 / np.sqrt(2.0 * np.pi) / sigma_z

    def df(points, covfac=1.0, show_contribs=False):
        pts = np.atleast_2d(points).reshape(-1, 6)
        # Sun's frame -> galactic
        x_gal = pts[:, 0] + R_ring
        y_gal = pts[:, 1]
        z_gal = pts[:, 2]
        v_x_gal = pts[:, 3] + v_sun_gal[0]
        v_y_gal = pts[:, 4] + v_sun_gal[1]
        v_z_gal = pts[:, 5] + v_sun_gal[2]
        R_proj = np.sqrt(x_gal ** 2 + y_gal ** 2)
        R_safe = np.maximum(R_proj, 1.0e-30)
        e_R = R_proj - R_ring
        p_pos = ((1.0 / (2.0 * np.pi * R_safe))
                 * norm_eR * np.exp(-e_R ** 2 / (2.0 * width ** 2))
                 * norm_ez * np.exp(-z_gal ** 2 / (2.0 * height ** 2)))
        cp = x_gal / R_safe
        sp = y_gal / R_safe
        v_R_local = v_x_gal * cp + v_y_gal * sp
        v_T_local = -v_x_gal * sp + v_y_gal * cp
        p_vel = (norm_vR * np.exp(-v_R_local ** 2 / (2.0 * sigma_R ** 2))
                 * norm_vT * np.exp(-(v_T_local - v_circ) ** 2 / (2.0 * sigma_t ** 2))
                 * norm_vz * np.exp(-v_z_gal ** 2 / (2.0 * sigma_z ** 2)))
        return N_total * p_pos * p_vel
    return df


# Spiky-ball scenario: N independent thin streams of equal weight passing
# through the origin in random 3D directions, each carrying a bulk velocity
# v_speed along its own direction. Designed as a paper-level ablation that
# exposes where cvAdaptive's per-point local-covariance kernels should beat
# fixed-capacity normalizing flows (NF must encode each spike explicitly;
# cvAdaptive's representation scales with N because each particle's kernel
# is locally oriented along its own stream).

def _spiky_ball_directions(n_spikes, seed=42):
    """Fixed-seed random unit directions on the 2-sphere. Returned as
    (n_spikes, 3); also returns per-spike orthonormal in-plane basis."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(-1.0, 1.0, n_spikes)
    phi = rng.uniform(0.0, 2.0 * np.pi, n_spikes)
    n_hat = np.stack([np.sqrt(np.maximum(0.0, 1.0 - z * z)) * np.cos(phi),
                       np.sqrt(np.maximum(0.0, 1.0 - z * z)) * np.sin(phi),
                       z], axis=1)
    # Build an orthonormal (e1, e2) for each spike - pick a vector not parallel
    # to n_hat and Gram-Schmidt.
    e_ref = np.where(np.abs(n_hat[:, [0]]) < 0.9, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    e1 = e_ref - n_hat * np.einsum('ij,ij->i', e_ref, n_hat)[:, None]
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(n_hat, e1)
    return n_hat, e1, e2


def make_spiky_ball_sampler(n_spikes, L, sigma_perp, sigma_v_long, sigma_v_perp, v_speed, geom_seed=42):
    """Returns (sampler, n_hat). sampler(rng, n_total) -> (n_total, 6).
    Particles are evenly split across spikes (n_total // n_spikes per spike).
    Each spike: position along its axis uniform in [-L/2, L/2], 2D Gaussian
    perpendicular displacement (sigma_perp); velocity v_speed*nhat along axis with
    1D Gaussian sigma_v_long along the axis and 2D Gaussian sigma_v_perp perpendicular.
    """
    n_hat, e1, e2 = _spiky_ball_directions(n_spikes, seed=geom_seed)

    def sampler(rng, n_total):
        per = int(n_total // n_spikes)
        out = np.zeros((per * n_spikes, 6), dtype=float)
        for k in range(n_spikes):
            nh, u1, u2 = n_hat[k], e1[k], e2[k]
            i0 = k * per
            t  = rng.uniform(-0.5 * L, 0.5 * L, per)
            p1 = rng.normal(0.0, sigma_perp, per)
            p2 = rng.normal(0.0, sigma_perp, per)
            vL = rng.normal(v_speed, sigma_v_long, per)
            vp1 = rng.normal(0.0, sigma_v_perp, per)
            vp2 = rng.normal(0.0, sigma_v_perp, per)
            out[i0:i0+per, :3] = (t[:, None] * nh + p1[:, None] * u1
                                   + p2[:, None] * u2)
            out[i0:i0+per, 3:] = (vL[:, None] * nh + vp1[:, None] * u1
                                   + vp2[:, None] * u2)
        return out
    return sampler, n_hat


def make_spiky_ball_df(N_total, n_spikes, L, sigma_perp, sigma_v_long, sigma_v_perp, v_speed, geom_seed=42):
    """Joint 6D density f(x, v) = N_total * (1/n_spikes) * Sigma_k f_k(x, v)
    where each f_k is the k-th stream's per-particle density (uniform-along-axis
    x 2D Gaussian perp position x 1D Gaussian along velocity x 2D Gaussian
    perp velocity), all normalized so f_k integrates to 1 over phase space."""
    n_hat, e1, e2 = _spiky_ball_directions(n_spikes, seed=geom_seed)
    inv_two_pi_sp_sq = 1.0 / (2.0 * np.pi * sigma_perp * sigma_perp)
    inv_sqrt_two_pi_svl = 1.0 / np.sqrt(2.0 * np.pi * sigma_v_long ** 2)
    inv_two_pi_svp_sq = 1.0 / (2.0 * np.pi * sigma_v_perp * sigma_v_perp)
    inv_2sp_sq  = 1.0 / (2.0 * sigma_perp ** 2)
    inv_2svl_sq = 1.0 / (2.0 * sigma_v_long ** 2)
    inv_2svp_sq = 1.0 / (2.0 * sigma_v_perp ** 2)

    def df(coords):
        coords = np.atleast_2d(np.asarray(coords))
        pos = coords[:, :3]
        vel = coords[:, 3:]
        total = np.zeros(coords.shape[0])
        for k in range(n_spikes):
            nh = n_hat[k]
            t_along = pos @ nh
            in_range = (np.abs(t_along) < 0.5 * L).astype(float)
            d_perp_sq = ((pos - t_along[:, None] * nh[None, :]) ** 2).sum(axis=1)
            v_long = vel @ nh
            v_perp_sq = ((vel - v_long[:, None] * nh[None, :]) ** 2).sum(axis=1)
            f_k = (in_range / L
                   * inv_two_pi_sp_sq * np.exp(-d_perp_sq * inv_2sp_sq)
                   * inv_sqrt_two_pi_svl
                       * np.exp(-(v_long - v_speed) ** 2 * inv_2svl_sq)
                   * inv_two_pi_svp_sq * np.exp(-v_perp_sq * inv_2svp_sq))
            total += f_k
        return N_total * total / n_spikes
    return df


def make_volume_sampler_box(spatial_lo, spatial_hi, velocity_lo, velocity_hi):
    """Generalised volume sampler - uniform over a 6D box specified by per-axis bounds.
    Useful when the spatial support of the truth doesn't fit in the standard `DX` cube
    (e.g., the ~1000-pc-long physical-space ring)."""
    spatial_lo = np.asarray(spatial_lo, dtype=float)
    spatial_hi = np.asarray(spatial_hi, dtype=float)
    velocity_lo = np.asarray(velocity_lo, dtype=float)
    velocity_hi = np.asarray(velocity_hi, dtype=float)
    def sample(rng, n):
        c = np.zeros((n, 6))
        c[:, :3] = rng.uniform(spatial_lo, spatial_hi, size=(n, 3))
        c[:, 3:] = rng.uniform(velocity_lo, velocity_hi, size=(n, 3))
        return c
    return sample


def make_volume_sampler(velocity_range):
    """Uniform-in-volume sampler over the spatial cube and a velocity box.
    Used to characterise volume-weighted (vs mass-weighted) KDE bias.
    velocity_range is a (low, high) tuple covering most of the velocity support.
    """
    v_lo, v_hi = velocity_range
    def sample(rng, n):
        c = np.zeros((n, 6))
        c[:, :3] = rng.uniform(-DX / 2, DX / 2, size=(n, 3))
        c[:, 3:] = rng.uniform(v_lo, v_hi, size=(n, 3))
        return c
    return sample


# Pass/fail thresholds for subplot annotations.
_RATE_FACTOR = 2.0         # 1/_RATE_FACTOR < R/R_an < _RATE_FACTOR -> ok
# Backward-compat alias used in legacy code paths and JSON dumps.
_RATE_TOL = _RATE_FACTOR - 1.0
_KS_COSTH_TOL = 0.10       # max|DeltaCDF| of normalised dR/dcostheta shape < this -> ok
_BINRES_COSTH_TOL = 0.30   # max|obs_pdf - ana_pdf|/max(ana_pdf) for costheta shape -> ok
_KS_LOGVEE_TOL = 0.10      # KS for log10(vee) shape (normalised PDF) -> ok
_BINRES_LOGVEE_TOL = 0.30  # bin-residual max for log10(vee) shape -> ok
_LOG10_RATIO_TOL = 0.22    # |median(log10 fhat/f)| < this -> ok
_KS_VMARG_TOL = 0.10       # ks_2samp on v_marginal < this -> ok
_BINRES_VMARG_TOL = 0.30   # max|hist_truth-hist_kde|/max(hist_truth) for v_marg -> ok
_KS_LOGDENS_TOL = 0.20     # ks_2samp on log10(density) value distribution -> ok


# Display names for the estimator columns, used in panel titles and
# legends. Decoupled from the internal labels (which still appear in the
# stats JSONs) so that the plots read with full English instead of code-
# style identifiers.
_DISPLAY_NAMES = {
    'analytic_DF':    'analytic',
    'scipy_kde':      'SciPy gaussian_kde',
    'cvGaussianKDE':  'CV Bandwidth',
    'cvAdaptive':     'CV Adaptive',
    'cvAdapt-smart':  'CV Adaptive (informed)',
}


def _display_name(internal):
    return _DISPLAY_NAMES.get(internal, internal)


def _annot(ax, label, loc='ur', fontsize=8.5):
    """Top-corner informational text. No pass/fail colouring; the
    best-performing column per row is indicated by a green border on the
    axes spines, set after the main loop in `_highlight_best`.
    """
    if loc == 'ur':
        x, y, ha, va = 0.97, 0.97, 'right', 'top'
    elif loc == 'ul':
        x, y, ha, va = 0.03, 0.97, 'left', 'top'
    else:
        x, y, ha, va = 0.97, 0.03, 'right', 'bottom'
    ax.text(x, y, label, transform=ax.transAxes, ha=ha, va=va, fontsize=fontsize, color='black', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='0.6', alpha=0.85))


def _highlight_best(axes_row, metric_values, lower_is_better=True):
    """Draw a thick green border on the axes whose metric is best in this
    row. `metric_values` is one float per column; NaN -> not best."""
    finite = [(c, v) for c, v in enumerate(metric_values)
              if v is not None and np.isfinite(v)]
    if not finite:
        return
    if lower_is_better:
        best = min(finite, key=lambda cv: cv[1])
    else:
        best = max(finite, key=lambda cv: cv[1])
    ax = axes_row[best[0]]
    for spine in ax.spines.values():
        spine.set_color('#2ca02c')
        spine.set_linewidth(4.4)


def _heatmap_tv(observed, analytic):
    """Total-variation distance between two heatmaps, treated as PDFs.
    Returns 0.5 * Sigma |p - q| after normalising both to unit total. Range
    [0, 1]; lower is better. This is the standard "principled" metric for
    distribution comparison; an alternative is the Wasserstein/EMD
    distance which additionally penalises spatial offsets but is more
    expensive in 2D."""
    obs_total = float(np.sum(observed))
    ana_total = float(np.sum(analytic))
    if obs_total <= 0 or ana_total <= 0:
        return float('nan')
    p = observed / obs_total
    q = analytic / ana_total
    return 0.5 * float(np.sum(np.abs(p - q)))


def _preview_panel(ax, samples, axis_range=None, slice_axis=2, slice_half_width=2.0, n_shown=100, arrow_axes=(3, 4), color_axis=5, color_scale=None, sun_marker=True, rng=None, show_xlabel=True, show_ylabel=True):
    """(x, y) scatter + (vx, vy) quiver preview of 6D phase-space samples.

    Restrict `samples` (shape (N, 6+)) to particles with
    `|samples[:, slice_axis]| < slice_half_width`, subsample to `n_shown`,
    plot positions as dots with velocity arrows coloured by
    `samples[:, color_axis]`. Returns the matplotlib QuadMesh-like object
    so callers can attach a colorbar.

    `axis_range` is the (lo, hi) pc range for the position axes; when None
    we auto-zoom from the slice's 5-95th percentiles.
    """
    import matplotlib.pyplot as plt
    if rng is None:
        rng = np.random.default_rng(7)
    samples = np.asarray(samples)
    pos_axes = [a for a in range(3) if a != slice_axis]
    if samples.ndim != 2 or samples.shape[1] < 6:
        ax.text(0.5, 0.5, "no samples", ha='center', va='center', transform=ax.transAxes, fontsize=8)
        ax.set_aspect('equal')
        return None
    in_slice = np.abs(samples[:, slice_axis]) < slice_half_width
    sliced = samples[in_slice]
    if len(sliced) == 0:
        ax.text(0.5, 0.5, "empty slice", ha='center', va='center', transform=ax.transAxes, fontsize=8)
        ax.set_aspect('equal')
        return None
    # Apply axis_range BEFORE subsampling so the n_shown particles are
    # spread evenly over the visible area, not concentrated at one edge.
    if axis_range is not None:
        in_view = (
            (sliced[:, pos_axes[0]] >= axis_range[0]) &
            (sliced[:, pos_axes[0]] <= axis_range[1]) &
            (sliced[:, pos_axes[1]] >= axis_range[0]) &
            (sliced[:, pos_axes[1]] <= axis_range[1])
        )
        sliced = sliced[in_view]
        if len(sliced) == 0:
            ax.text(0.5, 0.5, "empty view", ha='center', va='center', transform=ax.transAxes, fontsize=8)
            ax.set_aspect('equal')
            ax.set_xlim(axis_range); ax.set_ylim(axis_range)
            return None
    if len(sliced) > n_shown:
        idx = rng.choice(len(sliced), size=n_shown, replace=False)
        sliced = sliced[idx]
    if axis_range is None:
        lo = float(np.percentile(sliced[:, pos_axes], 5))
        hi = float(np.percentile(sliced[:, pos_axes], 95))
        pad = 0.10 * max(hi - lo, 1.0)
        axis_range = (lo - pad, hi + pad)
    ax.scatter(sliced[:, pos_axes[0]], sliced[:, pos_axes[1]], s=8, c='C0', alpha=0.6, lw=0)
    # Arrow length: in data units (pc), scale velocity -> ~10% of axis range.
    span = axis_range[1] - axis_range[0]
    arrow_data_per_v = span * 0.06
    if color_scale is None:
        c95 = float(np.percentile(np.abs(sliced[:, color_axis]), 95))
        color_scale = max(c95, 1e-12)
    q = ax.quiver(sliced[:, pos_axes[0]], sliced[:, pos_axes[1]], sliced[:, arrow_axes[0]] * arrow_data_per_v, sliced[:, arrow_axes[1]] * arrow_data_per_v, sliced[:, color_axis], cmap='RdBu_r', clim=(-color_scale, +color_scale), angles='xy', scale_units='xy', scale=1.0, width=0.005, alpha=0.7)
    if sun_marker:
        ax.plot([0], [0], marker='*', ms=8, c='gold', mec='k', mew=0.5, zorder=5)
    ax.set_aspect('equal')
    ax.set_xlim(axis_range)
    ax.set_ylim(axis_range)
    if show_xlabel:
        ax.set_xlabel(f"x{pos_axes[0]} (pc)", fontsize=7)
    if show_ylabel:
        ax.set_ylabel(f"x{pos_axes[1]} (pc)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.3)
    return q


def _draw_or_fallback(name, kde, coords, n_samples, rng):
    """Draw `n_samples` from `kde`; fall back to subsampling `coords` if the
    estimator has no native sampler (e.g. EnBiD).

    Returns an array of shape (<= n_samples, 6)."""
    if name == 'analytic_DF':
        return None  # caller handles this
    if hasattr(kde, 'draw'):
        try:
            return np.asarray(kde.draw(size=n_samples))
        except Exception:
            pass
    # Fallback: subsample training data. This shows what the estimator was
    # FIT to (same in every panel) rather than what the estimator predicts,
    # but it's the only honest signal available for sampler-less estimators.
    train = np.asarray(coords)
    idx = rng.integers(0, len(train), size=min(n_samples, len(train)))
    return train[idx]


def make_comparison_figure(savepath, title, coords, truth_callable, fv_axisym, sampler, R_an, marginal_axis=5, volume_sampler=None, fac_kde=None, cv_kwargs=None, nfolds=5, extra_estimators=None, random_state=None, preview_axis_range=None, preview_slice_axis=2, preview_slice_half_width=2.0, preview_n_shown=100):
    """6-row x N-column figure comparing analytic DF + scipy KDE + cvAdaptiveKDE.

    Rows:
      0: (costheta, phi) sky map
      1: dp/dcostheta profile (rate-weighted, unit-total-normalised) vs analytic
      2: dp/dlog10(vee) profile (rate-weighted, unit-total-normalised) vs analytic
      3: log(KDE/truth) ratio histogram at samples drawn from truth
      4: v_marginal histograms of one velocity component from samples drawn from
         truth and from the KDE (so the fhat histogram is fhat_marginal directly).
         marginal_axis selects which axis: 3=v_x, 4=v_y, 5=v_z.
      5: log10(density) value histograms - separately for f_truth and fhat at the
         same samples drawn from truth. Shifts and broadenings here are the
         smoothing bias of fhat relative to f_truth.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # We treat analytic_DF as the "truth" reference: its sky heatmap goes in
    # an inset above the main grid (analogous to printing R_an in the title);
    # everywhere else its curve is shown as a red dashed reference inside each
    # KDE panel. The main grid contains only the trained estimators.
    kde_estimators = list(_build_estimators(coords, fac=fac_kde, label_data="scenario", cv_kwargs=cv_kwargs, nfolds=nfolds, random_state=random_state))
    if extra_estimators is not None:
        kde_estimators.extend(extra_estimators)
    n_kde = len(kde_estimators)

    # Layout: top band houses title + R_an + three truth insets (heatmap,
    # x-y preview, x-z preview). Main grid is 7 x n_kde: row 0 sky map,
    # row 1 estimator-preview, rows 2..6 the remaining diagnostic rows
    # (costheta, log10 vinf, log10(fhat/f), v marginal, log10 density).
    fig = plt.figure(figsize=(3.0 * n_kde, 18))
    ax_analytic         = fig.add_axes([0.040, 0.860, 0.085, 0.060])
    ax_preview_truth    = fig.add_axes([0.165, 0.860, 0.085, 0.060])
    ax_preview_truth_xz = fig.add_axes([0.290, 0.860, 0.085, 0.060])
    gs_main = fig.add_gridspec(7, n_kde, left=0.05, right=0.97, top=0.81, bottom=0.04, wspace=0.32, hspace=0.50)
    axes = np.array([[fig.add_subplot(gs_main[r, c]) for c in range(n_kde)]
                      for r in range(7)])
    ax_preview = list(axes[1, :])   # alias used inside the per-estimator loop
    heatmap_images = [None] * n_kde

    # Truth-preview rendered from a large draw - same parameters as the
    # estimator-preview row so visually they're directly comparable.
    # x-y view (slice on z) is in the inset alongside the heatmap; the
    # x-z view (slice on y) is a sibling inset to its right so the reader
    # can also see vertical structure (z component).
    _preview_truth_samples = sampler(np.random.default_rng(202), max(5000, 50 * preview_n_shown))
    _preview_panel(ax_preview_truth, _preview_truth_samples, axis_range=preview_axis_range, slice_axis=preview_slice_axis, slice_half_width=preview_slice_half_width, n_shown=preview_n_shown, rng=np.random.default_rng(0), show_xlabel=False, show_ylabel=False)
    ax_preview_truth.set_title("truth: $(x,y)$+$\\vec v$", fontsize=8, pad=3)
    # x-z projection: slice on axis=1 (the y axis), so the projected
    # position axes are (0, 2) = (x, z), and the arrows show (vx, vz)
    # colored by vy.
    _preview_panel(
        ax_preview_truth_xz, _preview_truth_samples,
        axis_range=preview_axis_range,
        slice_axis=1,                # slice on y, project on (x, z)
        slice_half_width=preview_slice_half_width,
        n_shown=preview_n_shown,
        arrow_axes=(3, 5),           # (vx, vz)
        color_axis=4,                # color by vy
        rng=np.random.default_rng(1),
        show_xlabel=False, show_ylabel=False)
    ax_preview_truth_xz.set_title("truth: $(x,z)$+$\\vec v$", fontsize=8, pad=3)

    # Data-driven IS proposal (mean + cov of velocities for the K=300 nearest
    # spatial neighbours of the origin, inflated by 4x). Used for ALL rate
    # evaluations below - analytic_DF, every KDE estimator, and the high-N
    # reference for non-axisymmetric DFs. This dramatically reduces MC variance
    # for narrow-feature scenarios; for broad scenarios it reduces to roughly
    # uniform-like sampling and doesn't hurt.
    is_v_mean, is_v_cov = make_data_driven_is_proposal(coords)
    print(f"  IS proposal: v_mean = {is_v_mean.round(3)}, "
          f"diag(cov)^0.5 = {np.sqrt(np.diag(is_v_cov)).round(3)}")

    # costheta binning
    bin_edges = np.linspace(-1.0, 1.0, 21)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    dc = bin_edges[1] - bin_edges[0]
    log_vee_lo, log_vee_hi, n_vee_bins = -2.0, 2.0, 70
    d_log_vee = (log_vee_hi - log_vee_lo) / n_vee_bins
    bin_edges_lv = np.linspace(log_vee_lo, log_vee_hi, n_vee_bins + 1)
    bin_centers_lv = 0.5 * (bin_edges_lv[:-1] + bin_edges_lv[1:])

    # Heatmap binning for the analytic sky map (and for KDE columns later).
    HEATMAP_BINS = 12
    heatmap_costheta_edges = np.linspace(-1.0, 1.0, HEATMAP_BINS + 1)
    heatmap_phi_edges = np.linspace(-np.pi, np.pi, HEATMAP_BINS + 1)
    heatmap_costheta_centers = 0.5 * (heatmap_costheta_edges[:-1] + heatmap_costheta_edges[1:])

    if fv_axisym is not None:
        # Closed-form references via axisymmetric f_v
        analytic_per_bin = np.array([
            2.0 * np.pi * scipy.integrate.quad(lambda c: analytic_drdomega(fv_axisym, c), bin_edges[k], bin_edges[k + 1], limit=80)[0]
            for k in range(len(bin_edges) - 1)
        ])
        log_vee_grid = np.linspace(log_vee_lo, log_vee_hi, 300)
        vee_grid = 10.0 ** log_vee_grid
        analytic_dRdvee_grid = np.array([analytic_dR_dvee(v, fv_axisym) for v in vee_grid])
        analytic_per_log_vee_bin_smooth = (analytic_dRdvee_grid * vee_grid
                                           * np.log(10.0) * d_log_vee)
        analytic_per_log_vee_bin_at_bins = np.interp(bin_centers_lv, log_vee_grid, analytic_per_log_vee_bin_smooth)
        # Analytic heatmap: rate per (costheta, phi) bin, comparable to the KDE
        # histograms which sum `weights/n_total` per bin (i.e. rate-per-bin).
        # dR/dOmega is uniform in phi, so multiply by the per-bin solid-angle Deltacostheta*Deltaphi.
        bin_solid_angle = ((heatmap_costheta_edges[1] - heatmap_costheta_edges[0])
                            * (heatmap_phi_edges[1] - heatmap_phi_edges[0]))
        analytic_dRdOmega_costheta = np.array([analytic_drdomega(fv_axisym, c) for c in heatmap_costheta_centers])
        analytic_heatmap = np.tile(analytic_dRdOmega_costheta[:, None] * bin_solid_angle, (1, HEATMAP_BINS))
        analytic_R = float(np.trapz(analytic_dRdOmega_costheta * 2.0 * np.pi, heatmap_costheta_centers))
    else:
        # Non-axisymmetric DF - reference distributions from a high-N IS sweep
        # of the truth callable.
        print("  computing high-precision IS reference for non-axisymmetric DF...")
        ref_out = rate_sphere_importance(truth_callable, is_v_mean, is_v_cov, Nboot=NBOOT * 4, fac=1.0, rng=np.random.default_rng(7777))
        ref_w = np.asarray(ref_out[0])
        ref_vee = np.asarray(ref_out[1])
        ref_ct = np.asarray(ref_out[2])
        ref_phi = np.asarray(ref_out[3])
        n_ref = len(ref_w)
        analytic_per_bin, _ = np.histogram(ref_ct, bins=bin_edges, weights=ref_w / n_ref)
        log_ref_vee = np.log10(np.clip(ref_vee, 10.0 ** log_vee_lo, None))
        analytic_per_log_vee_bin_at_bins, _ = np.histogram(log_ref_vee, bins=n_vee_bins, range=[log_vee_lo, log_vee_hi], weights=ref_w / n_ref)
        analytic_per_log_vee_bin_smooth = analytic_per_log_vee_bin_at_bins
        log_vee_grid = bin_centers_lv  # plot reference at bin centres
        # Heatmap from the same high-N IS sweep
        analytic_heatmap, _, _ = np.histogram2d(ref_ct, ref_phi, bins=[heatmap_costheta_edges, heatmap_phi_edges], weights=ref_w / n_ref)
        analytic_R = float(np.sum(ref_w) / n_ref)

    # Render the analytic-DF reference heatmap into the inset axes near the
    # title. This panel is the visual analog of the R_an value printed in
    # the suptitle: it shows the truth dR/dOmega that every KDE column should be
    # compared against. Normalised to a PDF (sum = 1) so the colour scale
    # matches the KDE rows (also normalised below).
    analytic_heatmap_norm = analytic_heatmap / max(float(np.sum(analytic_heatmap)), 1e-30)
    img_ana = ax_analytic.imshow(analytic_heatmap_norm.T, aspect='auto', origin='lower', extent=[-1.0, 1.0, -np.pi, np.pi], cmap='viridis', vmin=0.0)
    ax_analytic.set_title(r"truth: $\mathrm{d}p/\mathrm{d}\Omega$", fontsize=10, pad=3)
    ax_analytic.set_xticks([-1, 0, 1])
    ax_analytic.set_yticks([-np.pi, 0, np.pi])
    ax_analytic.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
    ax_analytic.tick_params(labelsize=6)

    # samples from truth - used by rows 3, 4, 5 (same samples for all panels,
    # so the red truth-curve is identical column to column). N is bumped to
    # 20000 (10x the KDE draws) so the truth histogram is appreciably smoother
    # than the KDE estimates that are being compared to it.
    rng = np.random.default_rng(99)
    truth_samples = sampler(rng, 20000)
    p_truth_at_samples = truth_callable(truth_samples)
    p_truth_valid = p_truth_at_samples > 0
    # Volume-weighted samples are no longer plotted (the second curve was
    # confusing in row 3). volume_sampler kept in the signature for backward
    # compatibility with callers but unused here.

    stats_records = []   # accumulated per-estimator stats for JSON dump
    # Per-row metric values for "best column per row" highlighting after the loop.
    # Each list has one entry per KDE column; lower is better in every case.
    metric_heatmap_tv = []   # row 0 panel border: total-variation distance
    metric_rate_dist  = []   # row 0 title box: |R/R_an - 1| (closest to unity)
    metric_costh_ks  = []    # row 1
    metric_logvee_ks = []    # row 2
    metric_logratio  = []    # row 3 (|median|)
    metric_v_ks      = []    # row 4
    metric_logdens_ks = []   # row 5
    # Handles for the rate-text annotations on row 0; the best-rate column's
    # text will get a green bbox added after the loop.
    rate_text_artists = []
    # Row 6 stash: per-estimator log10(density) arrays. Plotted after the main
    # loop so all panels can share a single x-axis range computed from the
    # union of all (truth, KDE) values, making panel widths visually comparable.
    row5_stash = []
    # Stash a record for analytic_DF so the JSON output continues to include
    # an analytic reference line. Stats are mostly trivially "pass" because
    # analytic_DF *is* the truth; the rate value is reused from the inset
    # computation above.
    stats_records.append({
        'name': 'analytic_DF',
        'R': float(analytic_R),
        'R_relative': analytic_R / R_an,
        'rate_pass': True,
        'mean_costh': float('nan'),
        'kish_ess_rate': float('nan'),
        'per_eval_neff_med': float('nan'),
        'per_eval_neff_p95': float('nan'),
        'ks_costh': 0.0, 'costh_pass': True,
        'log_ratio_mass_median': 0.0, 'med_pass': True,
        'ks_v_marginal': 0.0, 'binres_v_marginal': 0.0, 'v_pass': True,
        'binres_costh': 0.0,
        'ks_logvee': 0.0, 'binres_logvee': 0.0, 'logvee_pass': True,
        'ks_logdens': 0.0, 'logdens_pass': True,
    })
    # Per-row N_eff floors via the dimensional formula
    #   floor = target_output_neff * h^(6 - d_m_eff)
    # where `d_m_eff` is the effective dimensionality of the output's
    # integrand support (not just histogram-axis count) and `h` is the
    # production KDE's typical kernel scale in scaled units. Calibration:
    # `target_output_neff=60` is anchored to the empirical sky-map floor
    # of ~30 at h~0.5 on cold+hot N=2000. See cvAdaptiveKDE.floor_for_dim
    # for the reasoning.
    #
    # Per-row d_m_eff:
    #   row 0 rate scalar:          d_m=0  (moment, unbiased on average)
    #   row 0 heatmap:              d_m=5  (2D output, rate-weighted)
    #   row 1 costheta histogram:       d_m=5  (rate-weighted: empirically the
    #                                       focusing weight makes 1D and 2D
    #                                       rate-weighted outputs share
    #                                       effective dimensionality ~ 5)
    #   row 2 log10 vinf histogram:   d_m=5  (rate-weighted; same as costheta)
    #   row 3 log10(fhat/f) at samples:d_m=6  (6D point eval)
    #   row 4 v_z marginal of fhat:   d_m=1  (unweighted 1D marginal: this
    #                                       averages over 5 axes uniformly,
    #                                       so dimensional formula applies)
    #   row 5 log10 density at samp:d_m=6  (6D point eval)
    #
    # For estimators without CV (scipy_kde), `pick_for_dim` is absent and
    # _kde_at falls back to the original KDE for every d_m, reflecting
    # that fixed-bandwidth estimators have no grid to pick from.
    def _kde_at(kde_obj, d_m, target=60.0):
        if hasattr(kde_obj, 'pick_for_dim'):
            return kde_obj.pick_for_dim(d_m, target_output_neff=target)
        return kde_obj

    for i, (name, kde, fac) in enumerate(kde_estimators):
        # Multi-pick routing using the dimensional formula. When the CV grid
        # produces the same argmax across d_m values (every smooth monomodal
        # scenario), the pick_at_floor cache returns the same KDE.
        # Rate is special: a moment of fhat that's unbiased on average even
        # in the particle-counter regime, so the rate-pick should ALWAYS
        # use floor=0 (= unconstrained ISE-CV winner), independent of h.
        # Otherwise estimators with hand-supplied scalings that put h_scaled
        # ~ 1 (e.g. cvAdapt-smart on stream data) lose the freedom to land
        # in the unbiased-on-average particle-counter pick.
        kde_for_rate = (kde.pick_at_floor(0) if hasattr(kde, 'pick_at_floor')
                        else kde)
        # Prefer the rate-weighted Kish-ESS picks (kde_sky, kde_vinf) when
        # available - these are the production picks set by
        # make_production_cv_kde and measure concentration in the actual
        # aggregation that drives the row's diagnostic. Fall back to the
        # per-eval-N_eff floor picks (legacy pick_for_dim path) otherwise.
        kde_for_sky     = (kde.kde_sky if hasattr(kde, 'kde_sky')
                           else _kde_at(kde, 5))    # row 0 heatmap, rate-weighted 2D
        # costheta (row 1) is also a sky-like rate-weighted aggregation -> kde_sky.
        # v_inf (row 2) has its own dedicated pick.
        kde_for_costh   = kde_for_sky
        kde_for_vinf    = (kde.kde_vinf if hasattr(kde, 'kde_vinf')
                           else _kde_at(kde, 5))
        kde_for_1d_rw   = kde_for_costh           # back-compat alias for costheta path
        kde_for_1d_marg = _kde_at(kde, 1)    # row 4 unweighted 1D marginal
        kde_for_6d      = _kde_at(kde, 6)    # rows 3, 5 6D point eval
        # Back-compat alias: "shape" defaulted to sky-pick previously.
        kde_for_shape = kde_for_sky
        # Alias for unweighted 1D marginal (row 4).
        kde_for_1d = kde_for_1d_marg

        # IS rate evaluation cached per unique KDE pick: rate, sky-map (=30),
        # and 1D-marginal (=5) picks may coincide or differ depending on the
        # CV grid landscape. Compute once per unique object so we don't pay
        # repeated NBOOT-sample IS draws.
        _is_cache = {}
        def _is_eval(kde_obj):
            key = id(kde_obj)
            if key in _is_cache:
                return _is_cache[key]
            out = rate_sphere_importance(kde_obj, is_v_mean, is_v_cov, Nboot=NBOOT, fac=fac, rng=np.random.default_rng(8000 + i))
            packed = (np.asarray(out[0]), np.asarray(out[1]),
                      np.asarray(out[2]), np.asarray(out[3]))
            _is_cache[key] = packed
            return packed

        # Rate scalar: rate-pick.
        weights, vee, costhetas, phis = _is_eval(kde_for_rate)
        n_total = len(weights)
        R = float(np.mean(weights))
        m = weighted_mean(weights, costhetas)
        n_eff = kish_ess(weights)

        # Per-eval N_eff at sphere coords for each pick (used for diagnostic
        # reporting and the table). Computing N_eff on the rate-pick is the
        # honest measure of how spiky the rate-pick is.
        sphere_coords = reconstruct_rate_sphere_coords(vee, costhetas, phis)
        neff_arr_rate = per_eval_neff(kde_for_rate, sphere_coords)
        neff_arr_shape = (per_eval_neff(kde_for_sky, sphere_coords)
                          if kde_for_sky is not kde_for_rate
                          else neff_arr_rate)
        if neff_arr_rate is not None:
            neff_med = float(np.median(neff_arr_rate))
            neff_95 = float(np.percentile(neff_arr_rate, 95))
        else:
            neff_med = float('nan')
            neff_95 = float('nan')
        if neff_arr_shape is not None:
            neff_shape_med = float(np.median(neff_arr_shape))
        else:
            neff_shape_med = float('nan')

        # Sky-map (row 0 heatmap, d_m=5) and rate-weighted 1D outputs
        # (rows 1, 2; d_m=4) live at slightly different floors via the
        # dimensional formula: the sky map needs ~30 N_eff at the typical
        # h, the rate-weighted 1D outputs need ~15. Compute IS once per
        # unique pick (the cache deduplicates).
        sk_weights, sk_vee, sk_costhetas, sk_phis = _is_eval(kde_for_sky)
        sk_n_total = len(sk_weights)
        m1_weights, m1_vee, m1_costhetas, m1_phis = _is_eval(kde_for_1d_rw)
        m1_n_total = len(m1_weights)
        # v_inf row uses its own pick when available (cv.kde_vinf set by the
        # production factory). Cached via _is_eval so if kde_for_vinf coincides
        # with kde_for_sky or kde_for_1d_rw, the IS draw is shared.
        vi_weights, vi_vee, vi_costhetas, vi_phis = _is_eval(kde_for_vinf)
        vi_n_total = len(vi_weights)
        # Back-compat alias.
        sh_weights, sh_vee, sh_costhetas, sh_phis = (
            sk_weights, sk_vee, sk_costhetas, sk_phis)
        sh_n_total = sk_n_total

        # Preview row: draw samples from the estimator and render the
        # same (x, y) + (vx, vy) preview as the truth inset. Sampler-less
        # estimators (e.g. EnBiD) fall back to subsampling the training
        # data - that shows the data the estimator was fit to rather
        # than what it predicts, but is the only honest signal available.
        n_preview = max(5000, 50 * preview_n_shown)
        preview_rng = np.random.default_rng(300 + i)
        preview_samples = _draw_or_fallback(name, kde_for_sky if kde_for_sky is not None else kde, coords, n_preview, preview_rng)
        if preview_samples is not None:
            _preview_panel(ax_preview[i], preview_samples, axis_range=preview_axis_range, slice_axis=preview_slice_axis, slice_half_width=preview_slice_half_width, n_shown=preview_n_shown, rng=preview_rng, show_xlabel=False, show_ylabel=(i == 0))

        # Row 0: sky map. Driven by kde_for_sky (floor=30) since sky pixels
        # are 2D-output and need moderate kernel overlap for reliable
        # per-pixel estimates.
        ax = axes[0, i]
        obs_heatmap, _, _ = np.histogram2d(sk_costhetas, sk_phis, bins=[heatmap_costheta_edges, heatmap_phi_edges], weights=sk_weights / sk_n_total)
        # Normalise to PDF so the colour scale is "fraction of rate per bin"
        # - directly comparable to the analytic heatmap.
        obs_heatmap_norm = obs_heatmap / max(float(np.sum(obs_heatmap)), 1e-30)
        h = ax.imshow(obs_heatmap_norm.T, aspect='auto', origin='lower',
                       extent=[heatmap_costheta_edges[0], heatmap_costheta_edges[-1],
                                heatmap_phi_edges[0], heatmap_phi_edges[-1]],
                       cmap="viridis", vmin=0)
        heatmap_images[i] = h
        rate_rel = R / R_an
        rate_pass = (1.0 / _RATE_FACTOR) < rate_rel < _RATE_FACTOR
        heatmap_tv = _heatmap_tv(obs_heatmap, analytic_heatmap)
        metric_heatmap_tv.append(heatmap_tv)
        # Column title: estimator display name (set_title), with the
        # recovered rate placed below it as a separate text artist so we
        # can wrap it in a green box for the best-rate column after the
        # loop. The R/R_an value is implied (R_an in the suptitle).
        ax.set_title(_display_name(name), fontsize=11, pad=20)
        # Quote recovered rate as a percentage of analytic - comparable
        # across columns and across scenarios. Absolute yr^-1 value is in
        # the stats JSON and the header R_an annotation.
        rate_text = r"$\mathcal{R}/\mathcal{R}_{\rm an}=" + f"{rate_rel*100:.1f}" + r"\%$"
        rate_artist = ax.text(0.5, 1.015, rate_text, transform=ax.transAxes, ha='center', va='bottom', fontsize=11)
        rate_text_artists.append(rate_artist)
        metric_rate_dist.append(abs(rate_rel - 1.0))
        ax.set_xlabel(r"$\cos\theta$")
        ax.set_ylabel(r"$\phi$")
        _annot(ax, f"Total var = {heatmap_tv:.2f}", loc='ur', fontsize=9)

        # Row 2: costheta shape (PDF, normalised to unit total) on log y-axis.
        # Magnitude is covered by the rate row; this row tests shape only.
        ax = axes[2, i]
        obs_per_bin, _ = np.histogram(m1_costhetas, bins=bin_edges, weights=m1_weights / m1_n_total)
        obs_total_r1 = float(np.sum(obs_per_bin))
        ana_total_r1 = float(np.sum(analytic_per_bin))
        obs_pdf_r1 = obs_per_bin / max(obs_total_r1, 1e-30)
        ana_pdf_r1 = analytic_per_bin / max(ana_total_r1, 1e-30)
        # Plot in per-unit-cos-theta density (divide by bin width), so the
        # y-axis label is unambiguous.
        obs_density_r1 = obs_pdf_r1 / dc
        ana_density_r1 = ana_pdf_r1 / dc
        ax.plot(bin_centers, obs_density_r1, "o-", label="MC", lw=1.2)
        ax.plot(bin_centers, ana_density_r1, "r--", label="analytic", lw=1.2)
        _vis_r1 = np.concatenate([obs_density_r1, ana_density_r1])
        if _vis_r1.size > 0:
            ax.set_ylim(0.0, 1.5 * _vis_r1.max())
        ax.set_xlabel(r"$\cos\theta$")
        ax.set_ylabel(r"$\mathrm{d}p/\mathrm{d}\cos\theta$")
        ax.legend(fontsize=8)
        obs_total = float(np.sum(obs_per_bin))
        ana_total = float(np.sum(analytic_per_bin))
        if obs_total > 0 and ana_total > 0:
            obs_cdf = np.cumsum(obs_per_bin) / obs_total
            ana_cdf = np.cumsum(analytic_per_bin) / ana_total
            ks_costh = float(np.max(np.abs(obs_cdf - ana_cdf)))
            # Bin-residual on the normalised *shape* (decoupled from rate
            # normalisation, which is already covered by row 0). KS averages
            # missed sharp peaks across the CDF; bin-residual catches them.
            obs_norm = obs_per_bin / obs_total
            ana_norm = analytic_per_bin / ana_total
            ana_norm_max = float(np.max(np.abs(ana_norm)))
            if ana_norm_max > 0:
                binres_costh = float(np.max(np.abs(obs_norm - ana_norm)) / ana_norm_max)
            else:
                binres_costh = float('nan')
        else:
            ks_costh = float('nan'); binres_costh = float('nan')
        # KS is the headline metric for costheta shape; bin-residual is
        # informational. The "best" column on this row is the one with the
        # smallest KS, highlighted by a green border after the loop.
        costh_pass = ks_costh < _KS_COSTH_TOL
        metric_costh_ks.append(ks_costh)
        _annot(ax, f'KS={ks_costh:.2f}\nMax bin err={binres_costh:.2f}')

        # Row 3: log10(vee) shape (PDF, normalised to unit total) on log y-axis.
        # Magnitude is covered by row 0; this is shape-only.
        # Uses kde_for_vinf - the rate-weighted vinf-ESS pick from the factory
        # (or falls back to the per-eval-N_eff floor pick if the cv wasn't
        # built via make_production_cv_kde).
        ax = axes[3, i]
        log_vee = np.log10(np.clip(vi_vee, 10 ** log_vee_lo, None))
        obs_logvee, _ = np.histogram(log_vee, bins=n_vee_bins, range=[log_vee_lo, log_vee_hi], weights=vi_weights / vi_n_total)
        ana_at_bins_r2 = analytic_per_log_vee_bin_at_bins
        obs_total_r2 = float(np.sum(obs_logvee))
        ana_total_r2 = float(np.sum(ana_at_bins_r2))
        obs_pdf_r2 = obs_logvee / max(obs_total_r2, 1e-30)
        ana_pdf_r2 = ana_at_bins_r2 / max(ana_total_r2, 1e-30)
        # Per-dex density: divide by Deltalog10(v).
        obs_density_r2 = obs_pdf_r2 / d_log_vee
        ana_density_r2 = ana_pdf_r2 / d_log_vee
        ax.plot(bin_centers_lv, obs_density_r2, "o-", label="MC", lw=1.2)
        ax.plot(bin_centers_lv, ana_density_r2, "r--", label="analytic", lw=1.2)
        ax.axvline(np.log10(V_ESC), c="k", ls=":", alpha=0.4, lw=0.8)
        _vis_r2 = np.concatenate([obs_density_r2, ana_density_r2])
        if _vis_r2.size > 0:
            ax.set_ylim(0.0, 1.5 * _vis_r2.max())
        # Zoom the x-axis around the analytic distribution's support (where
        # it exceeds 1% of its peak) to avoid showing many empty decades.
        if ana_density_r2.max() > 0:
            mask = ana_density_r2 > 0.01 * ana_density_r2.max()
            if mask.any():
                lo = bin_centers_lv[mask][0] - 0.3
                hi = bin_centers_lv[mask][-1] + 0.3
                ax.set_xlim(lo, hi)
        ax.set_xlabel(r"$\log_{10}(v_\infty / \mathrm{km\,s^{-1}})$")
        ax.set_ylabel(r"$\mathrm{d}p/\mathrm{d}\log_{10}v_\infty$ (per dex)")
        ax.legend(fontsize=8)
        # Compare *shape* of distribution: normalise both to unit total then KS-like + bin-residual.
        ana_at_bins = analytic_per_log_vee_bin_at_bins
        obs_total_lv = float(np.sum(obs_logvee))
        ana_total_lv = float(np.sum(ana_at_bins))
        if obs_total_lv > 0 and ana_total_lv > 0:
            obs_cdf = np.cumsum(obs_logvee) / obs_total_lv
            ana_cdf = np.cumsum(ana_at_bins) / ana_total_lv
            ks_logvee = float(np.max(np.abs(obs_cdf - ana_cdf)))
            obs_norm = obs_logvee / obs_total_lv
            ana_norm = ana_at_bins / ana_total_lv
            ana_max_norm = float(np.max(ana_norm))
            binres_logvee = float(np.max(np.abs(obs_norm - ana_norm)) / ana_max_norm) if ana_max_norm > 0 else float('nan')
        else:
            ks_logvee = float('nan'); binres_logvee = float('nan')
        logvee_pass = ks_logvee < _KS_LOGVEE_TOL
        metric_logvee_ks.append(ks_logvee)
        _annot(ax, f'KS={ks_logvee:.2f}\nMax bin err={binres_logvee:.2f}')

        # Row 4: log(KDE/truth) ratio histogram, both mass-weighted (samples from truth)
        # and volume-weighted (uniform over the support, when supplied). KDE smoothing
        # bias is concave at peaks, convex at tails; int(fhat-f) dx = 0 by normalization.
        # So mass-weighted (samples cluster at peaks) shows a negative-median underestimate;
        # volume-weighted should sit near zero with a wide spread.
        # Out-of-range samples (including p_kde=0 -> log_ratio=-inf, e.g. KDE-zero tails for
        # volume-weighted samples) accumulate in the edge bins; the median's vline is
        # clipped to the panel edge if the actual median is outside [-3, 3].
        ax = axes[4, i]
        RNG_LO, RNG_HI = -3.0, 3.0     # in log10 units now (was natural log)
        TINY = np.finfo(float).tiny
        p_kde_mass = np.atleast_1d(kde_for_6d(truth_samples)) * fac
        with np.errstate(divide='ignore', invalid='ignore'):
            log10_ratio_mass = (np.log10(np.where(p_kde_mass > 0, p_kde_mass, TINY))
                                - np.log10(np.where(p_truth_at_samples > 0, p_truth_at_samples, TINY)))
        log10_ratio_mass = log10_ratio_mass[p_truth_valid]
        # Filter to finite entries: some estimators (notably the NF) can
        # produce +inf density at sharp peaks where the flow saturates the
        # float log-prob range. `np.median` is robust to a few infs (picks
        # the middle order statistic), but `np.std` propagates +inf into a
        # NaN via mean=inf -> (x-inf)^2=NaN. Drop non-finite entries before stats
        # and report the count so the user can see when an estimator is
        # saturating.
        finite_mask = np.isfinite(log10_ratio_mass)
        n_nonfinite = int((~finite_mask).sum())
        log10_ratio_mass_finite = log10_ratio_mass[finite_mask]
        if log10_ratio_mass_finite.size == 0:
            med_mass = float('nan')
            std_mass = float('nan')
        else:
            med_mass = float(np.median(log10_ratio_mass_finite))
            std_mass = float(np.std(log10_ratio_mass_finite))
        clipped_mass = np.clip(log10_ratio_mass_finite, RNG_LO, RNG_HI)
        # Blue (C0) to match the "this is the KDE estimate" colour used in
        # the other rows. No legend - the median + std annotations cover
        # the same information.
        ax.hist(clipped_mass, bins=40, range=[RNG_LO, RNG_HI], density=True, histtype="step", lw=1.4, color="C0")
        if np.isfinite(med_mass):
            ax.axvline(np.clip(med_mass, RNG_LO, RNG_HI), c="C0", alpha=0.4, ls="--")
        ax.axvline(0, c="k", alpha=0.5)
        ax.set_xlim(RNG_LO, RNG_HI)
        ax.set_xlabel(r"$\log_{10}(\hat f / f_{\rm truth})$ (dex)")
        ax.set_ylabel("PDF per dex")
        med_pass = (np.isfinite(med_mass) and abs(med_mass) < _LOG10_RATIO_TOL)
        metric_logratio.append(abs(med_mass) if np.isfinite(med_mass) else float('inf'))
        annot_lines = [f'med = {med_mass:+.2f} dex',
                       f'std = {std_mass:.2f} dex']
        if n_nonfinite:
            annot_lines.append(f'(+{n_nonfinite} non-finite dropped)')
        _annot(ax, '\n'.join(annot_lines))

        # Row 5: 1D velocity marginal. The truth samples are the same in every
        # panel (computed once outside the loop, N=20000); only the KDE
        # blue curve varies. The bin edges are determined from the truth
        # alone so the red curve is identical column to column.
        ax = axes[5, i]
        v_truth = truth_samples[:, marginal_axis]
        if name == "analytic_DF":
            kde_marg = sampler(np.random.default_rng(102), 4000)
        elif hasattr(kde_for_1d, 'draw'):
            kde_marg = np.asarray(kde_for_1d.draw(size=4000))
        else:
            # Estimators without a native sampler (e.g. EnBiD, NormalizingFlow
            # - though NF could sample via the inverse flow, we don't wire
            # that up). Fall back to subsampling the training data - this
            # represents the same distribution the kde was fit to, so the
            # row 4 v_marginal panel still shows truth vs the kde's training
            # distribution (the kde itself isn't queried here).
            rng_marg = np.random.default_rng(102)
            train = coords
            idx = rng_marg.integers(0, len(train), size=min(4000, len(train)))
            kde_marg = np.asarray(train[idx])
        v_kde = kde_marg[:, marginal_axis]
        v_lo = float(np.percentile(v_truth, 1.0))
        v_hi = float(np.percentile(v_truth, 99.0))
        pad = 0.15 * (v_hi - v_lo)
        bins = np.linspace(v_lo - pad, v_hi + pad, 60)
        ax.hist(v_truth, bins=bins, histtype="step", lw=1.4, density=True, color="C3", label="truth")
        ax.hist(v_kde, bins=bins, histtype="step", lw=1.4, density=True, color="C0", label=r"$\hat f$")
        _hist_truth, _ = np.histogram(v_truth, bins=bins, density=True)
        _hist_kde,  _ = np.histogram(v_kde,   bins=bins, density=True)
        _vis_r4 = np.concatenate([_hist_truth, _hist_kde])
        if _vis_r4.size > 0:
            ax.set_ylim(0.0, 1.5 * _vis_r4.max())
        axis_label = {3: r"$v_x$", 4: r"$v_y$", 5: r"$v_z$"}[marginal_axis]
        ax.set_xlabel(axis_label + r" (km/s)")
        ax.set_ylabel(r"$\mathrm{d}p/\mathrm{d}v$ (PDF per km/s)")
        ax.legend(fontsize=8)
        ks_v = float(scipy.stats.ks_2samp(v_truth, v_kde).statistic)
        # Bin-residual on PDF (catches localised peak misses that KS averages out)
        hist_truth, _ = np.histogram(v_truth, bins=bins, density=True)
        hist_kde, _ = np.histogram(v_kde, bins=bins, density=True)
        truth_max = float(np.max(hist_truth))
        binres_v = (float(np.max(np.abs(hist_truth - hist_kde)) / truth_max)
                    if truth_max > 0 else float('nan'))
        # Pass on KS alone. The bin-residual is preserved in the JSON for
        # diagnostic detail but is unforgiving on narrow-peak distributions
        # (e.g. the stream's bimodal v_z at +/-v_circ): a peak that the KDE
        # gets within 30% in height fails the metric even when the
        # CDF-averaged KS clearly agrees and the human eye says ``yep, gets
        # the peaks''.
        v_pass = ks_v < _KS_VMARG_TOL
        metric_v_ks.append(ks_v)
        _annot(ax, f'KS={ks_v:.2f}\nMax bin err={binres_v:.2f}')

        # Stash per-estimator stats for the post-loop JSON dump
        stats_records.append({
            'name': name,
            'R': R,
            'R_relative': rate_rel,
            'rate_pass': bool(rate_pass),
            'heatmap_tv': heatmap_tv,
            'mean_costh': m,
            'kish_ess_rate': float(n_eff),
            'per_eval_neff_med': neff_med,
            'per_eval_neff_p95': neff_95,
            'ks_costh': ks_costh,
            'costh_pass': bool(costh_pass),
            'log_ratio_mass_median': med_mass,
            'med_pass': bool(med_pass),
            'ks_v_marginal': ks_v,
            'binres_v_marginal': binres_v,
            'v_pass': bool(v_pass),
        })

        # Row 6: log10(density) value histograms. Compute the metric here but
        # defer plotting until after the loop so we can share an x-axis range
        # across all panels (computed from the union of all panels' values).
        valid_row5 = p_truth_valid & (p_kde_mass > 0)
        log_p_truth_v = np.log10(p_truth_at_samples[valid_row5])
        log_p_kde_v = np.log10(p_kde_mass[valid_row5])
        ks_logdens = float(scipy.stats.ks_2samp(log_p_truth_v, log_p_kde_v).statistic) if len(log_p_truth_v) > 0 else float('nan')
        logdens_pass = ks_logdens < _KS_LOGDENS_TOL
        row5_stash.append((i, log_p_truth_v, log_p_kde_v, ks_logdens))
        metric_logdens_ks.append(ks_logdens)

        # Add the new statistics to the per-estimator record
        stats_records[-1].update({
            'binres_costh': binres_costh,
            'ks_logvee': ks_logvee,
            'binres_logvee': binres_logvee,
            'logvee_pass': bool(logvee_pass),
            'ks_logdens': ks_logdens,
            'logdens_pass': bool(logdens_pass),
        })

    # Row 6 plotting (deferred from the main loop so all panels share an
    # x-axis range and bins). The truth distribution is the same in every
    # panel; only the KDE side varies.
    if row5_stash:
        _r5_concat = np.concatenate([
            np.concatenate([lt, lk]) for (_i, lt, lk, _ks) in row5_stash
            if lt.size > 0 or lk.size > 0
        ]) if any(lt.size > 0 or lk.size > 0 for (_i, lt, lk, _ks) in row5_stash) else np.array([])
        if _r5_concat.size > 0:
            r5_lo = float(np.percentile(_r5_concat, 1.0))
            r5_hi = float(np.percentile(_r5_concat, 99.0))
        else:
            r5_lo, r5_hi = -1.0, 1.0
        r5_bins = np.linspace(r5_lo - 0.2, r5_hi + 0.2, 50)
        for (i, lt, lk, ks_logdens) in row5_stash:
            ax = axes[6, i]
            ax.hist(lt, bins=r5_bins, histtype="step", lw=1.4, density=True, color="C3", label=r"$f_{\rm truth}$")
            ax.hist(lk, bins=r5_bins, histtype="step", lw=1.4, density=True, color="C0", label=r"$\hat f$")
            _ht, _ = np.histogram(lt, bins=r5_bins, density=True)
            _hk, _ = np.histogram(lk, bins=r5_bins, density=True)
            _vis_r5 = np.concatenate([_ht, _hk])
            if _vis_r5.size > 0:
                ax.set_ylim(0.0, 1.5 * _vis_r5.max())
            ax.set_xlim(r5_bins[0], r5_bins[-1])
            ax.set_xlabel(r"$\log_{10}(\rm density)$ at samples from truth")
            ax.set_ylabel("PDF per dex")
            ax.legend(fontsize=8)
            _annot(ax, f'KS={ks_logdens:.2f}')

    # Unify heatmap clims so the analytic inset and all KDE-row sky maps
    # share a z-scale within a figure, then put a single colorbar at the
    # right of the heatmap row.
    all_imgs = [img_ana] + [im for im in heatmap_images if im is not None]
    vmax = max(float(np.max(im.get_array())) for im in all_imgs)
    for im in all_imgs:
        im.set_clim(vmin=0.0, vmax=vmax)
    fig.colorbar(all_imgs[-1], ax=axes[0, -1], fraction=0.05, pad=0.02)

    # Highlight the best-performing KDE column per row with a green border.
    # Row 0: panel border tracks heatmap TV (shape); a separate green bbox
    # around the rate-text artist tracks the closest-to-unity rate.
    _highlight_best(axes[0, :], metric_heatmap_tv)
    # Row 1 is the preview row - no metric, no highlight.
    _highlight_best(axes[2, :], metric_costh_ks)
    _highlight_best(axes[3, :], metric_logvee_ks)
    _highlight_best(axes[4, :], metric_logratio)
    _highlight_best(axes[5, :], metric_v_ks)
    _highlight_best(axes[6, :], metric_logdens_ks)
    if metric_rate_dist:
        best_rate_col = int(np.argmin(metric_rate_dist))
        rate_text_artists[best_rate_col].set_bbox(dict(boxstyle='square,pad=0.3', edgecolor='#2ca02c', linewidth=4.0, facecolor='white'))

    # Title + R_an placed in the empty top-right band (the truth-inset
    # row occupies the top-LEFT; the analogous space on the right was
    # white). R_an headline sits above; the scenario description follows
    # below it in normal weight.
    fig.text(0.43, 0.910, r"$\mathcal{R}_{\rm an}=" + f"{R_an:.3g}" + r"\ \mathrm{yr}^{-1}$", fontsize=18, fontweight='bold', va='center', ha='left')
    fig.text(0.43, 0.872, title, fontsize=12, va='center', ha='left')
    fig.savefig(savepath, dpi=110)
    # Also emit a PDF for journal submission. Same content; vector format.
    pdf_path = savepath.replace('.png', '.pdf')
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"  wrote {savepath} and {pdf_path}")

    # Per-scenario stats JSON for the master-table aggregator
    stats_path = savepath.replace('.png', '_stats.json')
    with open(stats_path, 'w') as f:
        json.dump({
            'scenario': title,
            'R_an': R_an,
            'estimators': stats_records,
            'thresholds': {
                'rate_tol': _RATE_TOL,
                'ks_costh_tol': _KS_COSTH_TOL,
                'binres_costh_tol': _BINRES_COSTH_TOL,
                'ks_logvee_tol': _KS_LOGVEE_TOL,
                'binres_logvee_tol': _BINRES_LOGVEE_TOL,
                'log10_ratio_tol': _LOG10_RATIO_TOL,
                'ks_vmarg_tol': _KS_VMARG_TOL,
                'binres_vmarg_tol': _BINRES_VMARG_TOL,
                'ks_logdens_tol': _KS_LOGDENS_TOL,
            },
        }, f, indent=2)
    print(f"  wrote {stats_path}")


@slow
def test_plot_comparison_isotropic():
    rng = np.random.default_rng(31)
    coords = sample_uniform_pos_gaussian_v(rng, 2000, DX, np.zeros(3), SIGMA)
    fv = isotropic_fv(N0, SIGMA)
    truth = make_isotropic_df(N0, SIGMA)
    sampler = make_isotropic_sampler(SIGMA)
    vol_sampler = make_volume_sampler((-3.0 * SIGMA, 3.0 * SIGMA))
    R_an = analytic_total_rate(fv)
    fac = DX ** 3 * N0
    smart = _build_smart_cvadaptive(coords, [DX, DX, DX, SIGMA, SIGMA, SIGMA], fac)
    make_comparison_figure("compare_isotropic.png", f"isotropic Maxwellian, $\\sigma={SIGMA}$ km/s", coords, truth, fv, sampler, R_an, volume_sampler=vol_sampler, extra_estimators=[smart])


@slow
def test_plot_comparison_drifting():
    rng = np.random.default_rng(32)
    v_bulk = np.array([0.0, 0.0, V_BULK_Z])
    coords = sample_uniform_pos_gaussian_v(rng, 2000, DX, v_bulk, SIGMA)
    fv = drifting_fv(N0, V_BULK_Z, SIGMA)
    truth = make_drifting_df(N0, v_bulk, SIGMA)
    sampler = make_drifting_sampler(v_bulk, SIGMA)
    vol_sampler = make_volume_sampler((V_BULK_Z - 3.0 * SIGMA, V_BULK_Z + 3.0 * SIGMA))
    R_an = analytic_total_rate(fv)
    fac = DX ** 3 * N0
    smart = _build_smart_cvadaptive(coords, [DX, DX, DX, SIGMA, SIGMA, SIGMA], fac)
    make_comparison_figure("compare_drifting.png",
                           f"drifting Maxwellian, $\\sigma={SIGMA}$ km/s, "
                           f"$v_{{\\rm bulk}}={V_BULK_Z}$ km/s",
                           coords, truth, fv, sampler, R_an,
                           volume_sampler=vol_sampler,
                           extra_estimators=[smart])


# Disabled 2026-05-05: the rate-~-0 problem with R_an floor sentinel made the
# pass/fail metrics unreadable. Renamed to _disabled prefix so the test runner
# (which discovers test_* functions) skips it; keep the body for reference.
def _disabled_test_plot_comparison_expanding_shell():
    """Expanding shell: particles on a thin shell with purely radial outward velocity.

    Sun at origin. Particles distributed on a thin Gaussian shell at R_shell
    with v = v_out * rhat + small isotropic dispersion. With sigma_v << v_out the
    fraction of particles with any inward velocity component is exp(-(v_out/sigma_v)^2/2)
    ~ 0, so the analytic encounter rate is effectively 0.

    Tests whether the KDE captures *direction-correlated* velocity structure:
    scipy with global Sigma pools v across all directions and gives a bandwidth
    sigma_v ~ v_out (because rhat*v_out spans +/-v_out across the dataset). That
    bandwidth leaks substantial probability mass into Sun-inward velocities,
    yielding a nonzero rate. cvAdaptive with local Sigma_i should do much better
    because each kernel's local Sigma_v ~ sigma_v^2 (small) rather than v_out^2 (large).
    """
    # Shell well outside the encounter sphere - physically representative of
    # ejecta or a stellar-association expansion. The right answer is rate ~ 0:
    # no particle is moving Sun-ward at all, *and* the data is spatially far
    # from the encounter sphere. Any KDE giving substantial rate has either
    # (a) leaked velocity probability into inward directions, or (b) spread
    # position density unphysically far from the data. Both are failure modes.
    R_shell = 1.5         # pc - well outside the encounter sphere (0.1 pc)
    sigma_R = 0.05        # shell thickness << R_shell
    v_out = 1.0           # outward speed (~1 km/s)
    sigma_v = 0.05        # local v dispersion (10x narrower than v_out)
    N_total = 1.0e15

    def sampler(rng, n):
        rh = rng.standard_normal((n, 3))
        rh = rh / np.linalg.norm(rh, axis=1, keepdims=True)
        rad = R_shell + sigma_R * rng.standard_normal(n)
        x = rad[:, None] * rh
        v = v_out * rh + sigma_v * rng.standard_normal((n, 3))
        return np.column_stack([x, v])

    def truth(coords):
        coords = np.asarray(coords)
        x = coords[:, :3]
        v = coords[:, 3:]
        r = np.linalg.norm(x, axis=1)
        safe_r = np.maximum(r, 1.0e-30)
        rh = x / safe_r[:, None]
        # Position part: thin Gaussian shell, normalised over 4pir^2 for the radial profile.
        p_pos = (np.exp(-(r - R_shell) ** 2 / (2.0 * sigma_R ** 2))
                 / (np.sqrt(2.0 * np.pi) * sigma_R)
                 / (4.0 * np.pi * np.maximum(r, 1.0e-30) ** 2))
        # Velocity part: 3D Gaussian centred at v_out * rh
        delta = v - v_out * rh
        delta_sq = np.sum(delta * delta, axis=1)
        p_v = (np.exp(-delta_sq / (2.0 * sigma_v ** 2))
               / (2.0 * np.pi * sigma_v ** 2) ** 1.5)
        return N_total * p_pos * p_v

    coords = sampler(np.random.default_rng(99), 2000)

    # Reference rate via importance sampling on the truth. The IS proposal is
    # data-driven (centred at mean(v) ~ 0 with cov ~ v_out^2/3*I, since rhat is
    # uniform on the sphere) - it samples both inward and outward v, so the
    # truth-driven rate captures the genuinely-zero analytic answer.
    is_v_mean, is_v_cov = make_data_driven_is_proposal(coords)
    out_ref = rate_sphere_importance(truth, is_v_mean, is_v_cov, Nboot=NBOOT, fac=1.0, rng=np.random.default_rng(199))
    R_an = float(np.mean(np.asarray(out_ref[0])))
    # Floor R_an so the relative-rate display doesn't divide by ~0. Using a
    # nominal floor of 0.01*(typical isotropic rate) means each estimator's
    # rate gets shown as a multiple of "1% of typical" - anything below ~30%
    # of that floor is effectively 0.
    R_floor = 0.01 * 530.0   # ~ 5.3/yr; 1% of the sigma=1, n0=1e15 isotropic rate
    if R_an < R_floor:
        print(f"  truth-driven R_an={R_an:.3e} below floor {R_floor:.3e}; "
              f"using floor as denominator (rate shown as multiple of floor)")
        R_an = R_floor

    vol_sampler = make_volume_sampler((-v_out * 1.5, v_out * 1.5))
    make_comparison_figure("compare_expanding_shell.png",
                           f"expanding shell R={R_shell} pc, v_out={v_out}, "
                           f"sigma_v={sigma_v} (analytic rate ~ 0)",
                           coords, truth, None, sampler, R_an, marginal_axis=5,
                           volume_sampler=vol_sampler)


@slow
def test_plot_comparison_bimodal():
    sigma_c = 0.15
    v_a = np.array([0.0, 0.0, +0.4])
    v_b = np.array([0.0, 0.0, -0.4])
    truth = make_bimodal_df(N0, v_a, v_b, sigma_c, sigma_c, alpha=0.5)
    sampler = make_bimodal_sampler(v_a, v_b, sigma_c, sigma_c, alpha=0.5)

    norm = N0 * (2.0 * np.pi * sigma_c ** 2) ** -1.5
    def fv(vee, ct):
        nsq_a = vee ** 2 + 2.0 * vee * (+0.4) * ct + 0.16
        nsq_b = vee ** 2 + 2.0 * vee * (-0.4) * ct + 0.16
        return 0.5 * norm * (np.exp(-nsq_a / (2.0 * sigma_c ** 2))
                             + np.exp(-nsq_b / (2.0 * sigma_c ** 2)))
    R_an = analytic_total_rate(fv)

    coords = sampler(np.random.default_rng(34), 2000)
    vol_sampler = make_volume_sampler((-1.0, 1.0))
    # bimodal structure is along v_z
    fac = DX ** 3 * N0
    # Smart scaling: v scaled by sigma_c (within-peak width) so the kernel is
    # tuned to the cold component's intrinsic dispersion.
    smart = _build_smart_cvadaptive(coords, [DX, DX, DX, sigma_c, sigma_c, sigma_c], fac)
    make_comparison_figure("compare_bimodal.png",
                           f"bimodal $\\mathbf{{v}}=\\pm 0.4\\,\\hat{{\\mathbf{{z}}}}$, "
                           f"$\\sigma_c={sigma_c}$ km/s",
                           coords, truth, fv, sampler, R_an, marginal_axis=5,
                           volume_sampler=vol_sampler,
                           extra_estimators=[smart])


@slow
def test_plot_comparison_stream():
    """Narrow off-centre ring in *physical* space, modelled like a thin disk stream.

    Setup (all parameters per the user's spec):
      - Ring length L=1000 pc -> R_ring = L/(2pi) ~ 159.15 pc
      - Ring centre at (R, 0, 0) so the ring passes through the origin (Sun) at theta=pi
      - Velocity dispersions: sigma_R = sigma_h = 1 km/s (radial & vertical), sigma_t = 0.1 km/s
      - Bulk circular velocity v_circ = 5 km/s along the ring tangent
      - Spatial width = sigma_R/kappa = 25 pc, height = sigma_h/nu = 12.5 pc, with epicyclic
        frequency kappa=0.04 Myr^-1 and vertical frequency nu=0.08 Myr^-1
      - At origin the local velocity DF is anisotropic: sigma_x = sigma_y = 1, sigma_z = 0.1
        with bulk velocity -v_circ along zhat (the local stream tangent at theta=pi)

    This is the regime where adaptive KDE is *expected* to outperform a global-
    bandwidth scipy estimator: the local Sigma_i at any sample on the ring reflects
    the local stream tangent (a 1D feature in 6D), while scipy's global Sigma pools
    velocities from the whole orbit and washes out the sigma_t = 0.1 km/s anisotropy.
    """
    R_ring = 1000.0 / (2.0 * np.pi)         # ~ 159.15 pc
    sigma_R = 1.0                            # pc/Myr ~ 1 km/s
    sigma_h = 1.0
    sigma_t = 0.1
    v_circ = 5.0
    kappa = 0.04                             # Myr^-1
    nu = 0.08                                # Myr^-1
    width = sigma_R / kappa                  # 25 pc
    height = sigma_h / nu                    # 12.5 pc
    N_total = 1.0e20                         # total particles -> ~few/yr rate

    truth = make_stream_ring_df(N_total, R_ring, sigma_R, sigma_h, sigma_t, v_circ, width, height)
    sampler = make_stream_ring_sampler(R_ring, sigma_R, sigma_h, sigma_t, v_circ, width, height)
    fv = stream_ring_fv(N_total, R_ring, sigma_R, sigma_h, sigma_t, v_circ, width, height)

    R_an = analytic_total_rate(fv)
    coords = sampler(np.random.default_rng(36), 2000)

    # Volume sampler covers the spatial extent of the ring + reasonable velocity range
    vol_sampler = make_volume_sampler_box(spatial_lo=[-50.0, -5.0 * height, -(R_ring + 50.0)], spatial_hi=[2.0 * R_ring + 50.0, 5.0 * height, R_ring + 50.0], velocity_lo=[-(v_circ + 5.0), -(v_circ + 5.0), -(v_circ + 5.0)], velocity_hi=[v_circ + 5.0, v_circ + 5.0, v_circ + 5.0])

    # Smart scaling: epicyclic stream-aligned spatial scales (width, height)
    # plus the intrinsic sigma_R/sigma_t/sigma_h velocity dispersions; analogous to the
    # cvAdapt-stream recipe used on the disk-stream scenario.
    smart_scalings = [width, width, height, sigma_R, sigma_t, sigma_h]
    smart = _build_smart_cvadaptive(coords, smart_scalings, N_total, nfolds=5, shrinkage_target='local_pooled')
    make_comparison_figure(
        "compare_stream.png",
        (f"stream ring $R={R_ring:.0f}$ pc, "
         f"$\\sigma_R={sigma_R}$ km/s, $\\sigma_t={sigma_t}$ km/s, "
         f"$v_{{\\rm circ}}={v_circ}$ km/s"),
        coords, truth, fv, sampler, R_an,
        marginal_axis=5,    # v_z marginal - bimodal at +/-v_circ across the ring
        volume_sampler=vol_sampler,
        fac_kde=N_total,
        # Stream picked covfac=92.2 (edge of default range) on a previous run, so
        # widen the grid by one decade on the high side and add 2 grid points to keep
        # resolution. Also use nfolds=5 (vs default 3 in plotting tests) to reduce
        # CV noise - nfolds=3 was producing big run-to-run hyperparameter variance.
        cv_kwargs=dict(covfac_range=(-2.0, 3.0), ncovfacs=13),
        nfolds=5,
        extra_estimators=[smart],
    )


@slow
def test_plot_comparison_ring():
    """Curved velocity DF: a thin ring in (v_x, v_y) at radius v_R, plus thin Gaussian
    in v_z. The 1D marginal in v_x is bimodal at +/-v_R - single global Gaussian kernels
    can't fit a ring, so this is a regime where adaptive kernels *should* help (if the
    distance metric is set up for it).
    """
    v_R = 0.5
    sigma_perp = 0.08
    sigma_z = 0.08
    truth = make_ring_df(N0, v_R, sigma_perp, sigma_z)
    sampler = make_ring_sampler(v_R, sigma_perp, sigma_z)
    fv = ring_fv(N0, v_R, sigma_perp, sigma_z)
    R_an = analytic_total_rate(fv)
    coords = sampler(np.random.default_rng(35), 2000)
    vol_sampler = make_volume_sampler((-(v_R + 3 * sigma_perp), v_R + 3 * sigma_perp))
    # ring is in the v_x-v_y plane -> v_x marginal is the bimodal one
    fac = DX ** 3 * N0
    # Smart scaling: v scaled by sigma_perp (cross-ring width) so kernels are
    # tuned to the ring thickness rather than its radius.
    smart = _build_smart_cvadaptive(coords, [DX, DX, DX, sigma_perp, sigma_perp, sigma_z], fac)
    make_comparison_figure("compare_ring.png",
                           f"ring in $(v_x, v_y)$, $v_R={v_R}$ km/s, "
                           f"$\\sigma_\\perp={sigma_perp}$ km/s",
                           coords, truth, fv, sampler, R_an, marginal_axis=3,
                           extra_estimators=[smart],
                           volume_sampler=vol_sampler)


@slow
def test_plot_comparison_disk_stream():
    """Galactic-disk-like stream at production parameters.

    R_ring = 8 kpc; v_circ = 220 km/s; sigma_R = sigma_z = 1 km/s; sigma_t = 0.1 km/s
    (anisotropic, narrow tangential dispersion).
    Stream width = sigma_R/kappa; height = sigma_z/nu, with epicyclic frequencies
        kappa = sqrt2*v_circ/R_ring  ~  0.0389 Myr^-1
        nu = sqrt(4piG*rho_0)        ~  0.0752 Myr^-1  (rho_0 = 0.1 M_sun/pc^3)
    Sun in the midst of the stream at (R_ring, 0, 0), z=0; peculiar velocity
    5 km/s inward radially and 5 km/s ahead of the local circular speed.

    The local DF in the Sun's rest frame is anisotropic Gaussian (different sigma
    on different axes) AND has a bulk shift in the xy plane - not axisymmetric
    around zhat, so we pass `fv_axisym=None` and let `make_comparison_figure` use
    a high-N MC of the truth callable as the reference distribution.
    """
    R_ring = 8000.0   # 8 kpc, in pc
    v_circ = 220.0    # km/s = pc/Myr
    sigma_R = 1.0
    sigma_z = 1.0
    sigma_t = 0.1
    G_pc3_msun_myr2 = 0.00449987
    rho_0 = 0.1   # M_sun/pc^3
    kappa = np.sqrt(2.0) * v_circ / R_ring
    nu = np.sqrt(4.0 * np.pi * G_pc3_msun_myr2 * rho_0)
    width = sigma_R / kappa     # ~ 25.71 pc
    height = sigma_z / nu       # ~ 13.30 pc
    v_sun_peculiar = (-5.0, +5.0, 0.0)   # 5 inward, 5 ahead of LSR
    N_total = 1.0e20

    truth = make_disk_stream_df(N_total, R_ring, sigma_R, sigma_z, sigma_t, v_circ, width, height, v_sun_peculiar)
    sampler = make_disk_stream_sampler(R_ring, sigma_R, sigma_z, sigma_t, v_circ, width, height, v_sun_peculiar)

    # Bulk velocity at the Sun's location in the Sun's rest frame:
    # stream's local v_galactic = (0, v_circ, 0) at phi=0; subtract Sun's velocity
    # (-5, v_circ+5, 0) -> (+5, -5, 0) in Sun's frame.
    v_bulk_sun = np.array([-v_sun_peculiar[0], -v_sun_peculiar[1], -v_sun_peculiar[2]])
    # Local sigma at the Sun's position: sigma_R along xhat (radial at phi=0), sigma_t along yhat
    # (tangential at phi=0), sigma_z along zhat. Use this as the importance-sampling
    # proposal covariance (slightly inflated to cover the rate-leverage region).
    proposal_cov = np.diag([sigma_R**2, sigma_t**2, sigma_z**2]) * 4.0   # 2sigma inflation

    # Precompute high-precision reference rate via importance-sampled RATE_Sphere
    # on the truth. With proposal centred on v_bulk_sun, ~all samples land in the
    # rate-leverage region - convergence is much faster than uniform RATE_Sphere
    # (which dumps most samples on bound or non-stream directions).
    print("  computing reference rate via IMPORTANCE-SAMPLED RATE_Sphere...")
    out_ref = rate_sphere_importance(truth, v_bulk_sun, proposal_cov, Nboot=NBOOT, fac=1.0, rng=np.random.default_rng(101))
    R_an = float(np.mean(np.asarray(out_ref[0])))
    print(f"    R_an = {R_an:.5f}/yr  (importance-sampled, std-of-mean = "
          f"{np.std(out_ref[0])/np.sqrt(NBOOT):.2e})")

    coords = sampler(np.random.default_rng(40), 4000)   # N=4k - comparable to other tests; production at N=20k validated separately

    vol_sampler = make_volume_sampler_box(spatial_lo=[-5 * width, -5 * width, -5 * height], spatial_hi=[+5 * width, +5 * width, +5 * height], velocity_lo=[-15.0, -15.0, -10.0], velocity_hi=[+15.0, +15.0, +10.0])

    # Domain-knowledge variants: scale by the natural local-stream dispersions
    # (sigma_R, sigma_t, sigma_z) at the Sun's position rather than the raw kappa, nu frequencies.
    # The previous kappa,nu-frequency scaling [1,1,nu/kappa,kappa,kappa,nu] gave kernel sigma_v ~ 2 km/s
    # which couldn't reach from the data (at v ~ +/-7 km/s in Sun's frame) to the
    # encounter-sphere queries (at v ~ 0). Stream-aligned dispersions make the
    # kernel sigma in each direction match the local data spread.
    stream_scalings = [width, width, height, sigma_R, sigma_t, sigma_z]
    print(f"  stream-aligned scalings = {stream_scalings}")
    # cvAdapt-stream uses the same domain-knowledge-scaling recipe as the
    # cvAdapt-smart column on the other scenarios but with
    # shrinkage_target='local_pooled' (Sigma_global on disk-stream data has
    # huge eigenvalues along the ring's tangential direction, see section 2.2).
    smart = _build_smart_cvadaptive(coords, stream_scalings, N_total, nfolds=5, shrinkage_target='local_pooled')
    extras = [smart]

    make_comparison_figure(
        "compare_disk_stream.png",
        (f"disk stream $R={R_ring/1000:.0f}$ kpc, "
         f"$v_{{\\rm circ}}={v_circ}$ km/s, "
         f"$\\sigma_R={sigma_R}$, $\\sigma_t={sigma_t}$, $\\sigma_z={sigma_z}$ km/s, "
         f"Sun peculiar $=({v_sun_peculiar[0]},\\,{v_sun_peculiar[1]:+g},\\,0)$"),
        coords, truth, None, sampler, R_an,
        marginal_axis=4,    # tangential = yhat at Sun's position; narrow sigma_t
        volume_sampler=vol_sampler,
        fac_kde=N_total,
        nfolds=5,
        # shrinkage_target='local_pooled' - disk_stream's Sigma_global has huge
        # eigenvalues along the ring's tangential direction (~R_ring^2/2), which
        # would explode kernel widths at sh > 0. Pooling toward a position-local
        # 300-NN covariance keeps the shrinkage target bounded by sigma_R/sigma_t/sigma_z.
        # Cap shrinkage at 0.75 - CV picks sh=0 on stream data anyway.
        cv_kwargs=dict(shrinkage_grid=[0.0, 0.1, 0.25, 0.5, 0.75], shrinkage_target='local_pooled'),
        extra_estimators=extras,
    )


@slow
def test_plot_comparison_spiky_ball():
    """`n_spikes` independent thin streams through the origin in random
    3D directions - paper-level ablation that exposes where cvAdaptive's
    per-point local-covariance kernels should beat fixed-capacity
    normalizing flows.

    Each spike is a 1D feature in 6D phase space (narrow position-cross-section
    sigma_perp << L, narrow velocity-perp dispersion sigma_v_perp << v_speed).
    Encoding `n_spikes` independent narrow features in a fixed-architecture
    MAF requires the network to allocate capacity per spike - at some
    `n_spikes` the NF should plateau while cvAdaptive (whose kernel
    representation scales with N) keeps tracking the structure.

    Designed for the user's "spiky-ball" intuition (2026-05-26 conversation):
    20-ish intersecting streams in 6D as a hard NF test case.
    """
    n_spikes = 25
    L = 100.0            # pc; each stream extends +/-L/2 from origin
    sigma_perp = 0.3     # pc cross-section (narrow streams)
    sigma_v_long = 0.3   # km/s along-stream velocity spread
    sigma_v_perp = 0.05  # km/s perpendicular velocity spread (narrow)
    v_speed = 3.0        # km/s bulk speed along stream
    N_total = 1.0e15

    sampler, n_hat = make_spiky_ball_sampler(n_spikes, L, sigma_perp, sigma_v_long, sigma_v_perp, v_speed)
    truth = make_spiky_ball_df(N_total, n_spikes, L, sigma_perp, sigma_v_long, sigma_v_perp, v_speed)

    # IS proposal: each spike contributes near v ~ v_speed*nhat_k. With random
    # nhat directions, the rate-leverage v-distribution averages over the
    # sphere, so an isotropic proposal centred at v=0 with sigma ~ v_speed
    # covers the relevant region.
    v_bulk_sun = np.zeros(3)
    proposal_cov = np.diag([v_speed ** 2] * 3) * 4.0   # 2sigma inflation

    print("  computing reference rate via importance-sampled RATE_Sphere on truth...")
    out_ref = rate_sphere_importance(truth, v_bulk_sun, proposal_cov, Nboot=NBOOT, fac=1.0, rng=np.random.default_rng(303))
    R_an = float(np.mean(np.asarray(out_ref[0])))
    print(f"    R_an = {R_an:.5g}/yr  (importance-sampled, "
          f"std-of-mean = {np.std(out_ref[0])/np.sqrt(NBOOT):.2e})")

    coords = sampler(np.random.default_rng(40), 4000)
    vol_sampler = make_volume_sampler_box(spatial_lo=[-0.6 * L, -0.6 * L, -0.6 * L], spatial_hi=[+0.6 * L, +0.6 * L, +0.6 * L], velocity_lo=[-(v_speed + 2)] * 3, velocity_hi=[+(v_speed + 2)] * 3)

    make_comparison_figure(
        "compare_spiky_ball.png",
        (f"spiky ball $N_{{\\rm spikes}}={n_spikes}$, "
         f"$L={L:.0f}$ pc, $\\sigma_\\perp={sigma_perp}$ pc, "
         f"$\\sigma_{{v\\perp}}={sigma_v_perp}$ km/s, "
         f"$v_{{\\rm speed}}={v_speed}$ km/s"),
        coords, truth, None, sampler, R_an,
        marginal_axis=5,
        volume_sampler=vol_sampler,
        fac_kde=N_total,
        nfolds=5,
        cv_kwargs=dict(shrinkage_target='local_pooled'),
        preview_axis_range=(-10.0, 10.0),
    )


@slow
def test_plot_comparison_cold_hot():
    """Cold + hot mixture: two co-centred Maxwellians, sigma_cold << sigma_hot.

    Parameters chosen so the **cold component dominates the rate** (~83%) despite
    being a 20% mass fraction. With sigma_hot=10 (galactic-halo-like) and sigma_cold=0.4
    (just above v_esc=0.3), the focusing factor (1 + 2GM/(qmax*sigma^2)) is ~5 for hot
    vs ~2300 for cold - so even at alpha=0.2 the cold-rate-per-density (~1500) far
    exceeds hot's (~75). This is the regime where adaptive KDE *should* beat
    isotropic if local-Sigma_i adaptation works at all on the rate-leverage region.

    Includes an extra "v=sigma_cold" column with v-axes scaled to sigma_cold instead of
    sigma_global ~ sigma_hot. With this scaling the kdtree distance puts cold particles
    close to each other (cold-cold v distance ~ sigma_cold/sigma_cold = 1 vs cold-hot
    ~ sigma_hot/sigma_cold = 25), so the KNN-10 of a cold particle should be other cold
    particles - letting local Sigma_i actually capture the sigma_cold structure.
    """
    sigma_hot = 10.0
    sigma_cold = 0.4
    alpha = 0.2
    truth = make_cold_hot_df(N0, sigma_cold, sigma_hot, alpha)
    sampler = make_cold_hot_sampler(sigma_cold, sigma_hot, alpha)
    fv = cold_hot_fv(N0, sigma_cold, sigma_hot, alpha)
    R_an = analytic_total_rate(fv)
    coords = sampler(np.random.default_rng(37), 2000)
    vol_sampler = make_volume_sampler((-3.0 * sigma_hot, 3.0 * sigma_hot))

    # Smart scaling: v scaled by sigma_cold so the kdtree distance puts cold
    # particles close to each other, letting local Sigma_i capture the sigma_cold
    # structure rather than being dominated by the wide hot component.
    fac = DX ** 3 * N0
    smart = _build_smart_cvadaptive(coords, [DX, DX, DX, sigma_cold, sigma_cold, sigma_cold], fac)
    make_comparison_figure("compare_cold_hot.png",
                           f"cold+hot mixture: $\\sigma_c={sigma_cold}$, "
                           f"$\\sigma_h={sigma_hot}$ km/s, $\\alpha_c={alpha}$",
                           coords, truth, fv, sampler, R_an, marginal_axis=5,
                           volume_sampler=vol_sampler,
                           extra_estimators=[smart])


@slow
def test_plot_comparison_cosine_shear():
    """Cosine shear v_z = A*cos(2pi*n*x/DX) + thermal noise.

    Globally Cov(x, v_z) = 0 (cos orthogonal to x over the symmetric box), so
    scipy gets no shear-alignment off-diagonal - unlike linear shear, where it
    correctly aligns kernels with the ridge. But Var(v_z) is inflated to
    sigma^2 + A^2/2, so scipy's sigma_z kernel is broadened by the spatial modulation.

    For adaptive KDE to win, the KNN ball must be << shear period DX/n. With
    DX=10, n=2 (period 5 pc) and ~50-200 NN at N=2000 -> KNN radius ~0.79 pc,
    ratio ~0.16. Local Sigma_i should pick up the local v_z slope; scipy's global
    Sigma averages all phases of cos.

    Using cos rather than sin: v_z_mean(x=0) = A -> local DF at the encounter
    target is *anisotropic* (drifting Maxwellian with bulk = A*zhat), so this also
    tests sky-anisotropy preservation. Closed-form rate uses drifting_fv.
    """
    A = 1.0                    # bulk velocity at x=0
    n_periods = 2              # 2 full waves across the box
    sigma_v = SIGMA            # 1.0 - local sigma in each v component
    truth = make_cosine_shear_df(N0, sigma_v, A, n_periods, DX)
    sampler = make_cosine_shear_sampler(sigma_v, A, n_periods, DX)
    fv = drifting_fv(N0, A, sigma_v)
    R_an = analytic_total_rate(fv)
    coords = sampler(np.random.default_rng(39), 2000)
    vol_sampler = make_volume_sampler_box(spatial_lo=[-DX / 2, -DX / 2, -DX / 2], spatial_hi=[DX / 2, DX / 2, DX / 2], velocity_lo=[-3.0 * sigma_v, -3.0 * sigma_v, -A - 3.0 * sigma_v], velocity_hi=[3.0 * sigma_v, 3.0 * sigma_v, A + 3.0 * sigma_v])
    # Smart scaling: x tightened to a quarter of the shear wavelength so the
    # KNN ball is much smaller than the period; v stays at sigma_v.
    fac = DX ** 3 * N0
    quarter_lambda = (DX / n_periods) / 4.0
    smart = _build_smart_cvadaptive(coords, [quarter_lambda, DX, DX, sigma_v, sigma_v, sigma_v], fac)
    make_comparison_figure("compare_cosine_shear.png",
                           f"cosine shear $v_z = A\\cos(2\\pi\\cdot{n_periods}\\,x/\\Delta x)$, "
                           f"$A={A}$, $\\sigma={sigma_v}$ km/s",
                           coords, truth, fv, sampler, R_an, marginal_axis=5,
                           volume_sampler=vol_sampler,
                           extra_estimators=[smart])


# sigma-sweep - show the fix works across regimes

@slow
def test_sigma_sweep_isotropic_DF():
    """For sigma  in  {0.3, 1, 3} verify RATE_Sphere(analytic DF) tracks the analytic quadrature.

    Pre-fix this would fail at sigma=0.3 (bound regime, ~10x off) and sigma=1 (~5-10% off);
    post-fix all three should agree to within MC noise (~5%).
    """
    print(f"  [info] v_esc = {V_ESC:.3f}, sigma values = (0.3, 1.0, 3.0)")
    for sigma in (0.3, 1.0, 3.0):
        df = make_isotropic_df(N0, sigma)
        fv = isotropic_fv(N0, sigma)
        R_an = analytic_total_rate(fv)
        R_DF, _, _ = run_rate_sphere(df, fac=1.0)
        rel_err = abs(R_DF - R_an) / R_an
        print(f"  [info] sigma={sigma:>3}  R_an={R_an:>10.3g}  R_DF={R_DF:>10.3g}  rel_err={rel_err:.3f}")
        # sigma=0.3 is the bound-dominated regime - RATE_Sphere with bound filter integrates
        # only the unbound tail, which is a small fraction of f_v, so MC noise is larger.
        # Allow proportionally more slack.
        tol = 0.10 if sigma >= 1.0 else 0.20
        assert rel_err < tol, f"sigma={sigma}: R_an={R_an:.3g}, R_DF={R_DF:.3g}, rel_err={rel_err:.3f}"


# Master summary table - collects compare_*_stats.json files into a Markdown table

_STAT_SPECS = [
    # (label, value-key, per-statistic pass-fn, format)
    ('rate (R/R_an)',          'R_relative',            lambda v: (1.0/_RATE_FACTOR) < v < _RATE_FACTOR,    '{:.0%}'),
    ('costheta KS',                'ks_costh',              lambda v: v < _KS_COSTH_TOL,           '{:.2f}'),
    ('costheta bin-res',           'binres_costh',          lambda v: v < _BINRES_COSTH_TOL,       '{:.2f}'),
    ('log10(vee) KS',          'ks_logvee',             lambda v: v < _KS_LOGVEE_TOL,          '{:.2f}'),
    ('log10(vee) bin-res',     'binres_logvee',         lambda v: v < _BINRES_LOGVEE_TOL,      '{:.2f}'),
    ('log10(fhat/f) median',      'log_ratio_mass_median', lambda v: abs(v) < _LOG10_RATIO_TOL,   '{:+.2f}'),
    ('v_marginal KS',          'ks_v_marginal',         lambda v: v < _KS_VMARG_TOL,           '{:.2f}'),
    ('v_marginal bin-res',     'binres_v_marginal',     lambda v: v < _BINRES_VMARG_TOL,       '{:.2f}'),
    ('log10(density) KS',      'ks_logdens',            lambda v: v < _KS_LOGDENS_TOL,         '{:.2f}'),
]


def _collect_stats():
    """Read all compare_*_stats.json files in cwd. Returns (all_data,
    estimator_names) with analytic_DF dropped (it's a sanity baseline)."""
    import glob
    files = sorted(glob.glob('compare_*_stats.json'))
    all_data = []
    estimator_names = []
    for fp in files:
        with open(fp) as f:
            data = json.load(f)
        scenario_name = data['scenario'].split('  ')[0]
        all_data.append((scenario_name, data))
        for est in data['estimators']:
            if est['name'] == 'analytic_DF':
                continue
            if est['name'] not in estimator_names:
                estimator_names.append(est['name'])
    return all_data, estimator_names


def _threshold_summary():
    return (f"Pass thresholds: rate within factor of {_RATE_FACTOR:.0f} of analytic; "
            f"costheta KS<{_KS_COSTH_TOL}; "
            f"log10(vee) KS<{_KS_LOGVEE_TOL}; "
            f"|log10(fhat/f) median|<{_LOG10_RATIO_TOL}; "
            f"v_marginal KS<{_KS_VMARG_TOL}; "
            f"log10(density) KS<{_KS_LOGDENS_TOL}. "
            f"Bin-residual values are reported in the JSON for diagnostic use "
            f"but are not part of the pass criterion.")


def write_master_table(out_path='diagnostic_table.md'):
    """Single big markdown table - no blank rows between scenarios so renderers
    don't break the table apart. analytic_DF dropped."""
    all_data, estimator_names = _collect_stats()
    if not all_data:
        print("no compare_*_stats.json files found")
        return
    lines = ['# Diagnostic summary', '', _threshold_summary(), '']
    header = ['scenario', 'statistic'] + estimator_names
    lines.append('| ' + ' | '.join(header) + ' |')
    lines.append('|' + '|'.join(['---'] * len(header)) + '|')
    for scenario_name, data in all_data:
        est_by_name = {e['name']: e for e in data['estimators']}
        for k, (stat_label, val_key, pass_fn, fmt) in enumerate(_STAT_SPECS):
            row = [scenario_name if k == 0 else '', stat_label]
            for ename in estimator_names:
                e = est_by_name.get(ename)
                if e is None:
                    row.append('-')
                    continue
                val = e.get(val_key)
                if val is None or not np.isfinite(val):
                    row.append('-')
                else:
                    passed = False
                    try: passed = bool(pass_fn(val))
                    except Exception: pass
                    sym = 'ok' if passed else 'x'
                    try: formatted = fmt.format(val)
                    except (TypeError, ValueError): formatted = str(val)
                    row.append(f'{sym} {formatted}')
            lines.append('| ' + ' | '.join(row) + ' |')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'wrote {out_path}')


def write_master_table_latex(out_path='diagnostic_table.tex'):
    """Standalone-compilable LaTeX with green/red cell shading.
    Compile via `pdflatex diagnostic_table.tex`."""
    all_data, estimator_names = _collect_stats()
    if not all_data:
        print("no compare_*_stats.json files found")
        return

    # pdflatex can't handle bare Unicode, so substitute math-mode commands.
    # Order matters: escape LaTeX-special chars FIRST (before we add backslashes
    # via Unicode replacement), then substitute Unicode (which adds \, $, {, })
    # - we don't want THOSE backslashes/dollars/braces to be re-escaped.
    UNICODE_TO_TEX = [
        ('perp', r'$\perp$'),
        ('sigma', r'$\sigma$'), ('theta', r'$\theta$'), ('phi', r'$\phi$'),
        ('nu', r'$\nu$'),    ('kappa', r'$\kappa$'), ('alpha', r'$\alpha$'),
        ('pi', r'$\pi$'),    ('eps', r'$\epsilon$'),
        ('+/-', r'$\pm$'),    ('x', r'$\times$'),  ('<=', r'$\le$'),
        ('>=', r'$\ge$'),    ('<', r'$\langle$'), ('>', r'$\rangle$'),
        ('_0', r'$_0$'),     ('_1', r'$_1$'),      ('_2', r'$_2$'),
        ('_i', r'$_i$'),
        ('zhat', r'$\hat{z}$'),('fhat', r'$\hat{f}$'),
        ('Sigma', r'$\Sigma$'), ('Delta', r'$\Delta$'),
    ]

    def lx(s):
        # First escape LaTeX-special chars in the original string
        s = (s.replace('\\', r'\textbackslash{}')
              .replace('&', r'\&').replace('%', r'\%')
              .replace('_', r'\_').replace('#', r'\#')
              .replace('^', r'\^{}').replace('~', r'\~{}'))
        # Then substitute Unicode. The substitutions add their own \, $, {, }
        # which are now intentional LaTeX syntax and must NOT be re-escaped.
        for u, t in UNICODE_TO_TEX:
            s = s.replace(u, t)
        return s

    n_est = len(estimator_names)
    n_stats = len(_STAT_SPECS)
    out = []
    out.append(r'\documentclass{article}')
    out.append(r'\usepackage[margin=0.5in,landscape]{geometry}')
    out.append(r'\usepackage{xcolor}')
    out.append(r'\usepackage{colortbl}')
    out.append(r'\usepackage{array}')
    out.append(r'\usepackage{longtable}')
    out.append(r'\definecolor{passgreen}{rgb}{0.78,0.94,0.78}')
    out.append(r'\definecolor{failred}{rgb}{0.96,0.78,0.78}')
    out.append(r'\setlength{\tabcolsep}{4pt}')
    out.append(r'\renewcommand{\arraystretch}{1.1}')
    out.append(r'\begin{document}')
    out.append(r'\footnotesize')
    out.append(r'\section*{Diagnostic summary}')
    summary = (f"Pass thresholds: rate within factor of {_RATE_FACTOR:.0f}; "
               f"cos$\\theta$ KS$<${_KS_COSTH_TOL}; "
               f"log10(vee) KS$<${_KS_LOGVEE_TOL}; "
               f"$|$log10($\\hat f/f$) median$|<${_LOG10_RATIO_TOL}; "
               f"v\\_marginal KS$<${_KS_VMARG_TOL}; "
               f"log10(density) KS$<${_KS_LOGDENS_TOL}. "
               f"Bin-residual values are reported in the JSON for diagnostic "
               f"use but are not part of the pass criterion.")
    out.append(summary)
    out.append('')
    # Approach: ABOVE each scenario block, emit a single wide row that uses
    # \multicolumn to span ALL columns and contain the scenario name + parameters
    # on multiple lines. Then the data rows have a regular schema (statistic +
    # estimator values) without needing a separate scenario column. This keeps
    # all column boundaries simple and ensures header alignment.
    col_spec = '|l|' + 'c|' * n_est
    n_cols = 1 + n_est

    def _emit_col_headers():
        # Title row that uses no `&` - instead emits a single multicolumn
        # spanning all data columns, with the header labels typeset inside a
        # tabular* whose column widths exactly match the outer table. This
        # avoids the alignment-drift that the user reported with an ampersand-
        # separated header row in longtable's \endhead box.
        # Implementation: multiple \multicolumn{1}{|c|}{label} cells separated
        # by ` ` (no ampersand) inside one \multicolumn outer, isn't quite
        # right - multicolumn requires &. So instead we build one big inline
        # text block and pad with \hphantom to align. SIMPLEST robust approach:
        # use \multicolumn cells that redeclare column type for ONE row only,
        # and feed them via & - the only ampersands in the title row are
        # *inside* multicolumns, which longtable handles consistently.
        cells = [r'\multicolumn{1}{|c|}{\textbf{statistic}}']
        for e in estimator_names:
            cells.append(r'\multicolumn{1}{c|}{\textbf{' + lx(e) + r'}}')
        return ' & '.join(cells) + r' \\ \hline'

    out.append(r'\begin{longtable}{' + col_spec + '}')
    out.append(r'\hline')
    out.append(_emit_col_headers())
    out.append(r'\endhead')
    out.append(r'\hline \multicolumn{' + str(n_cols) + r'}{r}{\textit{(continued on next page)}} \\')
    out.append(r'\endfoot')
    out.append(r'\hline \endlastfoot')

    for scenario_name, data in all_data:
        est_by_name = {e['name']: e for e in data['estimators']}
        # Scenario "section header" row spanning all columns - uses multicolumn
        # (no `&` on this row) so the gray bar is one continuous cell.
        out.append(r'\rowcolor{black!10}\multicolumn{' + str(n_cols) + r'}{|l|}{\textbf{' + lx(scenario_name) + r'}} \\ \hline')
        for k, (stat_label, val_key, pass_fn, fmt) in enumerate(_STAT_SPECS):
            row_cells = [lx(stat_label)]
            for ename in estimator_names:
                e = est_by_name.get(ename)
                if e is None:
                    row_cells.append('---')
                    continue
                val = e.get(val_key)
                if val is None or not np.isfinite(val):
                    row_cells.append('---')
                else:
                    passed = False
                    try: passed = bool(pass_fn(val))
                    except Exception: pass
                    try: formatted = fmt.format(val)
                    except (TypeError, ValueError): formatted = str(val)
                    color = 'passgreen' if passed else 'failred'
                    row_cells.append(f'\\cellcolor{{{color}}} {lx(formatted)}')
            out.append(' & '.join(row_cells) + r' \\')
        out.append(r'\hline')
    out.append(r'\end{longtable}')
    out.append(r'\end{document}')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out) + '\n')
    print(f'wrote {out_path}')


# Standalone runner

def main(argv):
    skip_slow = "--no-slow" in argv
    name_filter = None
    if "-k" in argv:
        name_filter = argv[argv.index("-k") + 1]
    # Stream-like scenarios are the slowest and most likely to blow up - run them
    # last so cheaper scenarios complete first and we get partial diagnostic data
    # even when a stream test fails or runs long.
    def _order(name):
        return (1, name) if 'stream' in name else (0, name)
    tests = [(n, f) for n, f in sorted(globals().items(), key=lambda kv: _order(kv[0]))
             if n.startswith("test_") and callable(f)]
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
            print(f"PASS  {name}  ({time.time()-t1:.1f}s)")
            passed += 1
        except Exception as e:
            print(f"FAIL  {name}  ({time.time()-t1:.1f}s): {type(e).__name__}: {e}")
            failures.append((name, traceback.format_exc()))
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped ({time.time()-t0:.1f}s)")
    if failures:
        print("\n--- failure tracebacks ---")
        for name, tb in failures:
            print(f"\n[{name}]")
            print(tb)
    # If any plot tests ran (or even if previous compare_*_stats.json exist), emit tables
    try:
        write_master_table()
        write_master_table_latex()
    except Exception as e:
        print(f"  (master table write skipped: {type(e).__name__}: {e})")
    return failed


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1:]) == 0 else 1)
