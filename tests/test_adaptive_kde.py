"""
Tests for the adaptiveKDE class in isostreams.py.

Focus: the locally-adaptive 6D phase-space KDE (joint, marginal, conditional,
sample-drawing). Ground truth is a 6D Gaussian where every density has a
closed form. Several tests are explicit regressions for the bugs identified
during the 2026-04 audit.

Run:
    ipython3 test_adaptive_kde.py                # all tests
    ipython3 test_adaptive_kde.py --no-slow      # skip Monte-Carlo integrations
    ipython3 test_adaptive_kde.py -k regression  # only tests whose name matches 'regression'

The file is also pytest-compatible if you ever install pytest:
    ipython3 -m pytest -- test_adaptive_kde.py
"""
import sys
import time
import traceback

import numpy as np
import scipy.stats

from cracked import adaptiveKDE, mockScipyKde


# --- ground-truth Gaussian

def _ground_truth_cov():
    rng = np.random.default_rng(7)
    A = rng.standard_normal((6, 6)) * 0.3
    return A @ A.T + np.eye(6)


GROUND_MEAN = np.array([1.0, -2.0, 0.5, 0.1, 0.2, -0.1])
GROUND_COV = _ground_truth_cov()


# --- lazy module-level fixtures

_samples = None
_kde_unit = None
_kde_scaled = None


def get_samples():
    global _samples
    if _samples is None:
        rng = np.random.default_rng(42)
        _samples = rng.multivariate_normal(GROUND_MEAN, GROUND_COV, size=1500)
    return _samples


def get_kde_unit():
    global _kde_unit
    if _kde_unit is None:
        _kde_unit = adaptiveKDE(get_samples().copy(), scalings=np.ones(6), nn=50, use_multiprocessing=False)
    return _kde_unit


def get_kde_scaled():
    global _kde_scaled
    if _kde_scaled is None:
        _kde_scaled = adaptiveKDE(get_samples().copy(), scalings=[2.0, 2.0, 2.0, 0.5, 0.5, 0.5], nn=50, use_multiprocessing=False)
    return _kde_scaled


def _analytic_conditional(x):
    """Analytic conditional p(v|x) for the ground-truth Gaussian."""
    mu_x, mu_v = GROUND_MEAN[:3], GROUND_MEAN[3:]
    Sxx, Svv, Svx = GROUND_COV[:3, :3], GROUND_COV[3:, 3:], GROUND_COV[3:, :3]
    Sxx_inv = np.linalg.inv(Sxx)
    mu_vgx = mu_v + Svx @ Sxx_inv @ (x - mu_x)
    cov_vgx = Svv - Svx @ Sxx_inv @ Svx.T
    return mu_vgx, cov_vgx


def _mc_integrate(integrand, sampler, n, seed=0):
    """Importance-sampled integral: int f dx ~ mean(f / w) under sampler w.

    Seeded by default so MC-integration tests are reproducible. covalpha=+0.3
    in particular has high importance-sampler variance (positive alpha widens
    kernels in low-density regions, fattening the tails the IS doesn't cover).
    """
    pts = sampler.rvs(size=n, random_state=seed)
    return float(np.mean(integrand(pts) / sampler.pdf(pts)))


# --- markers

def slow(fn):
    fn._slow = True
    return fn


# Pytest compatibility: if pytest is installed, also expose these as marker decorators
# so that `pytest -m "not slow"` works. (No-op when running via main() below.)
try:
    import pytest as _pytest  # noqa: F401
    slow = _pytest.mark.slow  # type: ignore
except ImportError:
    pass


# Constructor & shape sanity

def test_constructor_shapes():
    kde = get_kde_unit()
    assert kde.data.shape[1] == 6
    assert kde.covariances.shape == (kde.data.shape[0], 6, 6)
    assert kde.choleskys.shape == (kde.data.shape[0], 6, 6)


def test_jacfac_factors():
    for kde in (get_kde_unit(), get_kde_scaled()):
        assert np.isclose(kde.jacfac, kde.jacfac_marg * kde.jacfac_cond)


def test_jacfac_unit_scales():
    kde = get_kde_unit()
    assert np.isclose(kde.jacfac, 1.0)
    assert np.isclose(kde.jacfac_marg, 1.0)
    assert np.isclose(kde.jacfac_cond, 1.0)


def test_jacfac_nonunit_scales():
    kde = get_kde_scaled()
    assert np.isclose(kde.jacfac_marg, 8.0)    # 2*2*2
    assert np.isclose(kde.jacfac_cond, 0.125)  # 0.5*0.5*0.5


def test_call_returns_positive_density():
    kde = get_kde_unit()
    rng = np.random.default_rng(0)
    pts = rng.multivariate_normal(GROUND_MEAN, GROUND_COV, size=20)
    p = kde(pts)
    assert p.shape == (20,)
    assert np.all(p > 0)


def test_evaluate_marginal_shape():
    kde = get_kde_unit()
    rng = np.random.default_rng(1)
    x_pts = rng.multivariate_normal(GROUND_MEAN[:3], GROUND_COV[:3, :3], size=10)
    p = kde.evaluate_marginal(x_pts)
    assert p.shape == (10,)
    assert np.all(p > 0)


def test_evaluate_conditional_shape():
    kde = get_kde_unit()
    rng = np.random.default_rng(2)
    v_pts = rng.multivariate_normal(GROUND_MEAN[3:], GROUND_COV[3:, 3:], size=10)
    p = kde.evaluate_conditional(v_pts, GROUND_MEAN[:3])
    assert p.shape == (10,)


# Density agreement with the analytic Gaussian

def test_joint_density_against_truth():
    kde = get_kde_unit()
    rng = np.random.default_rng(3)
    pts = rng.multivariate_normal(GROUND_MEAN, GROUND_COV, size=300)
    p_kde = kde(pts)
    p_true = scipy.stats.multivariate_normal(GROUND_MEAN, GROUND_COV).pdf(pts)
    log_ratio = np.log(p_kde) - np.log(p_true)
    assert abs(np.median(log_ratio)) < 0.5, f"median log ratio {np.median(log_ratio)}"
    assert np.std(log_ratio) < 1.0


def test_marginal_density_against_truth():
    kde = get_kde_unit()
    rng = np.random.default_rng(4)
    x_pts = rng.multivariate_normal(GROUND_MEAN[:3], GROUND_COV[:3, :3], size=300)
    p_kde = kde.evaluate_marginal(x_pts)
    p_true = scipy.stats.multivariate_normal(GROUND_MEAN[:3], GROUND_COV[:3, :3]).pdf(x_pts)
    log_ratio = np.log(p_kde) - np.log(p_true)
    assert abs(np.median(log_ratio)) < 0.5, f"median log ratio {np.median(log_ratio)}"


def test_conditional_density_against_truth():
    kde = get_kde_unit()
    x = GROUND_MEAN[:3] + np.array([0.1, -0.1, 0.05])
    mu_vgx, cov_vgx = _analytic_conditional(x)
    rng = np.random.default_rng(5)
    v_pts = rng.multivariate_normal(mu_vgx, cov_vgx, size=300)
    p_kde = kde.evaluate_conditional(v_pts, x)
    p_true = scipy.stats.multivariate_normal(mu_vgx, cov_vgx).pdf(v_pts)
    log_ratio = np.log(np.clip(p_kde, 1e-30, None)) - np.log(p_true)
    assert abs(np.median(log_ratio)) < 0.5, f"median log ratio {np.median(log_ratio)}"


# Normalization (integrates to 1)

@slow
def test_joint_integrates_to_one():
    kde = get_kde_unit()
    sampler = scipy.stats.multivariate_normal(GROUND_MEAN, GROUND_COV * 4.0)
    integral = _mc_integrate(lambda p: kde(p), sampler, n=10000)
    assert 0.85 < integral < 1.15, f"integral={integral}"


@slow
def test_marginal_integrates_to_one():
    kde = get_kde_unit()
    sampler = scipy.stats.multivariate_normal(GROUND_MEAN[:3], GROUND_COV[:3, :3] * 4.0)
    integral = _mc_integrate(lambda p: kde.evaluate_marginal(p), sampler, n=8000)
    assert 0.85 < integral < 1.15, f"integral={integral}"


@slow
def test_conditional_integrates_to_one_unit_scales():
    """Regression for the spurious /N divisor (Bug 3)."""
    kde = get_kde_unit()
    sampler = scipy.stats.multivariate_normal(GROUND_MEAN[3:], GROUND_COV[3:, 3:] * 4.0)
    rng = np.random.default_rng(6)
    for _ in range(3):
        x = GROUND_MEAN[:3] + rng.standard_normal(3) * 0.5
        integral = _mc_integrate(lambda v: kde.evaluate_conditional(v, x), sampler, n=8000)
        assert 0.85 < integral < 1.15, f"x={x}: integral={integral:.3g}"


@slow
def test_conditional_integrates_to_one_nonunit_scales():
    """Regression for the wrong jacfac in evaluate_conditional (Bug 2)."""
    kde = get_kde_scaled()
    sampler = scipy.stats.multivariate_normal(GROUND_MEAN[3:], GROUND_COV[3:, 3:] * 4.0)
    integral = _mc_integrate(lambda v: kde.evaluate_conditional(v, GROUND_MEAN[:3]), sampler, n=8000)
    assert 0.85 < integral < 1.15, f"integral={integral:.3g}"


@slow
def test_conditional_integrates_to_one_with_covalpha():
    """Bug 1 distorts pofzgivenx when covalpha != 0. The integral is a coarse sanity
    check; the sharper Bug 1 test is `test_regression_marginal_unsummed_sums_to_marginal`."""
    kde = get_kde_unit()
    sampler = scipy.stats.multivariate_normal(GROUND_MEAN[3:], GROUND_COV[3:, 3:] * 4.0)
    for covalpha in (-0.3, 0.0, 0.3):
        integral = _mc_integrate(lambda v: kde.evaluate_conditional(v, GROUND_MEAN[:3], covalpha=covalpha), sampler, n=8000)
        assert 0.85 < integral < 1.15, f"covalpha={covalpha}: integral={integral:.3g}"


# Self-consistency: p(x,v) == p(x) * p(v|x)

@slow
def test_joint_equals_marginal_times_conditional():
    kde = get_kde_unit()
    samples = get_samples()
    rng = np.random.default_rng(7)
    idxs = rng.choice(len(samples), size=15, replace=False)
    for i in idxs:
        xv = samples[i].reshape(1, 6)
        x = xv[0, :3]
        v = xv[0, 3:].reshape(1, 3)
        p_joint = kde(xv)[0]
        p_marg = kde.evaluate_marginal(x)[0]
        p_cond = kde.evaluate_conditional(v, x)[0]
        ratio = p_joint / (p_marg * p_cond)
        assert 0.5 < ratio < 2.0, f"i={i}: ratio={ratio:.3g}"


# Comparison with scipy.stats.gaussian_kde via mockScipyKde
# (same internals, global covariance, Scott's rule)

def test_mock_matches_scipy_gaussian_kde():
    samples = get_samples()
    mock = mockScipyKde(samples.copy(), scalings=np.ones(6), nn=50)
    scipy_kde = scipy.stats.gaussian_kde(samples.T)
    rng = np.random.default_rng(8)
    pts = rng.multivariate_normal(GROUND_MEAN, GROUND_COV, size=30)
    log_ratio = np.log(mock(pts)) - np.log(scipy_kde(pts.T))
    assert abs(np.median(log_ratio)) < 0.1, f"median log ratio {np.median(log_ratio):.3g}"
    assert np.std(log_ratio) < 0.1


# Sample drawing

def test_draw_shape():
    assert get_kde_unit().draw(size=100).shape == (100, 6)


def test_draw_marginal_shape():
    assert get_kde_unit().draw_marginal(size=50).shape == (50, 3)


def test_draw_conditional_shape():
    out = get_kde_unit().draw_conditional_on_x(GROUND_MEAN[:3], size=50)
    assert out.shape == (50, 3)


@slow
def test_draw_recovers_data_moments():
    kde = get_kde_unit()
    samples = get_samples()
    drawn = kde.draw(size=5000)
    np.testing.assert_allclose(drawn.mean(0), samples.mean(0), atol=0.2)
    det_ratio = np.linalg.det(np.cov(drawn.T)) / np.linalg.det(np.cov(samples.T))
    assert 0.7 < det_ratio < 5.0, f"det_ratio={det_ratio:.3g}"


@slow
def test_draw_conditional_mean_matches_analytic():
    """For a Gaussian source the conditional mean is linear in x; samples should track it."""
    kde = get_kde_unit()
    x = GROUND_MEAN[:3] + np.array([0.3, -0.2, 0.1])
    mu_vgx, _ = _analytic_conditional(x)
    drawn = kde.draw_conditional_on_x(x, size=4000)
    np.testing.assert_allclose(drawn.mean(0), mu_vgx, atol=0.25)


# Regression tests for the four issues found in the audit

def test_regression_marginal_unsummed_sums_to_marginal():
    """Bug 1: sum_z unsummed[z] must equal the total marginal, even when covalpha != 0.

    Pre-fix, `unsummed` was missing the `dim*logcovfac` factor that `est` includes.
    With covalpha=0 the missing factor was a constant in z and cancelled in the
    pofzgivenx ratio; with covalpha != 0 it broke that consistency.
    """
    kde = get_kde_unit()
    x = GROUND_MEAN[:3]
    for covalpha in (-0.3, 0.0, 0.3):
        unsummed = kde.evaluate_marginal(x, returnUnsummed=True, covalpha=covalpha)
        marginal = kde.evaluate_marginal(x, covalpha=covalpha)
        np.testing.assert_allclose(unsummed.sum(), marginal, rtol=1e-6, err_msg=f"covalpha={covalpha}: sum(unsummed)={unsummed.sum()} marginal={marginal}")


@slow
def test_regression_conditional_uses_jacfac_cond():
    """Bug 2: `evaluate_conditional` previously divided by the full 6D jacfac.

    With scales = [2,2,2,0.5,0.5,0.5], jacfac=1 but jacfac_cond=0.125. Pre-fix the
    conditional was off by jacfac_marg/N = 8/N. Post-fix it integrates to ~1.
    """
    kde = get_kde_scaled()
    sampler = scipy.stats.multivariate_normal(GROUND_MEAN[3:], GROUND_COV[3:, 3:] * 4.0)
    integral = _mc_integrate(lambda v: kde.evaluate_conditional(v, GROUND_MEAN[:3]), sampler, n=6000)
    assert 0.85 < integral < 1.15, f"integral={integral:.3g}"


@slow
def test_regression_conditional_does_not_divide_by_N():
    """Bug 3: the spurious /N divisor in the conditional. Pre-fix the integral was
    ~1/N; post-fix it's ~1.
    """
    kde = get_kde_unit()
    sampler = scipy.stats.multivariate_normal(GROUND_MEAN[3:], GROUND_COV[3:, 3:] * 4.0)
    integral = _mc_integrate(lambda v: kde.evaluate_conditional(v, GROUND_MEAN[:3]), sampler, n=6000)
    n_data = kde.data.shape[0]
    assert integral > 0.5, \
        f"integral={integral:.3g} suggests /N divisor still present (n_data={n_data})"


def test_regression_local_cov_no_self_duplication():
    """Concern 1: the local covariance must equal the sample cov of the nn nearest
    neighbors (which already includes the data point itself), not a covariance with the
    point doubled. Inspect `covariances_local` - the pre-Silverman cache (since the
    2026-05-19 rescaling convention, `covariances` includes a uniform silverman_sq
    factor that this regression check shouldn't have to track).
    """
    kde = adaptiveKDE(get_samples()[:200].copy(), scalings=np.ones(6), nn=20, use_multiprocessing=False)
    expected = np.cov(kde.data[kde.neighbor_inds[0, :], :].T)
    np.testing.assert_allclose(kde.covariances_local[0], expected, rtol=1e-10)


# Hyperparameter behaviour

def test_covfac_smooths_density():
    """Larger covfac -> broader kernels -> lower density at the mode."""
    kde = get_kde_unit()
    pt = GROUND_MEAN.reshape(1, 6)
    p_narrow = kde(pt, covfac=0.5)[0]
    p_wide = kde(pt, covfac=2.0)[0]
    assert p_narrow > p_wide


def test_covalpha_default_zero():
    kde = get_kde_unit()
    pt = GROUND_MEAN.reshape(1, 6)
    assert np.isclose(kde(pt)[0], kde(pt, covalpha=0.0)[0])


def test_covfac_increases_drawn_spread():
    kde = get_kde_unit()
    s_narrow = kde.draw(size=2000, covfac=0.25)
    s_wide = kde.draw(size=2000, covfac=4.0)
    assert s_wide.std(0).mean() > s_narrow.std(0).mean()


# Boundary-effect characterization (informational; relevant to test_rate_sphere)

@slow
def test_uniform_cube_isotropy_at_origin():
    """
    Sample uniformly in a position cube + isotropic Gaussian velocity, then evaluate
    the KDE on a sphere at the origin. By symmetry the densities should be isotropic;
    edge effects in the local covariance (kernels at the cube faces lean inward) can
    break this. We only assert the densities are finite and report the anisotropy
    so the magnitude can be tracked.
    """
    rng = np.random.default_rng(11)
    n = 3000
    sigma = 0.1
    coords = np.zeros((n, 6))
    coords[:, :3] = rng.uniform(-5, 5, size=(n, 3))
    coords[:, 3:] = rng.standard_normal(size=(n, 3)) * sigma
    kde = adaptiveKDE(coords, scalings=np.ones(6), nn=50, use_multiprocessing=False)
    n_dirs = 80
    rng2 = np.random.default_rng(12)
    costheta = rng2.uniform(-1, 1, n_dirs)
    phi = rng2.uniform(0, 2 * np.pi, n_dirs)
    sintheta = np.sqrt(1 - costheta ** 2)
    r = 0.1
    pts = np.zeros((n_dirs, 6))
    pts[:, 0] = r * sintheta * np.cos(phi)
    pts[:, 1] = r * sintheta * np.sin(phi)
    pts[:, 2] = r * costheta
    densities = kde(pts)
    assert np.all(np.isfinite(densities))
    aniso = float(densities.std() / densities.mean())
    print(f"  [info] uniform-cube anisotropy at origin: {aniso:.3f}")


# Standalone runner (so the file works without pytest installed)

def _is_test(name, fn):
    return name.startswith("test_") and callable(fn)


def main(argv):
    skip_slow = "--no-slow" in argv
    name_filter = None
    if "-k" in argv:
        name_filter = argv[argv.index("-k") + 1]
    tests = [(name, fn) for name, fn in sorted(globals().items()) if _is_test(name, fn)]
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
            dt = time.time() - t1
            print(f"PASS  {name}  ({dt:.2f}s)")
            passed += 1
        except Exception as e:
            dt = time.time() - t1
            print(f"FAIL  {name}  ({dt:.2f}s): {type(e).__name__}: {e}")
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
