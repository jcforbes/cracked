"""KDE-based encounter-rate evaluation.

Public API:
  RATE_Sphere                  - uniform-sphere MC estimator for stream->target encounter rate
  rate_sphere_importance       - same, with Gaussian importance sampling on velocity
  make_data_driven_is_proposal - Gaussian IS proposal from a position-local data subset
  G, pcperau                   - physical constants (pc/Msun/Myr; pc per AU)

All rate functions take a `cvkde` callable that returns a density at 6D phase-space
points (x, y, z, vx, vy, vz) in pc and pc/Myr, plus a Sun-frame location
(`xloc`, `vloc`). The integrand assumes the KDE returns a number density; callers
of probability-density estimators should pre-multiply by their sample-count factor.
"""
import numpy as np
import scipy.stats


# Physical constants (cgs in galactic units: pc, M_sun, Myr).
G_PC3_MSUN_MYR2 = 0.00449987   # pc^3 / (M_sun * Myr^2)
PC_PER_AU = 1.0 / 206265.0

# Short aliases used by callers; kept for ergonomic parity with prior isostreams API.
G = G_PC3_MSUN_MYR2
pcperau = PC_PER_AU


def RATE_Sphere(cvkde, xloc, vloc, qmaxAU=5., Nboot=100000, fac=1., r=0.1, v0=1.0):
    """Uniform-sphere Monte-Carlo encounter-rate estimator.

    Integrates over a sphere of radius `r` (pc) around the target at `xloc`,
    sampling (u, costheta, phi) uniformly. Each sample contributes
        fhat(x_eval, v_eval) * vee * sigma_geom(vee) * 1{unbound}
    where sigma_geom = pi r^2 sin^2theta_c includes gravitational focusing for a 1-M_sun
    target. Returns (weights*Vol, vee, costhetas, phis, eccentricities,
    thetac, v_inf).

    For narrow features (streams, cold components) use `rate_sphere_importance`
    instead - uniform-sphere variance is impractical there.

    The `vee = v0 * tan(u)` substitution maps u  in  [0, pi/2) to vee  in  [0, inf),
    giving the Jacobian `tan^3u / cos^2u` baked into the integrand below.
    """
    NbootInt = int(Nboot)

    us = np.random.random(size=NbootInt) * np.pi / 2.0
    costhetas = np.random.random(size=NbootInt) * 2.0 - 1.0
    phis = np.random.random(size=NbootInt) * 2.0 * np.pi

    thetas = np.arccos(costhetas)
    rhatX = np.sin(thetas) * np.cos(phis)
    rhatY = np.sin(thetas) * np.sin(phis)
    rhatZ = costhetas

    coords = np.zeros((NbootInt, 6))

    vee = v0 * np.tan(us)
    v_esc_sq = 2.0 * G * 1.0 / r
    vinfsq = vee * vee + v_esc_sq
    unbound = vee * vee > v_esc_sq

    sinthetac = np.zeros_like(vee)
    sinthetac[unbound] = (qmaxAU * pcperau / r) * np.sqrt((1.0 + 2.0 * G * 1.0 / (qmaxAU * pcperau * vinfsq[unbound])) / (1.0 - 2.0 * G * 1.0 / (r * vinfsq[unbound])))
    sinthetac = np.clip(sinthetac, 0.0, 1.0)
    thetac = np.arcsin(sinthetac)

    coords[:, 0] = xloc[0] + r * rhatX
    coords[:, 1] = xloc[1] + r * rhatY
    coords[:, 2] = xloc[2] + r * rhatZ
    coords[:, 3] = vloc[0] - vee * rhatX
    coords[:, 4] = vloc[1] - vee * rhatY
    coords[:, 5] = vloc[2] - vee * rhatZ

    f = cvkde(coords)

    integrand = (f * (np.tan(us) ** 3 / np.cos(us) ** 2)
                 * np.sin(thetac) ** 2 * np.pi * v0 ** 4 * r ** 2
                 * fac * 1.0e-6)  # Myr^-1 -> yr^-1

    Vol = 2.0 * np.pi * 2.0 * np.pi / 2.0

    costhetavs = np.random.random(size=NbootInt) * np.cos(thetac)
    eccentricities = np.sqrt(1.0 + (r * np.sin(np.arccos(costhetavs))) ** 2 * (vee * vee / vinfsq) / (G / vinfsq) ** 2)

    return Vol * integrand, vee, costhetas, phis, eccentricities, thetac, np.sqrt(vinfsq)


def make_data_driven_is_proposal(coords, xloc=(0.0, 0.0, 0.0), K=None, inflation=4.0, sigma_floor_factor=0.05, vloc=None):
    """Derive an IS proposal (Gaussian on galactic-frame velocity) from a
    position-local data subset around the Sun's position `xloc`.

    For narrow scenarios the rate integrand peaks at v ~ v_data_local_bulk
    (in galactic frame); sampling from a Gaussian centred there is much more
    efficient than uniform sphere. For broad scenarios this still works - the
    proposal is wider, the rate-eval samples cover more of the sphere, and IS
    reduces to roughly uniform behaviour.

    `K` is the number of nearest spatial points used to estimate the local
    velocity distribution. Default (`K=None`) selects `K = max(50, floor(sqrtN))` so
    the per-trial subset grows with sample size for asymptotic consistency
    while staying narrow enough at large N that the proposal tracks the local
    structure rather than averaging over the whole dataset (critical for
    stream-like scenarios where the global v-cov is v_circ-dominated).

    The diagonal floor `sigma_floor_factor x local_v_std` (computed from the
    same K-NN subset, NOT the global v-std) prevents singular cov when the
    local subset has near-zero spread on some axis. The local reference is
    essential for stream-like data: the global v-std is dominated by v_circ
    across the whole ring (~155 km/s for an 8 kpc disk) and would bloat the
    floor by >1000x over the actual local sigma_t~0.1 km/s.

    `vloc`: if provided, return a 2-component MIXTURE proposal instead of a
    single Gaussian. The first component is the data-driven Gaussian above
    (centred where fhat has its mass); the second is centred at `vloc` (the
    Sun's velocity) with the same shape covariance, covering the sigma_geom
    peak at v_rel = 0. This fixes the N_eff=1 failure mode where the
    integrand peak sits between the data bulk and the Sun's velocity and
    a single-Gaussian proposal at the data bulk misses it.

    Returns
    -------
    v_mean, v_cov
        - With `vloc=None` (legacy): shapes (3,), (3, 3).  Pass straight into
          `rate_sphere_importance(..., v_proposal_mean=v_mean, v_proposal_cov=v_cov)`.
        - With `vloc` set: shapes (2, 3), (2, 3, 3). `rate_sphere_importance`
          detects the 2D mean and uses a balanced mixture-IS draw.
    """
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 6:
        raise ValueError(
            f"make_data_driven_is_proposal expects coords of shape (N, 6) "
            f"with columns [x, y, z, vx, vy, vz]; got shape {coords.shape}.")
    N = coords.shape[0]
    if N < 4:
        raise ValueError(
            f"make_data_driven_is_proposal needs at least 4 particles to "
            f"estimate a 3x3 velocity covariance; got {N}.")
    if K is None:
        K = max(50, int(np.floor(np.sqrt(N))))
    xloc = np.asarray(xloc, dtype=float)
    rsq = ((coords[:, 0] - xloc[0]) ** 2
           + (coords[:, 1] - xloc[1]) ** 2
           + (coords[:, 2] - xloc[2]) ** 2)
    sortr = np.argsort(rsq)
    K_eff = min(K, N - 1)
    subset_v = coords[sortr[:K_eff], 3:]
    v_mean = subset_v.mean(axis=0)
    v_cov = np.cov(subset_v.T) * inflation
    local_v_std = subset_v.std(axis=0, ddof=1) if K_eff > 1 else np.ones(3)
    v_cov = v_cov + np.diag((sigma_floor_factor * local_v_std) ** 2)
    if vloc is None:
        return v_mean, v_cov
    # Mixture proposal: data-driven component + Sun-centered component.
    vloc_arr = np.asarray(vloc, dtype=float).reshape(3)
    v_means = np.stack([v_mean, vloc_arr], axis=0)            # (2, 3)
    v_covs  = np.stack([v_cov, v_cov], axis=0)                # (2, 3, 3)
    return v_means, v_covs


def rate_sphere_importance(cvkde, v_proposal_mean, v_proposal_cov, xloc=(0.0, 0.0, 0.0), vloc=(0.0, 0.0, 0.0), Nboot=30000, fac=1.0, qmaxAU=5.0, r=0.1, M_target=1.0, rng=None):
    """RATE_Sphere with importance sampling on velocity, for narrow features
    where uniform sphere sampling has prohibitive variance.

    Mirrors RATE_Sphere's return tuple (weights, vee, costhetas, phis,
    eccentricities, thetac, vinf).

    `xloc, vloc` give the Sun's position and velocity in the same galactic
    frame as the KDE's training data. `v_proposal_mean, v_proposal_cov` are
    in galactic-frame v as well - `make_data_driven_is_proposal` returns
    these directly from a position-local subset of the data.

    Sampling strategy: draw v_galactic ~ N(v_proposal_mean, v_proposal_cov),
    convert to Sun-frame v_rel = v_galactic - vloc, then derive (vee, costheta,
    phi) from rhat = -v_rel/|v_rel|. Each sample contributes
        fhat(x_eval, v_eval) * vee * sigma_geom(vee) * 1{unbound} * vee^2 / q(v) * fac * 1e-6
    where the vee^2 is the spherical-volume Jacobian and 1/q(v) is the
    importance weight. The mean of these contributions over N samples is the
    rate estimate (no Vol prefactor - the Jacobian and proposal absorb the
    integration measure correctly).

    `M_target` is the target body's mass in M_sun (default 1, i.e. Sun-like).
    """
    n = int(Nboot)
    if rng is None:
        rng = np.random.default_rng()
    xloc = np.asarray(xloc, dtype=float)
    vloc = np.asarray(vloc, dtype=float)
    GM = G_PC3_MSUN_MYR2 * float(M_target)

    # Validate the proposal. Two accepted shapes:
    #   (3,), (3, 3)         -> single Gaussian (legacy)
    #   (K, 3), (K, 3, 3)    -> K-component mixture (balanced weights 1/K)
    v_mean_arr = np.asarray(v_proposal_mean, dtype=float)
    v_cov_arr  = np.asarray(v_proposal_cov,  dtype=float)
    if v_mean_arr.ndim == 1:
        if v_mean_arr.size != 3 or v_cov_arr.shape != (3, 3):
            raise ValueError(
                f"rate_sphere_importance: single-Gaussian proposal must be a "
                f"3-vector mean and a 3x3 covariance; got "
                f"mean.shape={v_mean_arr.shape}, cov.shape={v_cov_arr.shape}. "
                f"This usually means the upstream `v_mean`/`v_cov` were built "
                f"from an empty position-local data subset (no particles "
                f"near xloc).")
        # Promote to a single-component mixture so the draw / weight code
        # downstream is uniform. For K=1, mixture pdf == component pdf, so
        # the rate estimate is bit-for-bit identical to the legacy path.
        v_mean_arr = v_mean_arr.reshape(1, 3)
        v_cov_arr  = v_cov_arr.reshape(1, 3, 3)
    if not (v_mean_arr.ndim == 2 and v_mean_arr.shape[1] == 3
            and v_cov_arr.ndim == 3 and v_cov_arr.shape[1:] == (3, 3)
            and v_cov_arr.shape[0] == v_mean_arr.shape[0]):
        raise ValueError(
            f"rate_sphere_importance: mixture proposal must have shapes "
            f"(K, 3), (K, 3, 3); got mean.shape={v_mean_arr.shape}, "
            f"cov.shape={v_cov_arr.shape}.")
    K = v_mean_arr.shape[0]
    # Balanced mixture draw: assign components round-robin (deterministic
    # over `n`) so per-component sample counts are exactly n/K each. This
    # gives lower variance than random allocation and avoids degenerate
    # zero-allocation in any component.
    component = np.arange(n) % K
    rng.shuffle(component)
    v_samples = np.empty((n, 3))
    for k in range(K):
        mask = (component == k)
        nk = int(mask.sum())
        if nk == 0:
            continue
        v_samples[mask] = rng.multivariate_normal(v_mean_arr[k], v_cov_arr[k], size=nk)
    # Velocity relative to the Sun.
    v_rel = v_samples - vloc[None, :]
    vee = np.linalg.norm(v_rel, axis=1)
    safe_vee = np.maximum(vee, 1.0e-30)
    rhat = -v_rel / safe_vee[:, None]
    costhetas = rhat[:, 2]
    phis = np.arctan2(rhat[:, 1], rhat[:, 0])

    coords = np.zeros((n, 6))
    coords[:, 0] = xloc[0] + r * rhat[:, 0]
    coords[:, 1] = xloc[1] + r * rhat[:, 1]
    coords[:, 2] = xloc[2] + r * rhat[:, 2]
    coords[:, 3] = v_samples[:, 0]
    coords[:, 4] = v_samples[:, 1]
    coords[:, 5] = v_samples[:, 2]

    qmax_pc_local = qmaxAU * PC_PER_AU
    v_esc_sq = 2.0 * GM / r
    v_inf_sq = vee * vee + v_esc_sq
    unbound = vee * vee > v_esc_sq
    sin2_thetac = np.zeros_like(vee)
    sin2_thetac[unbound] = (qmax_pc_local / r) ** 2 * (
        (1.0 + 2.0 * GM / (qmax_pc_local * v_inf_sq[unbound]))
        / (1.0 - 2.0 * GM / (r * v_inf_sq[unbound]))
    )
    sin2_thetac = np.clip(sin2_thetac, 0.0, 1.0)
    thetac = np.arcsin(np.sqrt(sin2_thetac))
    sigma_geom = np.pi * r * r * sin2_thetac

    f = np.atleast_1d(cvkde(coords))

    # Mixture-IS density: q(v) = (1/K) Sigma_k N(v | mu_k, Sigma_k). This is the
    # density of ANY sample under the balanced mixture, regardless of which
    # component it was actually drawn from - the "balance heuristic" for
    # multi-proposal importance sampling. For K=1 this collapses to the
    # single-Gaussian density (legacy behavior, identical numerically).
    from scipy.special import logsumexp
    log_q_components = np.stack([
        scipy.stats.multivariate_normal(
            mean=v_mean_arr[k], cov=v_cov_arr[k]).logpdf(v_samples)
        for k in range(K)
    ], axis=0)   # shape (K, n)
    log_q = logsumexp(log_q_components, axis=0) - np.log(K)
    q = np.exp(log_q)
    weights = np.where(unbound, f * vee * sigma_geom * fac * 1.0e-6 / np.maximum(q, 1.0e-300), 0.0)

    costhetavs = rng.uniform(0.0, np.cos(thetac))
    safe_vinf = np.sqrt(v_inf_sq)
    eccentricities = np.sqrt(1.0 + (r * np.sin(np.arccos(costhetavs))) ** 2 * (vee * vee / v_inf_sq) / (GM / v_inf_sq) ** 2)
    return weights, vee, costhetas, phis, eccentricities, thetac, safe_vinf


def make_production_cv_kde(coords, *, xloc, vloc, encounter_radius_pc, qmaxAU=5.0, M_target=1.0, ncovfacs=11, covfac_range=(-3.0, 0.5), n_neff_pts=300, is_proposal_K=None, is_proposal_inflation=4.0, is_proposal_sigma_floor_factor=0.05, neff_seed=0, sky_ess_floor=10.0, vinf_ess_floor=5.0, rr_method='ISE', roi6=None, random_state=None, suppress_stdout=False, weights=None):
    """Build the production-canonical cvAdaptiveKDE for an encounter-rate problem.

    Single source of truth. Both production and the convergence test should call
    this rather than reconstructing the kwargs - prior drift between the two
    (e.g. production silently dropping `roi6=200`, ~100x CV slowdown) is the
    failure mode this function exists to prevent.

    Fixed-by-design hyperparameters (do not parameterise without good reason):
      - nfolds=5
      - covalpha_range=(-0.2, 0.6), ncovalphas=5  -> [-0.2, 0, 0.2, 0.4, 0.6]
      - shrinkage_grid=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
      - scalings_grid=['auto', 'narrow', 'narrow_local']
      - shrinkage_target='local_pooled'
      - rr_method='ISE'   (alias for the historical 'bootstrap'; the
                           objective IS an ISE approximation, the bootstrap
                           is just the RR-term estimator. Pass 'LOO' for
                           leave-one-out log-likelihood instead.)
      - roi6=None                              (no CV throttle by default;
                                                 134-file paired study on
                                                 production data showed roi6=200
                                                 gave median 1.6x *worse* sky
                                                 ESS than roi6=None - the
                                                 200-NN cluster constrains CV
                                                 away from kernels that fit the
                                                 sky-locus. Pass roi6=200 to
                                                 restore the old behavior.)
      - neff_floor=30
      - stability_lambda=2.0

    Parameters
    ----------
    coords : (N, 6) array
        Training data in galactic-frame (x, y, z, vx, vy, vz) - pc / pc*Myr^-1.
    xloc, vloc : (3,) arrays
        Sun's position and velocity in the same galactic frame; together they
        define the 6D ROI center used by the CV throttle and the centre of the
        encounter-sphere N_eff probe.
    encounter_radius_pc : float (required)
        Radius of the encounter sphere on which the CV N_eff floor is checked.
        MUST match the `r` passed to `rate_sphere_importance` at evaluation
        time - otherwise the floor is enforced at one sphere radius and the
        actual rate / sky-map evaluation happens at a different one, which
        can let the floor mechanism silently fall back to the unconstrained
        rate pick (observed 2026-05-25 production failure: cv_sky N_eff = 2
        on "corr" scenario because floor was at r=0.1 but eval was at r=1.0).
    ncovfacs : int
        Number of covfac points. 11 is the production default; 13 is the
        older-stream widening; 15 is the convergence-test setting.
    covfac_range : (float, float)
        log10 bounds of the covfac grid.

    Returns
    -------
    cvAdaptiveKDE
        Built and CV'd. Caller picks the per-task estimator via
        `.kde_rate`, `.kde_shape`, `.pick_for_dim(...)` as usual.
    """
    from .kde import cvAdaptiveKDE   # local import avoids circular module load
    import contextlib, io, sys

    coords = np.asarray(coords)
    xloc = np.asarray(xloc, dtype=float)
    vloc = np.asarray(vloc, dtype=float)

    # IS proposal: 2-component mixture (data-bulk + Sun-centered). Covers
    # both the fhat peak (around the local-data bulk velocity) and the
    # sigma_geom peak (at v_rel = 0, i.e. v_galactic = vloc). Single-Gaussian
    # proposals failed with N_eff=1 in production whenever the bulk velocity
    # differed from vloc by more than ~3*sigma_data_local. Returned shapes are
    # (2, 3) and (2, 3, 3); rate_sphere_importance auto-detects.
    v_mean, v_cov = make_data_driven_is_proposal(coords, xloc=xloc, K=is_proposal_K, inflation=is_proposal_inflation, sigma_floor_factor=is_proposal_sigma_floor_factor, vloc=vloc)

    # N_eff evaluation points: phase-space points on a sphere of radius
    # `encounter_radius_pc` around `xloc`, with velocities drawn from the IS
    # proposal so the floor measures N_eff where rate_sphere_importance will
    # later evaluate the density. Drawn from the mixture: round-robin
    # component assignment + per-component MV-normal sampling.
    rng_pts = np.random.default_rng(neff_seed)
    n_hat = rng_pts.normal(size=(n_neff_pts, 3))
    n_hat /= np.linalg.norm(n_hat, axis=1, keepdims=True)
    K_mix = v_mean.shape[0] if v_mean.ndim == 2 else 1
    if K_mix == 1:
        v_samples = rng_pts.multivariate_normal(v_mean.reshape(3), v_cov.reshape(3, 3), size=n_neff_pts)
    else:
        component = np.arange(n_neff_pts) % K_mix
        rng_pts.shuffle(component)
        v_samples = np.empty((n_neff_pts, 3))
        for k in range(K_mix):
            mask = (component == k)
            nk = int(mask.sum())
            if nk == 0:
                continue
            v_samples[mask] = rng_pts.multivariate_normal(v_mean[k], v_cov[k], size=nk)
    neff_pts = np.zeros((n_neff_pts, 6))
    neff_pts[:, 0] = xloc[0] + encounter_radius_pc * n_hat[:, 0]
    neff_pts[:, 1] = xloc[1] + encounter_radius_pc * n_hat[:, 1]
    neff_pts[:, 2] = xloc[2] + encounter_radius_pc * n_hat[:, 2]
    neff_pts[:, 3:] = v_samples

    # 6D ROI center: Sun's position + Sun's velocity, in the data's galactic frame.
    roi6_center = np.concatenate([xloc, vloc]).tolist()

    # Geometry inputs for the rate-weighted sky / v_inf ESS metrics computed
    # inside CV. These are derived from `neff_pts` plus the encounter geometry
    # - exactly the same per-sample weights `rate_sphere_importance` would
    # construct, but precomputed once so the CV inner loop just multiplies by
    # fhat and bins.
    v_rel_neff = neff_pts[:, 3:] - vloc[None, :]
    vee_neff = np.linalg.norm(v_rel_neff, axis=1)
    safe_vee = np.maximum(vee_neff, 1.0e-30)
    rhat_neff = -v_rel_neff / safe_vee[:, None]
    sky_bin_costheta = rhat_neff[:, 2]
    sky_bin_phi = np.arctan2(rhat_neff[:, 1], rhat_neff[:, 0])
    GM = G_PC3_MSUN_MYR2 * float(M_target)
    r_pc = float(encounter_radius_pc)
    qmax_pc = qmaxAU * PC_PER_AU
    v_esc_sq = 2.0 * GM / r_pc
    v_inf_sq = vee_neff * vee_neff + v_esc_sq
    unbound = vee_neff * vee_neff > v_esc_sq
    sin2_thetac = np.zeros_like(vee_neff)
    sin2_thetac[unbound] = (qmax_pc / r_pc) ** 2 * (
        (1.0 + 2.0 * GM / (qmax_pc * v_inf_sq[unbound]))
        / (1.0 - 2.0 * GM / (r_pc * v_inf_sq[unbound]))
    )
    sin2_thetac = np.clip(sin2_thetac, 0.0, 1.0)
    sigma_geom = np.pi * r_pc * r_pc * sin2_thetac
    rate_weight_geom_factor = np.where(unbound,
                                        vee_neff * sigma_geom,
                                        0.0)
    # v_inf bin coord: log10 of v_inf (~spans 1-3 decades for typical eval pts)
    vinf_bin_coord = np.log10(np.sqrt(v_inf_sq) + 1.0e-30)

    @contextlib.contextmanager
    def _quiet():
        if not suppress_stdout:
            yield
            return
        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

    with _quiet():
        cv = cvAdaptiveKDE(
            coords,
            nfolds=5,
            ncovfacs=ncovfacs,
            covfac_range=covfac_range,
            covalpha_range=(-0.2, 0.6),
            ncovalphas=5,
            shrinkage_grid=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
            scalings_grid=['auto', 'narrow', 'narrow_local'],
            shrinkage_target='local_pooled',
            roi=None,
            rr_method=rr_method,
            neff_eval_points=neff_pts,
            neff_floor=30.0,
            roi6=roi6,
            roiCenter6=roi6_center if roi6 is not None else None,
            stability_lambda=2.0,
            random_state=random_state,
            weights=weights,
            # Rate-weighted ESS metrics for the sky / v_inf picks.
            sky_bin_costheta=sky_bin_costheta,
            sky_bin_phi=sky_bin_phi,
            vinf_bin_coord=vinf_bin_coord,
            rate_weight_geom_factor=rate_weight_geom_factor,
        )

    # Attach the IS proposal the factory built. Downstream rate_sphere_importance
    # should use the SAME proposal as the N_eff floor; passing these back lets
    # callers do `rate_sphere_importance(cv.kde_rate,
    # v_proposal_mean=cv.v_proposal_mean, v_proposal_cov=cv.v_proposal_cov, ...)`
    # without rebuilding (and possibly inconsistent) proposals.
    cv.v_proposal_mean = v_mean
    cv.v_proposal_cov = v_cov
    cv.encounter_radius_pc = float(encounter_radius_pc)
    # Sky / v_inf picks at the user-specified ESS thresholds. Cached on the
    # cv object so callers can reach them as attributes (cv.kde_sky, cv.kde_vinf)
    # parallel to cv.kde_rate / cv.kde_shape.
    cv.kde_sky = cv.pick_at_sky_floor(sky_ess_floor)
    cv.kde_vinf = cv.pick_at_vinf_floor(vinf_ess_floor)
    return cv


# Post-training reliability diagnostics
#
# Two per-estimator metrics that production code (e.g. isostreams_prod)
# stores per isostream to flag which rate estimates to trust:
#   data_neff_<label>      - Kish ESS of training-particle contributions
#                              to the rate sum (NaN for NF since it is
#                              not a kernel method).
#   nf_rate_ensemble_*     - std/mean of the rate computed by each of
#                              the NF ensemble's flows independently
#                              (NF self-confidence in the rate).
# Both are computed AFTER rate_sphere_importance has been run for every
# estimator; their inputs are the per-estimator sample outputs.

def eval6_from_sphere_samples(sphere_samples, xsun, vsun, encounter_r):
    """Reconstruct the 6D phase-space evaluation points from
    `rate_sphere_importance` outputs (cos theta, phi, |v_rel|).

    Inverse of `rate_sphere_importance`'s eval-point construction.
    Returns an (N, 6) array of (x_sun+r*nhat,  v_rel + v_sun) in the data
    frame, where nhat = (sinthetacosphi, sinthetasinphi, costheta) and v_rel = -|v_rel|*nhat.

    `sphere_samples` is the tuple returned by rate_sphere_importance:
    (resj, vsphere, costhetasphere, phisphere, ecc, thetac, vinfty).
    Only vsphere, costhetasphere, phisphere are used.
    """
    _, vsphereL, costhetaL, phiL, *_ = sphere_samples
    xsun = np.asarray(xsun, dtype=float)
    vsun = np.asarray(vsun, dtype=float)
    sinth = np.sqrt(np.maximum(1.0 - costhetaL ** 2, 0.0))
    nx = sinth * np.cos(phiL)
    ny = sinth * np.sin(phiL)
    nz = costhetaL
    out = np.empty((len(vsphereL), 6))
    out[:, 0] = xsun[0] + encounter_r * nx
    out[:, 1] = xsun[1] + encounter_r * ny
    out[:, 2] = xsun[2] + encounter_r * nz
    # v_rel = -|v_rel|*nhat  (rhat from importance sampler)
    out[:, 3] = -vsphereL * nx + vsun[0]
    out[:, 4] = -vsphereL * ny + vsun[1]
    out[:, 5] = -vsphereL * nz + vsun[2]
    return out


def reliability_report(estimators, samples_by_estimator, *, xsun, vsun, encounter_r, nf_label='nf', verbose=False):
    """Per-estimator post-training reliability metrics for the rate.

    Parameters
    ----------
    estimators : dict[str, tuple]
        Mapping label -> (kde_obj, ...).  Only kde_obj is used.  Match
        the structure isostreams_prod uses (`estimators[label][0]`).
    samples_by_estimator : dict[str, tuple]
        Mapping label -> output of rate_sphere_importance (the 7-tuple
        (resj, vsphere, costhetasphere, phisphere, ecc, thetac, vinfty)).
    xsun, vsun : (3,) array-like
        Sun position / velocity in the data frame.  Required to
        reconstruct the 6D eval points.
    encounter_r : float
        Encounter sphere radius (same `r` passed to rate_sphere_importance).
    nf_label : str
        Label under which the NF estimator is stored in `estimators`.
        Default 'nf'.  Set to None to skip the NF ensemble metric.

    Returns
    -------
    dict ready to merge into the production save payload.  Keys:
        data_neff_<label>             - float (NaN if the estimator has
                                         no data_side_neff method).
        nf_rate_per_flow              - (n_ens,) array (only present if
                                         NF is in `estimators` and has
                                         per_flow_density).
        nf_rate_ensemble_mean / _std / _cv  - floats.
    """
    out = {}
    for label, samples in samples_by_estimator.items():
        kde_obj = estimators[label][0]
        resjL = samples[0]
        if hasattr(kde_obj, 'data_side_neff'):
            try:
                eval6 = eval6_from_sphere_samples(samples, xsun, vsun, encounter_r)
                dsn = float(kde_obj.data_side_neff(eval6, eval_weights=resjL))
            except Exception as e:
                if verbose:
                    print(f"  data_side_neff failed for {label}: "
                          f"{type(e).__name__}: {e}")
                dsn = float('nan')
        else:
            dsn = float('nan')
        out[f'data_neff_{label}'] = dsn

    if nf_label is not None and nf_label in estimators:
        nf_kde = estimators[nf_label][0]
        if hasattr(nf_kde, 'per_flow_density') \
                and nf_label in samples_by_estimator:
            try:
                from scipy.special import logsumexp
                nf_samples = samples_by_estimator[nf_label]
                nf_resjL = nf_samples[0]
                eval6_nf = eval6_from_sphere_samples(nf_samples, xsun, vsun, encounter_r)
                per_flow = nf_kde.per_flow_density(eval6_nf)
                log_per_flow = np.log(np.maximum(per_flow, 1.0e-300))
                log_f_ens = (logsumexp(log_per_flow, axis=0)
                             - np.log(per_flow.shape[0]))
                f_ens = np.exp(log_f_ens)
                safe = f_ens > 0
                per_flow_rates = np.array([float(np.mean(nf_resjL[safe] * per_flow[k, safe] / f_ens[safe])) for k in range(per_flow.shape[0])])
                mean = float(np.mean(per_flow_rates))
                std = float(np.std(per_flow_rates))
                out['nf_rate_per_flow']      = per_flow_rates
                out['nf_rate_ensemble_mean'] = mean
                out['nf_rate_ensemble_std']  = std
                out['nf_rate_ensemble_cv']   = (std / mean
                                                if mean > 0 else float('nan'))
            except Exception as e:
                if verbose:
                    print(f"  NF ensemble-variance diagnostic failed: "
                          f"{type(e).__name__}: {e}")
    return out
