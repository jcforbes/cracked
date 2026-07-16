"""Recommend a KDE method given a dataset and a task.

Companion to scripts/method_regime_map.py - encodes the same regime
logic programmatically. Diagnoses the data along three axes (sample
size, multi-scale velocity dispersion, narrow-feature presence) and
picks scipy_kde / cvGaussianKDE / cvAdaptiveKDE / NormalizingFlowKDE
accordingly.

Usage:
    from cracked.recommend import recommend_method
    rec = recommend_method(coords, task='rate')
    print(rec['rationale'])
    # rec['method']      -> e.g. 'cvAdaptiveKDE'
    # rec['method_args'] -> kwargs you can splat into the constructor

Or from the command line (loads a numpy/dill pickle of (N, 6) coords):
    ipython3 -m cracked.recommend mydata.npy --task rate
"""
from __future__ import annotations

from typing import Any
import argparse
import numpy as np

from .kde import _detect_narrow_v_sigma


# Diagnostics

def analyze_data(coords: np.ndarray) -> dict:
    """Return a dict of summary properties used to pick a method.

    `coords` is an (N, d) array; d=6 expected (3 positions + 3 velocities)
    for the rate / sky-map applications cracked targets, but a 3D pure-
    velocity dataset also works for density-only tasks.
    """
    coords = np.asarray(coords)
    if coords.ndim != 2:
        raise ValueError(f"expected 2D coords array, got shape {coords.shape}")
    N, d = coords.shape

    # Per-axis statistics
    sigma_global = np.std(coords, axis=0)

    diag: dict[str, Any] = {
        'N': int(N),
        'd': int(d),
        'sigma_global': sigma_global.tolist(),
    }

    # Multi-scale velocity dispersion: per-axis narrow-component GMM
    # detection (only meaningful on the velocity axes for the typical
    # 6D phase-space layout).
    if d >= 6:
        narrow_sigma = _detect_narrow_v_sigma(coords)
        v_sigma_global = sigma_global[3:6]
        ratio = np.array(narrow_sigma) / np.maximum(v_sigma_global, 1e-30)
        diag['v_sigma_narrow'] = list(map(float, narrow_sigma))
        diag['v_sigma_global'] = list(map(float, v_sigma_global))
        diag['narrow_ratio'] = list(map(float, ratio))
        diag['multi_scale_flag'] = bool(np.any(ratio < 0.3))
    else:
        diag['multi_scale_flag'] = False

    # Spatial extent / velocity extent ratio: a stream-like dataset
    # has positions spanning much wider than velocities.
    if d >= 6:
        pos_extent = float(np.median(sigma_global[:3]))
        vel_extent = float(np.median(sigma_global[3:6]))
        diag['pos_to_vel_ratio'] = (pos_extent / max(vel_extent, 1e-30))
        diag['stream_like_flag'] = diag['pos_to_vel_ratio'] > 100.0

    # 6D anisotropy: condition number of the sample covariance.
    # Thin 1D features in 6D produce global cov condition number >> 1
    # when ALL features point the same way (single stream). For
    # many-direction features (spiky ball) the global cov is isotropic
    # and this check misses - see the local-covariance test below.
    try:
        cov = np.cov(coords, rowvar=False)
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        eigvals = np.clip(eigvals, 1e-30, None)
        diag['cov_eigvals'] = eigvals.tolist()
        diag['cov_cond_number'] = float(eigvals[0] / eigvals[-1])
        global_cov_narrow = diag['cov_cond_number'] > 1.0e4
    except Exception:
        global_cov_narrow = False
        cov = None

    # Local-covariance test: for each of M random sample points, compute
    # the covariance of its K nearest neighbors. If the local cov is
    # systematically much narrower than the global cov, the data has
    # local 1D-ish features (streams, spikes) even if the global cov
    # is isotropic. Detect by the median ratio
    #     trace(local_cov_i) / trace(global_cov)
    # over the M sample points. Median << 1 implies local concentration.
    local_narrowness_score = 1.0
    local_cond_score = 1.0
    try:
        from scipy.spatial import cKDTree
        # Scale by per-axis std so kdtree distance treats axes evenly.
        scales = np.where(sigma_global > 0, sigma_global, 1.0)
        scaled = coords / scales
        tree = cKDTree(scaled)
        M = min(200, N)
        rng = np.random.default_rng(13)
        sample_idx = rng.choice(N, size=M, replace=False)
        K = min(20, N - 1)
        _, neighbor_idx = tree.query(scaled[sample_idx], k=K)
        global_trace = float(np.trace(cov))
        local_traces = []
        local_conds = []
        for i in range(M):
            pts = coords[neighbor_idx[i]]
            c = np.cov(pts, rowvar=False)
            local_traces.append(float(np.trace(c)))
            try:
                ev = np.linalg.eigvalsh(c)
                ev = np.clip(ev, 1e-30, None)
                local_conds.append(float(ev[-1] / ev[0]))
            except Exception:
                local_conds.append(1.0)
        local_narrowness_score = float(np.median(local_traces) / max(global_trace, 1e-30))
        local_cond_score = float(np.median(local_conds))
        diag['local_trace_ratio'] = local_narrowness_score
        diag['local_cond_med'] = local_cond_score
    except Exception:
        diag['local_trace_ratio'] = 1.0
        diag['local_cond_med'] = 1.0

    # Local "feature" = either local trace is much smaller than global
    # (1D feature contained in a 6D box) OR the local cov has a big
    # condition number (1D direction within the K-NN cluster).
    local_feature = (local_narrowness_score < 0.05) or (local_cond_score > 1.0e3)
    diag['narrow_feature_flag'] = bool(global_cov_narrow or local_feature)

    return diag


# Local-to-xloc diagnostics
#
# Three measurements at a target encounter point xloc that, together
# with the existing global flags from analyze_data, classify the
# (data, task) regime per the empirically-validated triplet in
# scripts/check_triplet_heuristic.py.

def analyze_local(coords: np.ndarray, xloc, R_proxy: float = 1.0, K: int = 50) -> dict:
    """Diagnostics evaluated at a target point `xloc` in 6D phase space.

    Parameters
    ----------
    coords : (N, 6) array
        Phase-space data: 3 positions + 3 velocities.
    xloc : (3,) array-like
        Target position in the data frame (the encounter sphere center
        for rate / sky-map tasks).
    R_proxy : float
        Radius (in position units) defining "near the target" for the
        n_near count.  Default 1 pc matches cracked's 0.1 pc encounter
        sphere x10.
    K : int
        Number of position-nearest neighbours used to estimate the
        local velocity dispersion.  Default 50.

    Returns
    -------
    dict with keys:
      n_near                : # particles within R_proxy of xloc (3D pos)
      coherence_min         : min over v-axes of sigma_v_local / sigma_v_global,
                              where sigma_v_local is computed over the K
                              position-nearest neighbours of xloc.
                              Small ratio => coherent local flow at xloc
                              (stream-like, where cvAdaptive earns it).
      multi_scale_min       : min over v-axes of sigma_narrow / sigma_v_local on
                              the K-NN sample.  Small ratio => multi-scale
                              velocity dispersion locally (cold+hot
                              family - favours cvAdapt-narrow / cvGauss-
                              narrow).
      coherence_per_axis    : 3-vector breakdown of coherence_min
      narrow_sigma_local    : 3-vector of GMM-detected narrow sigma's
      sigma_v_local         : 3-vector of sigma_v in the K-NN of xloc
      sigma_v_global        : 3-vector of sigma_v across all data
    """
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] < 6:
        raise ValueError("analyze_local requires (N, 6) phase-space coords")
    xloc = np.asarray(xloc, dtype=float).reshape(3)
    pos = coords[:, :3]
    vel = coords[:, 3:6]

    r = np.linalg.norm(pos - xloc[None, :], axis=1)
    n_near = int((r < R_proxy).sum())

    K_use = int(min(K, len(coords)))
    if K_use < 5:
        # Pathological: not enough data to compute local diagnostics
        return dict(n_near=n_near, coherence_min=1.0, multi_scale_min=1.0, coherence_per_axis=[1.0, 1.0, 1.0], narrow_sigma_local=[1.0, 1.0, 1.0], sigma_v_local=[1.0, 1.0, 1.0], sigma_v_global=[1.0, 1.0, 1.0])
    nn_idx = np.argpartition(r, K_use - 1)[:K_use]
    local_v = vel[nn_idx]

    sigma_v_local = np.std(local_v, axis=0)
    sigma_v_global = np.std(vel, axis=0)
    coh_per_axis = sigma_v_local / np.maximum(sigma_v_global, 1e-30)
    coherence_min = float(np.min(coh_per_axis))

    # GMM narrow-component on the LOCAL velocity sample.
    local6 = np.hstack([coords[nn_idx, :3], local_v])
    narrow_sigma = np.array(_detect_narrow_v_sigma(local6))
    ms_per_axis = narrow_sigma / np.maximum(sigma_v_local, 1e-30)
    multi_scale_min = float(np.min(ms_per_axis))

    return dict(n_near=n_near, coherence_min=coherence_min, coherence_per_axis=coh_per_axis.tolist(), multi_scale_min=multi_scale_min, narrow_sigma_local=narrow_sigma.tolist(), sigma_v_local=sigma_v_local.tolist(), sigma_v_global=sigma_v_global.tolist())


# Structural classifier: (local + global flags) -> regime class.

def classify_structure(local: dict | None, gdiag: dict, coh_extreme: float = 0.01, ms_threshold: float = 0.1, cond_threshold: float = 1.0e4) -> str:
    """Map diagnostics to a structural class.

    Precedence (validated 5/5 on the regime map):
      1. local coherence_min < 0.01   => 'narrow_coherent'  (stream-like)
      2. local multi_scale_min < 0.1  => 'multi_scale'      (cold+hot-like)
      3. global narrow_feature_flag AND local_cond_med > 1e4
                                       => 'many_narrow'    (spiky-ball-like)
      4. else                          => 'smooth'

    If `local` is None, only the global flags are used (the n_near /
    coh / ms checks are skipped) and the classifier may under-fire
    the 'narrow_coherent' / 'multi_scale' classes.
    """
    if local is not None:
        if local['coherence_min'] < coh_extreme:
            return 'narrow_coherent'
        if local['multi_scale_min'] < ms_threshold:
            return 'multi_scale'
    if gdiag.get('narrow_feature_flag', False) \
            and gdiag.get('local_cond_med', 0.0) > cond_threshold:
        return 'many_narrow'
    # When local diagnostics are unavailable, fall back to the older
    # global multi_scale_flag as a backstop for the cold+hot family.
    if local is None and gdiag.get('multi_scale_flag', False):
        return 'multi_scale'
    return 'smooth'


# Task-aware method table - encodes the regime map directly.
#
# Each entry is (method_name, default_kwargs, rationale_kernel).
# The method names match the cracked package's public classes /
# factory functions.
_METHOD_TABLE: dict[tuple[str, str], tuple[str, dict, str]] = {
    # smooth / anisotropic data
    ('smooth', 'rate'):
        ('NormalizingFlowKDE', dict(ensemble_size=10),
         "smooth data with no narrow features or local multi-scale; "
         "NF edges out scipy by a few percent on rate and is the "
         "no-regret choice across tasks"),
    ('smooth', 'sky_map'):
        ('NormalizingFlowKDE', dict(ensemble_size=10),
         "smooth data; NF gets the sky-shape best (lowest costheta "
         "KS+bin-res in the cracked suite)"),
    ('smooth', '1d_marginal'):
        ('NormalizingFlowKDE', dict(ensemble_size=10),
         "smooth data; NF wins the rate-weighted log10(vinf) shape on "
         "every smooth/anisotropic scenario"),
    ('smooth', 'density'):
        ('NormalizingFlowKDE', dict(ensemble_size=10),
         "smooth data; NF wins log10(density) KS on every "
         "smooth/anisotropic scenario"),
    # multi-scale velocity (cold+hot-like)
    ('multi_scale', 'rate'):
        ('cvAdaptiveKDE', dict(scalings='narrow'),
         "multi-scale velocity at xloc (narrow component detected "
         "locally); cvAdapt-narrow tightens the kernel to the cold-"
         "component scale and recovers ~100% of the rate"),
    ('multi_scale', 'sky_map'):
        ('gaussianKDEWrapper', {},
         "multi-scale at xloc; the global kernel of scipy.stats."
         "gaussian_kde happens to hit the right average bandwidth on "
         "the sky even though its rate is wrong - wins costheta KS"),
    ('multi_scale', '1d_marginal'):
        ('cvAdaptiveKDE', dict(scalings='narrow'),
         "multi-scale at xloc; cvAdapt-narrow has the best v_marginal "
         "fit (narrowest cold-component reconstruction)"),
    ('multi_scale', 'density'):
        ('NormalizingFlowKDE', dict(ensemble_size=10),
         "multi-scale at xloc; NF still wins log10(density) KS - it "
         "learns the bimodal mixture from data without a per-axis "
         "scale prior"),
    # narrow coherent flow (disk-stream-like)
    ('narrow_coherent', 'rate'):
        ('cvAdaptiveKDE', {},
         "extreme coherent local flow through xloc (sigma_v_local/sigma_v_global "
         "< 0.01); cvAdaptive (default scalings) recovers the rate "
         "via per-particle local Sigma_i along the stream - scipy/NF "
         "under-recover by >=50x"),
    ('narrow_coherent', 'sky_map'):
        ('gaussianKDEWrapper', {},
         "narrow coherent flow; scipy paradoxically wins sky-shape "
         "because cvAdapt's narrow kernels create sky-map artifacts "
         "and NF over-smooths the tight directional band"),
    ('narrow_coherent', '1d_marginal'):
        ('NormalizingFlowKDE', dict(ensemble_size=10),
         "narrow coherent flow; NF wins 1D vinf shape - the rate-weighted "
         "speed distribution is captured well by the flow's global "
         "fit even though the absolute rate is low"),
    ('narrow_coherent', 'density'):
        ('NormalizingFlowKDE', dict(ensemble_size=10),
         "narrow coherent flow; NF is least-bad at log10(density) - "
         "all methods structurally fail here, NF less so than scipy/"
         "cvGauss"),
    # many narrow manifolds (spiky-ball-like)
    ('many_narrow', 'rate'):
        ('cvAdaptiveKDE', {},
         "many narrow 1D-ish manifolds (local_cond_med > 1e4); "
         "cvAdaptive's per-particle local Sigma_i captures each manifold's "
         "direction - wins rate by a wide margin"),
    ('many_narrow', 'sky_map'):
        ('gaussianKDEWrapper', {},
         "many narrow manifolds; scipy's global kernel is the only "
         "thing keeping the sky distribution smooth at the cost of "
         "the rate"),
    ('many_narrow', '1d_marginal'):
        ('NormalizingFlowKDE', dict(ensemble_size=10),
         "many narrow manifolds; NF gives the best (still imperfect) "
         "log10(vinf) shape recovery"),
    ('many_narrow', 'density'):
        ('cvAdaptiveKDE', {},
         "many narrow manifolds; cvAdaptive is least-bad on "
         "log10(density) - all methods fail here but cvAdapt closer "
         "than NF/scipy"),
}


# Recommendation logic

VALID_TASKS = {'rate', 'sky_map', '1d_marginal', 'density', 'sample'}


def recommend_method(coords: np.ndarray, task: str = 'rate', xloc=None, vloc=None, encounter_radius_pc: float | None = None, R_proxy_pc: float = 1.0, K_local: int = 50, n_near_min: int = 20, verbose: bool = True) -> dict:
    """Recommend a KDE method given a dataset and a task.

    Parameters
    ----------
    coords : (N, 6) array
        Phase-space data: 3 positions + 3 velocities.
    task : {'rate', 'sky_map', '1d_marginal', 'density', 'sample'}
        What you intend to do with the estimator.  Routing comes from
        the regime-map empirical winners - sky_map/1d_marginal/density
        give different recommendations than rate even on the same data.
    xloc, vloc : (3,) array-like, optional
        Encounter target position / velocity.  When `xloc` is provided
        the recommender runs `analyze_local(coords, xloc)` and uses the
        local diagnostics (coh_min, ms_min, n_near) to classify the
        structural regime - this is what's needed to distinguish a
        smooth ring (scipy/NF fine) from a stream passing through xloc
        (cvAdaptive mandatory).  Without xloc the recommender falls
        back to global flags only and may misclassify near-target
        structure.
    encounter_radius_pc : float, optional
        Encounter sphere radius (for production cracked: 0.1 pc).  Used
        only in the rationale text, not in routing.
    R_proxy_pc : float
        Radius for the n_near count (default 1.0).
    K_local : int
        K used in analyze_local's KNN of xloc (default 50).
    n_near_min : int
        Below this n_near, the rate estimate is flagged as fundamentally
        noisy regardless of method choice.

    Returns
    -------
    dict
        - method : str - class name in `cracked` (one of NormalizingFlowKDE,
                          cvAdaptiveKDE, cvGaussianKDE, gaussianKDEWrapper)
        - method_args : dict of kwargs for the constructor
        - structural_class : str - smooth | multi_scale | narrow_coherent
                              | many_narrow
        - rationale : str (human-readable)
        - diagnostics : dict with 'global' (from analyze_data) and
                          'local' (from analyze_local) sub-dicts.
    """
    if task not in VALID_TASKS:
        raise ValueError(f"task must be one of {sorted(VALID_TASKS)}, got "
                         f"{task!r}")

    gdiag = analyze_data(coords)
    N = gdiag['N']
    ldiag = analyze_local(coords, xloc, R_proxy=R_proxy_pc, K=K_local) \
        if xloc is not None else None

    structural = classify_structure(ldiag, gdiag)

    # task='sample' bypasses the table - only NF generates samples.
    if task == 'sample':
        method = 'NormalizingFlowKDE'
        method_args = dict(ensemble_size=10)
        rationale = (
            "task='sample' - only the NormalizingFlow ensemble provides "
            "a true generative model. cv*KDE.draw() resamples from the "
            "training support, not from a learned manifold.")
    else:
        method, method_args, rationale = _METHOD_TABLE[(structural, task)]
        method_args = dict(method_args)  # defensive copy

    # Append low-n_near caveat to rationale if applicable.
    if ldiag is not None and ldiag['n_near'] < n_near_min \
            and task in ('rate', 'sky_map'):
        rationale = (rationale + f"  Caveat: only {ldiag['n_near']} "
                     f"particles within {R_proxy_pc} pc of xloc - the "
                     f"rate is fundamentally noisy at this dataset size "
                     f"regardless of method choice (this is a data "
                     f"limit, not a method limit).")

    rec = dict(method=method, method_args=method_args, structural_class=structural, task=task, rationale=rationale, diagnostics=dict(global_=gdiag, local=ldiag))
    if verbose:
        print_recommendation(rec, task=task)
    return rec


def recommend_for_isostream(coords, *, xloc, vloc=None, R_proxy_pc: float = 1.0, tasks=('rate', 'sky_map', '1d_marginal', 'density'), verbose: bool = False) -> dict:
    """Run `recommend_method` for several tasks on one isostream.

    Convenience for production code that wants the recommended method
    for each task it's about to compute.  Reuses `analyze_data` /
    `analyze_local` results (which are independent of `task`), so the
    cost is one analyse + N table lookups regardless of `tasks` length.

    Returns dict task -> recommendation.  Each recommendation has the
    same shape as `recommend_method`'s return value.
    """
    gdiag = analyze_data(coords)
    ldiag = analyze_local(coords, xloc, R_proxy=R_proxy_pc) \
        if xloc is not None else None
    structural = classify_structure(ldiag, gdiag)

    out = {}
    for task in tasks:
        if task not in VALID_TASKS:
            raise ValueError(f"unknown task {task!r}")
        if task == 'sample':
            method, args, rationale = (
                'NormalizingFlowKDE', dict(ensemble_size=10),
                "task='sample' - NF is the only generative estimator.")
        else:
            method, args, rationale = _METHOD_TABLE[(structural, task)]
        out[task] = dict(method=method, method_args=dict(args), structural_class=structural, task=task, rationale=rationale, diagnostics=dict(global_=gdiag, local=ldiag))
        if verbose:
            print_recommendation(out[task], task=task)
    return out


def print_recommendation(rec: dict, task: str = '<unspecified>') -> None:
    """Pretty-print a recommendation dict to stdout."""
    diag = rec.get('diagnostics', {})
    # Back-compat: old shape had analyze_data fields flat at top level.
    gdiag = diag.get('global_', diag)
    ldiag = diag.get('local')
    print(f"\n-- Method recommendation (task: {task}) --")
    print(f"  N = {gdiag['N']:,}  ({gdiag['d']}D)")
    if 'cov_cond_number' in gdiag:
        print(f"  6D cov cond:             {gdiag['cov_cond_number']:.2g}")
    if 'local_trace_ratio' in gdiag:
        print(f"  global local trace ratio:{gdiag['local_trace_ratio']:.3f}")
    if 'local_cond_med' in gdiag:
        print(f"  global local cov cond:   {gdiag['local_cond_med']:.2g}  "
              f"(per-cluster, median over data)")
    if ldiag is not None:
        print(f"  -- local (at xloc) --")
        print(f"  n_near (R_proxy):        {ldiag['n_near']}")
        print(f"  coherence_min:           {ldiag['coherence_min']:.4f}  "
              f"(sigma_v_local / sigma_v_global; << 1 => coherent flow through xloc)")
        print(f"  multi_scale_min:         {ldiag['multi_scale_min']:.4f}  "
              f"(sigma_narrow / sigma_v_local; << 1 => multi-scale velocity at xloc)")
    flags = []
    for k in ('multi_scale_flag', 'narrow_feature_flag', 'stream_like_flag'):
        if gdiag.get(k):
            flags.append(k.replace('_flag', ''))
    print(f"  global flags:            {flags if flags else ['smooth']}")
    if 'structural_class' in rec:
        print(f"  -> structural class:      {rec['structural_class']}")
    print(f"\n  -> use {rec['method']}")
    if rec['method_args']:
        print(f"     with kwargs: {rec['method_args']}")
    print(f"\n  rationale:")
    line = ""
    for w in rec['rationale'].split():
        if len(line) + len(w) > 72:
            print(f"    {line}")
            line = w
        else:
            line = (line + " " + w).lstrip()
    if line:
        print(f"    {line}")
    print()


# CLI entrypoint

def _load_coords(path: str) -> np.ndarray:
    """Load a (N, 6) coords array from a file. Supports .npy, .pickle/.pik
    (via dill), .csv (whitespace or comma-separated)."""
    if path.endswith('.npy'):
        return np.load(path)
    if path.endswith('.csv') or path.endswith('.txt'):
        return np.loadtxt(path, delimiter=None)
    if path.endswith(('.pickle', '.pik', '.dill')):
        import dill
        with open(path, 'rb') as f:
            obj = dill.load(f)
        if isinstance(obj, np.ndarray):
            return obj
        if isinstance(obj, dict):
            # Try common keys
            for k in ('coords', 'data', 'xv', 'phase_space'):
                if k in obj and isinstance(obj[k], np.ndarray):
                    return obj[k]
            raise ValueError(f"pickle dict has no recognized coords key; "
                             f"keys are {list(obj.keys())[:8]}")
        raise ValueError(f"pickle object is type {type(obj).__name__}; "
                         f"can't extract coords")
    raise ValueError(f"unrecognized file extension on {path}")


def main():
    parser = argparse.ArgumentParser(description="Recommend a KDE method for a 6D dataset.")
    parser.add_argument("data_path", help="Path to data file (.npy, .csv, .pik/.pickle)")
    parser.add_argument("--task", default="rate", choices=sorted(VALID_TASKS), help="Task you intend to use the estimator for.")
    parser.add_argument("--xloc", default=None, type=float, nargs=3, help="Sun/target position (3 floats).")
    parser.add_argument("--vloc", default=None, type=float, nargs=3, help="Sun/target velocity (3 floats).")
    parser.add_argument("--encounter_radius_pc", default=1.0, type=float)
    args = parser.parse_args()

    coords = _load_coords(args.data_path)
    print(f"loaded {args.data_path}: shape = {coords.shape}")
    recommend_method(coords, task=args.task, xloc=args.xloc, vloc=args.vloc, encounter_radius_pc=args.encounter_radius_pc, verbose=True)


if __name__ == "__main__":
    main()
