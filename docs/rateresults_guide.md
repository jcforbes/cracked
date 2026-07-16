# Reading `*_rateresults.pik`

A `*_rateresults.pik` file is the per-isostream output of the
production driver (`isostreams_prod.py`). One file per isostream. It
holds the importance-sampled rate integral and its differential
breakdowns (sky map, log10 vinf), per-estimator sample arrays, the
post-hoc reliability diagnostics, and the pre-run recommender's
prediction of which estimator should have been used.

This file is a **data dictionary** - it tells you what's in the
pickle and how to compute things from it. For *picking* an estimator,
see:

- [picking_pre_run.md](picking_pre_run.md) - before you've run the
  rate, with only training data in hand
- [picking_post_run.md](picking_post_run.md) - after the rate is
  computed and reliability diagnostics are available (this is the
  guide that overrides the pre-run pick when both exist)

Naming: for an isostream CSV named `foo.csv`, the rateresults file is
`foo_rateresults.pik`.

## Loading

```python
import pickle

# If the file was saved on a CUDA node and you're on CPU, call this
# *before* the load so torch deserialises GPU tensors to CPU.
from cracked.normalizing_flow import patch_torch_load_to_cpu
patch_torch_load_to_cpu()

with open("foo_rateresults.pik", "rb") as f:
    rr = pickle.load(f)

print(sorted(rr.keys()))
```

The file is a flat dict. Keys grouped below by purpose.

## Geometry and proposal

| Key | Type | Meaning |
|---|---|---|
| `xsunj` | (3,) | Sun position in the isostream's data frame, pc |
| `vsunj` | (3,) | Sun velocity in the data frame, pc/Myr |
| `v_proposal_mean` | (K, 3) | IS proposal mixture component means |
| `v_proposal_cov`  | (K, 3, 3) | IS proposal component covariances |
| `encounter_r_pc`  | float | Encounter sphere radius (typically 0.1 pc) |
| `qmaxAU` | float | Maximum encounter pericenter (typically 5 AU) |
| `fac` | float | Density normalisation (`mprog * isos_per_msun`) |
| `mprog` | float | Progenitor mass, M_sun |
| `isos_per_msun` | float | Isostreams per progenitor M_sun |
| `nboot` | int | Number of IS draws per estimator |

## Per-estimator rate samples

Five estimator labels are stored:

| Label | Estimator |
|---|---|
| `scipy` | `gaussianKDEWrapper` (Scott's-rule scipy.stats.gaussian_kde) |
| `cv_rate` | `cvAdaptiveKDE` pick optimised for rate (no N_eff floor) |
| `cv_sky`  | `cvAdaptiveKDE` pick for the (cos theta, phi) sky map |
| `cv_vinf` | `cvAdaptiveKDE` pick for the log10 vinf marginal |
| `nf` | `NormalizingFlowKDE` (MAF ensemble, size 10) |

Per label, the file stores (each is a 1D array of length `nboot`):

| Key | Meaning |
|---|---|
| `resj_<label>` | Rate-integrand contribution per IS sample, **already multiplied by `fac`** |
| `vsphere_<label>` | \|v_rel\| at the encounter sphere |
| `costhetasphere_<label>` | cos theta of the encounter direction |
| `phisphere_<label>` | phi of the encounter direction |
| `vinftys_<label>` | vinf (asymptotic speed) |
| `eccentricities_<label>` | Orbit eccentricity |
| `thetacs_<label>` | Critical theta (gravitational focusing half-angle) |

### Computing the rate and its differentials

```python
import numpy as np

label = "cv_rate"        # pick the estimator you trust (see post-run guide)
resj = rr[f"resj_{label}"]
N = len(resj)

# Total rate, encounters per Myr (x whatever fac's units are)
R = resj.sum() / N

# IS-side Kish ESS - how many samples actually carry the rate
kish = (resj.sum())**2 / (resj**2).sum()
print(f"R = {R:.3g} /Myr,  IS Kish ESS = {kish:.1f} / {N}")

# Sky map (rate per steradian) - rate-weighted 2D histogram of (cos theta, phi)
costh = rr[f"costhetasphere_{label}"]
phi   = rr[f"phisphere_{label}"]
h, ce, pe = np.histogram2d(costh, phi, bins=[18, 28],
                            range=[(-1, 1), (-np.pi, np.pi)],
                            weights=resj / N)
# h has units of rate per bin; divide by bin area for dR/dOmega.

# 1D vinf marginal (rate per d log10 vinf)
vinf = rr[f"vinftys_{label}"]
log_v = np.log10(np.maximum(vinf, 1e-12))
hist, edges = np.histogram(log_v, bins=40, weights=resj / N)
```

## Reliability metrics

These power the post-run selection rules in
[picking_post_run.md](picking_post_run.md).

| Key | Meaning |
|---|---|
| `data_neff_<label>` | Kish ESS of *training-data* contributions to the rate sum. Low => kernel saw only a few particles. NaN for `nf` (NF isn't a kernel method). |
| `nf_rate_per_flow` | (n_ens,) - rate computed by each MAF flow individually |
| `nf_rate_ensemble_mean` | Mean of the per-flow rates |
| `nf_rate_ensemble_std`  | Std of the per-flow rates |
| `nf_rate_ensemble_cv`   | std / mean - the NF self-confidence headline |

## Pre-run recommendation

The recommender (`cracked.recommend.recommend_for_isostream`) was
called on the training data before the rate was computed. The keys
below are the **pre-run** prediction - they're what the structural
heuristic would have chosen with no access to the actual rates or
their reliability. See [picking_pre_run.md](picking_pre_run.md) for
the structural classes and method table; see
[picking_post_run.md](picking_post_run.md) for when to override
these with the post-run pick.

| Key | Meaning |
|---|---|
| `recommended_method_rate`        | Pre-run pick for the **rate** task |
| `recommended_method_sky_map`     | Pre-run pick for the **sky map** |
| `recommended_method_1d_marginal` | Pre-run pick for the **log10 vinf marginal** |
| `recommended_method_density`     | Pre-run pick for **6D density** |
| `recommended_class_<task>`       | Structural class - one of `smooth`, `multi_scale`, `narrow_coherent`, `many_narrow`. Same across tasks. |
| `recommend_n_near`               | # training particles within ~1 pc of the Sun (3D pos) |
| `recommend_coherence_min`        | min over v-axes of sigma_v_local / sigma_v_global at the Sun |
| `recommend_multi_scale_min`      | min over v-axes of sigma_narrow / sigma_v_local at the Sun |
| `recommend_local_trace_ratio`    | median local-cluster trace / global trace |
| `recommend_local_cond_med`       | median local-cluster covariance condition number |
| `recommend_cov_cond_number`      | Global 6D covariance condition number |
| `recommend_narrow_feature_flag`  | Bool - global narrow features detected |
| `recommend_multi_scale_flag`     | Bool - global multi-scale velocity dispersion |

## Other contents

| Key | Meaning |
|---|---|
| `cv_diag` | Dict of per-cvAdapt-pick diagnostics (covfac, alpha, shrinkage, scaling, asymptotic correction). One sub-dict per `cv_rate / cv_sky / cv_vinf`. |

## Quick reference - what to read first

1. Sanity-check: `rr["nboot"]`, `rr["fac"]`, `rr["xsunj"]`, `rr["vsunj"]`.
2. Decide which label to trust per task using
   [picking_post_run.md](picking_post_run.md) (or fall back to the
   `recommended_method_<task>` pre-run pick if the post-run keys
   aren't there).
3. Compute the rate and its differentials from
   `resj_<label>` and the per-IS-sample arrays.
4. Sanity-check the IS-side Kish ESS of `resj_<label>` against `nboot`.
