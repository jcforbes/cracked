# Picking an estimator before you've run the rate

Use this when you have an isostream's training data in hand but haven't
yet computed the rate. You need a single estimator per task and you
can't look at post-hoc reliability diagnostics. The pre-run heuristics
in `cracked.recommend` are built around three local-structure
statistics computed at the Sun's location in 6D phase space.

If you have **already** produced a `*_rateresults.pik` file with all
five estimator outputs, use the post-run heuristics in
[picking_post_run.md](picking_post_run.md) instead - they're more
accurate because they see the actual data N_eff and NF ensemble
spread.

## The triplet

`cracked.recommend.analyze_local(coords, xloc)` returns three numbers:

| Metric | What it measures | What "small" means |
|---|---|---|
| `n_near` | # training particles within ~1 pc of the Sun (3D position) | < ~30 => data is sparse near the encounter sphere; *every* estimator is operating in extrapolation regime |
| `coherence_min` | min over v-axes of sigma_v_local / sigma_v_global at the Sun | < ~0.1 => coherent stream flow through the Sun (the local v-distribution is much narrower than the global one) |
| `multi_scale_min` | min over v-axes of sigma_narrow / sigma_v_local at the Sun | < ~0.1 => cold+hot mixture locally (a narrow component sits inside a wider one along the same axis) |

The thresholds are deliberately loose - they're the boundaries between
structurally distinct regimes, not finely-tuned hyperparameters.

## Structural classes

`classify_structure` collapses the triplet into one of four classes:

| Class | Triggered by | Example dataset |
|---|---|---|
| `smooth` | None of the others fire | isotropic Maxwellian, drifting Maxwellian |
| `narrow_coherent` | `coherence_min < 0.01` | thin stream passing through the Sun (disk-stream-like) |
| `multi_scale` | `multi_scale_min < 0.1` | cold+hot mixture |
| `many_narrow` | Global `narrow_feature_flag` AND `local_cond_med > 1e4` | spiky / multi-modal local v-distribution (a few sharp peaks) |

`narrow_coherent` is tested before `multi_scale` so a coherent stream
isn't mislabeled as multi-scale when its local sigma_v is much smaller
than the global. The 0.01 / 0.1 / 1e4 thresholds come from the
regime-map empirical study.

## Per-task method table

`recommend_for_isostream(coords, xloc=..., vloc=..., tasks=(...))`
maps (class, task) to a method via `cracked.recommend._METHOD_TABLE`.
The condensed form:

| Class | rate | sky_map | 1d_marginal | density |
|---|---|---|---|---|
| smooth | cvAdaptive | scipy | NF | NF |
| multi_scale | cvAdaptive | scipy | NF | NF |
| narrow_coherent | cvAdaptive | NF | NF | NF |
| many_narrow | cvAdaptive | NF | NF | NF |

Three things to notice:

1. **Rate is always cvAdaptive** - empirically the most robust on rate
   across all four classes (per the regime-map tests).
2. **1D marginal and 6D density are always NF** - NF's smooth, globally
   coherent density estimate is the clean winner whenever NF
   converges.
3. **Sky map depends on coherence/narrow structure** - scipy's single
   global bandwidth gives smoother sky histograms when the local
   v-distribution is well-resolved; NF wins on streams and many-narrow
   data where scipy's bandwidth is wrong for the leverage region.

## Calling it

```python
from cracked.recommend import recommend_for_isostream

recs = recommend_for_isostream(
    coords,                              # (N, 6) phase-space training data
    xloc=(0.0, 0.0, 0.0),                # Sun position in data frame
    vloc=(0.0, 0.0, 0.0),                # Sun velocity in data frame
    tasks=('rate', 'sky_map', '1d_marginal', 'density'),
)
for task, rec in recs.items():
    print(f"  {task}: {rec['method']:>20s}  "
          f"(class={rec['structural_class']})")
```

## Caveats

The pre-run guide commits to one method per task before it can see
how reliable that estimator actually is on this dataset. Several
failure modes are invisible to the heuristic:

- **Sparse data at the sphere** (`n_near < 10`). Every kernel method's
  rate is dominated by 1-2 particles, regardless of structural class.
  The post-run `data_neff_<label>` reveals this; the pre-run triplet
  flags it via `n_near` but the method table doesn't fall back.
- **NF training non-convergence.** NF's ensemble spread
  (`nf_rate_ensemble_cv`) is the only honest signal that the flow has
  failed; the pre-run heuristic can't predict this.
- **Hyperparameter regret.** The CV inside cvAdaptive picks per-trial
  hyperparameters that can vary substantially across seeds; the
  pre-run guide doesn't know which pick was made.

When any of these matter, use the post-run guide.

## Output keys (for cross-reference)

`recommend_for_isostream`'s output keys are persisted into the
rateresults pickle by `isostreams_prod.py`:

| Pickle key | Source |
|---|---|
| `recommended_method_<task>` | `rec['method']` |
| `recommended_class_<task>` | `rec['structural_class']` (same across tasks) |
| `recommend_n_near` | triplet `n_near` |
| `recommend_coherence_min` | triplet `coherence_min` |
| `recommend_multi_scale_min` | triplet `multi_scale_min` |
| `recommend_local_cond_med` | median local-cluster cov condition number |
| `recommend_narrow_feature_flag` | bool, global narrow feature detected |
| `recommend_multi_scale_flag` | bool, global multi-scale dispersion |

See [picking_post_run.md](picking_post_run.md) for how these get
overridden by post-run reliability metrics when available.
