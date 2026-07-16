# Picking an estimator after you've run the rate

Use this when you have a `*_rateresults.pik` from the production
driver - i.e. all five estimators (`scipy`, `cv_rate`, `cv_sky`,
`cv_vinf`, `nf`) have already been evaluated and their reliability
diagnostics are sitting in the pickle. This information dominates
the pre-run heuristic; trust the post-run rule whenever both are
available.

If you only have training data, use
[picking_pre_run.md](picking_pre_run.md).

## The two reliability signals

| Signal | What it measures | "Reliable" threshold |
|---|---|---|
| `data_neff_<label>` | Kish ESS of *training-data* contributions to the rate sum. Low => the kernel saw only 1-2 particles in the rate-leverage region. NaN for NF. | `> 30` |
| `nf_rate_ensemble_cv` | std / mean of per-flow rates across the MAF ensemble. Large => NF is extrapolating wildly and the flows disagree. | `< 0.3` |

These are the only two numbers the post-run rules depend on. The
30 and 0.3 thresholds are loose by design - they're the boundaries
between "you have something" and "every estimator is guessing," not
finely-tuned.

## The per-task rules

The rules use both the reliability signals **and** the
`recommended_class_<task>` pre-run structural class (also in the
pickle - see [picking_pre_run.md](picking_pre_run.md) for what the
four classes mean). Including the class is what makes the post-run
rule actually beat NF-default - empirically, the classes
correctly identify the cases where the kernel methods beat NF and
the reliability metrics alone don't.

| Task | Rule |
|---|---|
| **rate** | `cv_rate` if class in {`multi_scale`, `narrow_coherent`, `many_narrow`}; else `nf` if `nf_rate_ensemble_cv < 0.3`; else `cv_rate` if `data_neff_cv_rate > 30`; else `scipy` if `data_neff_scipy > 30`; else flag |
| **sky map** | `scipy` if class is `multi_scale`; else `nf` if NF reliable; else `scipy` |
| **1D marginal** | same as sky map |
| **6D density** | `nf` if NF reliable; else `cv_rate` |

The empirical justifications (from the 9-scenario regime test suite -
see `scripts/check_post_run_heuristic.py` for the validation):

- **Rate.** On the four structurally-hard classes (multi_scale,
  narrow_coherent, many_narrow), cv_rate beats NF on rate even when
  NF's ensemble is well-converged - the adaptive bandwidth resolves
  the rate-leverage region better than NF's smooth interpolant. On
  smooth/well-resolved data, NF wins because both methods are accurate
  there and NF's smoothness helps in the IS-sample weighting.
- **Sky / 1D marginal.** NF wins by default - smoother sky maps and
  cleaner 1D recovery. The one consistent counterexample is
  `multi_scale` data (cold+hot mixture), where scipy's global
  bandwidth captures the dominant component better than NF.
- **6D density.** NF wins almost everywhere. The cv_rate fallback
  only fires when NF's ensemble doesn't converge.

### Match rate against the test suite

| Rule | rate | sky | 1D | density | total |
|---|---|---|---|---|---|
| naive NF-default per-task | 5/9 | 4/9 | 6/9 | 7/9 | 22/36 |
| **this guide** | **7/9** | **5/9** | **7/9** | **7/9** | **26/36 (72%)** |

The remaining 10/36 misses are mostly cases where the structural
class is genuinely ambiguous (disk_stream and stream_ring are both
`narrow_coherent` but disagree on whether NF or scipy wins the sky).
Resolving these would require either richer class labels or
direct per-scenario lookup - both of which would overlearn the test
set. The 72% rule is the simplest one that captures every pattern
visible across multiple scenarios.

## "What if everything is unreliable?"

If `data_neff_cv_rate < 30` AND `data_neff_scipy < 30` AND
`nf_rate_ensemble_cv > 0.3`, you have no trustworthy estimator on this
isostream. Either:

- The data is too sparse near the encounter sphere for any method to
  resolve the rate-leverage region. Look at `recommend_n_near` - if
  it's below 10, you're in extrapolation territory regardless of
  method.
- The NF training failed (look at `nf_rate_per_flow` - a multi-modal
  histogram across flows is the signature). Re-run with a different
  random seed or a deeper architecture.

Flag the isostream and move on; don't pick the "least bad" estimator
and pretend it's reliable.

## Calling it

A simple decoder, given a loaded rateresults dict `rr`:

```python
HARD = {'multi_scale', 'narrow_coherent', 'many_narrow'}

def pick_post_run(rr, task,
                  neff_floor=30.0, nf_cv_max=0.3):
    nf_ok = rr.get('nf_rate_ensemble_cv', float('inf')) < nf_cv_max
    cls = rr.get(f'recommended_class_{task}',
                 rr.get('recommended_class_rate'))

    if task == 'rate':
        if cls in HARD:
            return 'cv_rate'
        if nf_ok:
            return 'nf'
        if rr.get('data_neff_cv_rate', 0) > neff_floor:
            return 'cv_rate'
        if rr.get('data_neff_scipy', 0) > neff_floor:
            return 'scipy'
        return None       # nothing trustworthy

    if task in ('sky_map', '1d_marginal'):
        if cls == 'multi_scale':
            return 'scipy'
        return 'nf' if nf_ok else 'scipy'

    if task == 'density':
        return 'nf' if nf_ok else 'cv_rate'

    raise ValueError(f"unknown task {task!r}")
```

Use it to map a rateresults pickle to a per-task estimator choice;
then read `resj_<label>`, `costhetasphere_<label>`, etc. for that
label.

## Cross-checking against the pre-run recommendation

Both selections are stored in the pickle: the post-run pick (which
this guide computes) and the pre-run pick (`recommended_method_<task>`,
written by `recommend_for_isostream`). When they agree, you have high
confidence. When they disagree, the post-run pick is the better answer
*on this isostream*, but the disagreement itself is interesting -
that's where the structural heuristic missed something about the
data, or where a structurally-good fit happens to have low
N_eff on this particular sphere geometry.

A quick audit:

```python
import pickle, glob
from cracked.normalizing_flow import patch_torch_load_to_cpu
patch_torch_load_to_cpu()

LABEL_TO_METHOD = {
    'scipy': 'gaussianKDEWrapper',
    'cv_rate': 'cvAdaptiveKDE',
    'nf': 'NormalizingFlowKDE',
}

agreed = disagreed = 0
for path in sorted(glob.glob('roinone_prod/*_rateresults.pik')):
    with open(path, 'rb') as f:
        rr = pickle.load(f)
    post = pick_post_run(rr, 'rate')
    pre = rr.get('recommended_method_rate')
    pre_label = {v: k for k, v in LABEL_TO_METHOD.items()}.get(pre)
    if post == pre_label:
        agreed += 1
    else:
        disagreed += 1
        print(f"  {path}: pre={pre_label}  post={post}")
print(f"agreed on {agreed}/{agreed+disagreed} isostreams")
```

A high disagreement rate => the pre-run heuristic is mis-classifying
your production data structurally and you should look at which
triplet boundary is being crossed wrongly.

## Why this overrides the pre-run guide

The pre-run heuristic has to commit before it can see how the data
actually distributes around the encounter sphere. The post-run
diagnostics measure the thing that actually matters for rate
fidelity - *how many training particles contributed to the rate
sum*. A cvAdaptive that scored well on the regime map can still
collapse to N_eff=3 on an isostream where the Sun happens to sit in
a low-density pocket; the pre-run heuristic can't see that.

So: when both are available, the post-run pick is what you should
quote. Keep the pre-run pick in the pickle as a sanity-check, not as
the primary recommendation.
