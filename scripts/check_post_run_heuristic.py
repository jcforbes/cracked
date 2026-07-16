"""Validate the post-run heuristic in docs/picking_post_run.md against
the empirical winner per (scenario, task) in compare_*_stats.json.

For each scenario x task, the script:
  1. Identifies the empirical-winner estimator by the most relevant
     accuracy metric (e.g. |R/R_an - 1| for rate, binres+KS for sky).
  2. Applies one of several candidate post-run rules and reports
     match rate per task.

Candidate rules under test:
  v1 (as written in docs/picking_post_run.md)
  v2 ("NF if reliable, else family fallback per task")
  v3 (NF-default, but cv_rate for rate when the pre-run structural
      class is one of {multi_scale, narrow_coherent, many_narrow})

Run:
    ipython3 scripts/check_post_run_heuristic.py
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir))


LABEL_OF = {
    'scipy_kde':              'scipy',
    'cvAdaptive':             'cv_rate',
    'cvAdapt-smart':          'cv_rate',
    'cvGaussianKDE':          'scipy',
    'NormalizingFlow (MAF)':  'nf',
    'EnBiD (ngb=64)':         'enbid',
    'analytic_DF':            'analytic',
}

# Pre-run structural class for each scenario (kept here so the validation is
# self-contained; the live inference goes through
# cracked.recommend.classify_structure).
STRUCTURAL_CLASS = {
    'isotropic':    'smooth',
    'drifting':     'smooth',
    'bimodal':      'smooth',           # mild bimodality but locally smooth
    'ring':         'smooth',           # ring in (v_x, v_y) is locally smooth at Sun
    'stream_ring':  'narrow_coherent',  # narrow tangential dispersion
    'cold+hot':     'multi_scale',
    'cosine_shear': 'smooth',           # globally structured, locally smooth-ish
    'disk_stream':  'narrow_coherent',  # sigma_t = 0.1
    'spiky_ball':   'many_narrow',
}


def pick_v1(reliability, task, structural_class=None, neff_floor=10.0, nf_cv_max=0.3, assume_nf_reliable=False):
    """The rule as currently in docs/picking_post_run.md.
    Rate: cv_rate if neff high, else scipy. Sky/1D: NF or scipy.
    Density: NF or cv_rate.
    """
    nf_cv = reliability.get('nf_rate_ensemble_cv', float('inf'))
    nf_ok = assume_nf_reliable or nf_cv < nf_cv_max
    if task == 'rate':
        if reliability.get('data_neff_cv_rate', 0) > neff_floor:
            return 'cv_rate'
        if reliability.get('data_neff_scipy', 0) > neff_floor:
            return 'scipy'
        return None
    if task in ('sky_map', '1d_marginal'):
        return 'nf' if nf_ok else 'scipy'
    if task == 'density':
        return 'nf' if nf_ok else 'cv_rate'
    raise ValueError(task)


def pick_v2(reliability, task, structural_class=None, neff_floor=10.0, nf_cv_max=0.3, assume_nf_reliable=False):
    """NF-default per-task. NF wins almost everywhere when it converges;
    falls back to the per-task kernel family when NF is unreliable.
    Ignores structural class.
    """
    nf_cv = reliability.get('nf_rate_ensemble_cv', float('inf'))
    nf_ok = assume_nf_reliable or nf_cv < nf_cv_max
    if nf_ok:
        return 'nf'
    if task == 'rate':
        if reliability.get('data_neff_cv_rate', 0) > neff_floor:
            return 'cv_rate'
        if reliability.get('data_neff_scipy', 0) > neff_floor:
            return 'scipy'
        return None
    if task in ('sky_map', '1d_marginal'):
        return 'scipy'
    if task == 'density':
        return 'cv_rate'
    raise ValueError(task)


def pick_v3(reliability, task, structural_class=None, neff_floor=10.0, nf_cv_max=0.3, assume_nf_reliable=False):
    """Hybrid: NF-default, but use the pre-run structural class to redirect
    when its empirical pattern says NF loses. The class-redirect rules
    encode (and only encode) patterns visible across multiple scenarios.
    """
    nf_cv = reliability.get('nf_rate_ensemble_cv', float('inf'))
    nf_ok = assume_nf_reliable or nf_cv < nf_cv_max

    # Empirical rate redirect: on the three "hard" structural classes,
    # cv_rate consistently beats NF on rate (cold+hot, cosine_shear,
    # disk_stream, spiky_ball - three different classes, same pattern).
    if task == 'rate' and structural_class in (
            'multi_scale', 'narrow_coherent', 'many_narrow'):
        if reliability.get('data_neff_cv_rate', 0) > neff_floor:
            return 'cv_rate'
        # cv_rate is structurally right but data is too sparse; fall
        # back to whichever has the higher signal
        return 'nf' if nf_ok else 'scipy'

    # Empirical sky/1D redirect: scipy beats NF on multi_scale and
    # narrow_coherent - the kernel-method's global bandwidth handles
    # the dominant component better than NF's smooth interpolant on
    # the leverage region.
    if task in ('sky_map', '1d_marginal') and structural_class in (
            'multi_scale', 'narrow_coherent'):
        return 'scipy'

    # Default branch - NF when reliable, family fallback per task.
    return pick_v2(reliability, task, structural_class, neff_floor=neff_floor, nf_cv_max=nf_cv_max, assume_nf_reliable=assume_nf_reliable)


def pick_v4(reliability, task, structural_class=None, neff_floor=10.0, nf_cv_max=0.3, assume_nf_reliable=False):
    """Cleaner hybrid:
      - rate: cv_rate if structural class is 'hard' (NO N_eff gate -
        the whole point of the class redirect is that the kernel
        N_eff isn't the right trust signal here); else NF; else scipy.
      - sky/1D: scipy ONLY when class is 'multi_scale' (narrow_coherent
        is ambiguous in the test set - disk_stream prefers scipy but
        stream_ring prefers NF; don't redirect on it). Default NF.
      - density: NF if reliable, else cv_rate.
    """
    nf_cv = reliability.get('nf_rate_ensemble_cv', float('inf'))
    nf_ok = assume_nf_reliable or nf_cv < nf_cv_max

    if task == 'rate':
        if structural_class in ('multi_scale', 'narrow_coherent',
                                  'many_narrow'):
            return 'cv_rate'
        if nf_ok:
            return 'nf'
        if reliability.get('data_neff_cv_rate', 0) > neff_floor:
            return 'cv_rate'
        return 'scipy' if reliability.get('data_neff_scipy', 0) > neff_floor else None

    if task in ('sky_map', '1d_marginal'):
        if structural_class == 'multi_scale':
            return 'scipy'
        return 'nf' if nf_ok else 'scipy'

    if task == 'density':
        return 'nf' if nf_ok else 'cv_rate'

    raise ValueError(task)


def empirical_winner(estimators, task):
    candidates = [e for e in estimators
                  if LABEL_OF.get(e['name']) not in (None, 'analytic',
                                                     'enbid')]

    def score(e):
        if task == 'rate':
            return abs(e.get('R_relative', 0) - 1)
        if task == 'sky_map':
            return e.get('binres_costh', 1) + e.get('ks_costh', 1)
        if task == '1d_marginal':
            return e.get('binres_logvee', 1) + e.get('ks_logvee', 1)
        if task == 'density':
            return e.get('ks_logdens', 1)
        raise ValueError(task)

    if not candidates:
        return None
    best = min(candidates, key=score)
    return LABEL_OF.get(best['name']), best['name'], score(best)


def reliability_for_label(estimators, label):
    for e in estimators:
        if LABEL_OF.get(e['name']) == label:
            return float(e.get('per_eval_neff_med', 0) or 0)
    return 0.0


def build_reliability(estimators):
    return {
        'data_neff_cv_rate': reliability_for_label(estimators, 'cv_rate'),
        'data_neff_scipy':   reliability_for_label(estimators, 'scipy'),
    }


_SHORT = {
    'isotropic Maxwellian': 'isotropic',
    'drifting Maxwellian': 'drifting',
    'bimodal': 'bimodal',
    'ring in $(v_x': 'ring',
    'stream ring $R=159$ pc': 'stream_ring',
    'cold+hot mixture': 'cold+hot',
    'cosine shear $v_z = A': 'cosine_shear',
    'disk stream $R=8$ kpc': 'disk_stream',
    'spiky ball $N_{': 'spiky_ball',
}

def short_scenario(s):
    head = s.split(',')[0].split(':')[0].strip()
    for prefix, short in _SHORT.items():
        if head.startswith(prefix):
            return short
    return head


def main():
    files = sorted(glob.glob('compare_*_stats.json'))
    if not files:
        print('no compare_*_stats.json files found')
        return 1
    tasks = ('rate', 'sky_map', '1d_marginal', 'density')
    rules = {'v1 (docs)': pick_v1, 'v2 (NF-default)': pick_v2,
             'v3 (NF + class redirect)': pick_v3,
             'v4 (rate-always-cv on hard / sky-redirect only on multi_scale)':
                 pick_v4}

    for rule_name, rule_fn in rules.items():
        for assume_nf in (True, False):
            label = ('NF reliable' if assume_nf
                     else 'NF unreliable everywhere')
            print(f"\n=== {rule_name}, {label} ===")
            print(f"{'scenario':<14s} {'task':<13s} "
                  f"{'rule':>10s}  {'emp':>10s}  ok?")
            print('-' * 60)
            agreed = total = 0
            per_task = {t: [0, 0] for t in tasks}
            for f in files:
                with open(f) as fh:
                    s = json.load(fh)
                sc = short_scenario(s['scenario'])
                cls = STRUCTURAL_CLASS.get(sc, 'smooth')
                ests = s['estimators']
                rel = build_reliability(ests)
                for task in tasks:
                    rule_pick = rule_fn(rel, task, structural_class=cls, assume_nf_reliable=assume_nf)
                    emp = empirical_winner(ests, task)
                    if emp is None:
                        continue
                    emp_label, emp_name, emp_score = emp
                    ok = (rule_pick == emp_label)
                    agreed += int(ok); total += 1
                    per_task[task][0] += int(ok); per_task[task][1] += 1
                    flag = ' ' if ok else '*'
                    print(f"{sc:<14s} {task:<13s} "
                          f"{str(rule_pick):>10s}  "
                          f"{emp_label or '?':>10s}  {flag}")
            print(f"\n  total: {agreed}/{total}", end='   ')
            for t, (a, n) in per_task.items():
                print(f"{t}={a}/{n}", end='  ')
            print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
