"""Run the comparison suite under several CV random_state seeds and aggregate
the per-scenario rates as mean +/- std across seeds. Output goes to
`multi_seed_summary.json` and `multi_seed_summary.md`.

Each "trial" rebuilds every estimator from scratch with the given
random_state, so the variation reflects the genuine CV-MC noise after the
recent seeding fixes (and not data variation, which is fixed by the
per-scenario sampler seed).

Usage:
    ipython3 run_multi_seed.py --seeds 0 1 2 3 4
    ipython3 run_multi_seed.py --seeds 0 1 2 --tests isotropic,bimodal
"""
import argparse
import json
import os
import shutil
import time

import numpy as np

# Make the tests/ directory importable so we can reuse the scenario builders
# living next to the test file (test_rate_sphere_analytic exposes them as
# normal functions; the `test_plot_comparison_*` callables drive the suite).
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, os.pardir, 'tests'))

import test_rate_sphere_analytic as trsa


SCENARIO_FUNCTIONS = {
    'isotropic':    trsa.test_plot_comparison_isotropic,
    'drifting':     trsa.test_plot_comparison_drifting,
    'bimodal':      trsa.test_plot_comparison_bimodal,
    'ring':         trsa.test_plot_comparison_ring,
    'cosine_shear': trsa.test_plot_comparison_cosine_shear,
    'cold_hot':     trsa.test_plot_comparison_cold_hot,
    'stream':       trsa.test_plot_comparison_stream,
    'disk_stream':  trsa.test_plot_comparison_disk_stream,
}


def _patch_random_state(seed):
    """Monkey-patch the test module so all CV builds use this random_state.

    The cleanest implementation would thread `random_state` through every
    test scenario function, but the existing tests don't expose it. We
    patch the two helpers (`_build_estimators` and `_build_smart_cvadaptive`)
    so they default to the seed; that covers every estimator built by the
    plot tests.
    """
    orig_build = trsa._build_estimators
    orig_smart = trsa._build_smart_cvadaptive

    def patched_build(*args, **kwargs):
        kwargs.setdefault('random_state', seed)
        return orig_build(*args, **kwargs)

    def patched_smart(*args, **kwargs):
        kwargs.setdefault('random_state', seed)
        return orig_smart(*args, **kwargs)

    trsa._build_estimators = patched_build
    trsa._build_smart_cvadaptive = patched_smart
    return orig_build, orig_smart


def _restore(orig_build, orig_smart):
    trsa._build_estimators = orig_build
    trsa._build_smart_cvadaptive = orig_smart


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', required=True)
    ap.add_argument('--tests', type=str, default=None, help='comma-separated list of scenario names; default = all')
    ap.add_argument('--out-dir', type=str, default='multi_seed_runs', help='where to stash per-seed compare_*_stats.json copies')
    args = ap.parse_args()

    if args.tests is None:
        scenarios = list(SCENARIO_FUNCTIONS.keys())
    else:
        scenarios = [s.strip() for s in args.tests.split(',')]
        for s in scenarios:
            if s not in SCENARIO_FUNCTIONS:
                raise SystemExit(f'unknown scenario {s!r}')

    os.makedirs(args.out_dir, exist_ok=True)

    # results[scenario][estimator] = list of R_relative across seeds
    results = {}
    t_start = time.time()
    for seed in args.seeds:
        orig = _patch_random_state(seed)
        try:
            for sc in scenarios:
                print(f'\n=== seed={seed}  scenario={sc} ===')
                t0 = time.time()
                SCENARIO_FUNCTIONS[sc]()
                print(f'    [{time.time()-t0:.0f}s]')

                stats_src = f'compare_{sc}_stats.json'
                stats_dst = os.path.join(args.out_dir, f'compare_{sc}_seed{seed}_stats.json')
                shutil.copyfile(stats_src, stats_dst)

                with open(stats_src) as f:
                    d = json.load(f)
                results.setdefault(sc, {})
                for est in d['estimators']:
                    results[sc].setdefault(est['name'], []).append(est['R_relative'])
        finally:
            _restore(*orig)

    # Aggregate
    summary = {'seeds': args.seeds, 'scenarios': {}}
    for sc, ests in results.items():
        summary['scenarios'][sc] = {}
        for nm, vals in ests.items():
            arr = np.asarray(vals, dtype=float)
            summary['scenarios'][sc][nm] = {
                'mean': float(np.mean(arr)),
                'std':  float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
                'n':    int(arr.size),
                'values': vals,
            }

    with open('multi_seed_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Markdown table: rows scenario, cols estimator, cells "mean +/- std"
    sc_order = scenarios
    est_names = ['analytic_DF', 'scipy_kde', 'cvGaussianKDE', 'cvAdaptive',
                 'cvAdapt-smart', 'cvAdapt-stream']
    lines = []
    lines.append(f'# Multi-seed rate-recovery summary')
    lines.append('')
    lines.append(f'Seeds: {args.seeds}')
    lines.append(f'Total wallclock: {time.time()-t_start:.0f} s')
    lines.append('')
    header = '| scenario | ' + ' | '.join(est_names) + ' |'
    sep = '|' + '|'.join(['---'] * (len(est_names) + 1)) + '|'
    lines.append(header)
    lines.append(sep)
    for sc in sc_order:
        if sc not in summary['scenarios']:
            continue
        row = [sc]
        for nm in est_names:
            ests = summary['scenarios'][sc]
            if nm in ests:
                m, s = ests[nm]['mean'], ests[nm]['std']
                row.append(f'{m:.2f} +/- {s:.2f}')
            else:
                row.append('-')
        lines.append('| ' + ' | '.join(row) + ' |')
    md = '\n'.join(lines)
    with open('multi_seed_summary.md', 'w') as f:
        f.write(md + '\n')
    print('\n' + md)


if __name__ == '__main__':
    main()
