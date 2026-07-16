"""Backfill the new reliability + recommendation diagnostics into
existing `*_rateresults.pik` files, without re-running rate_sphere_
importance.

For each (foo.csv, foo.pickle, foo_rateresults.pik) triple, this script:

  1. Reconstructs the `estimators` dict from `foo.pickle`
     (jres.kdeScipy, jres.kde.kde_rate / .kde_sky / .kde_vinf, jres.kdeNF).
  2. Pulls `samples_by_estimator` from the existing rateresults file
     (the resj_/vsphere_/costhetasphere_/phisphere_ arrays already there).
  3. Calls `cracked.rate_sphere.reliability_report` to compute
     data_neff_<label> and the NF rate-ensemble stats.
  4. Loads xv from foo.csv and calls
     `cracked.recommend.recommend_for_isostream` for the four estimation
     tasks; saves the recommended method + structural class + the local
     and global triplet diagnostics.
  5. Writes the merged result back to foo_rateresults.pik.

Usage:
    ipython3 scripts/backfill_rateresults_diagnostics.py PATH [PATH ...]
    ipython3 scripts/backfill_rateresults_diagnostics.py 'roinone_prod/*_rateresults.pik'

Add --dry-run to see which keys would be added without writing.
Add --force to overwrite existing diagnostic keys (default is to skip
files whose diagnostics are already present).
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
import traceback
from typing import Any

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, 'src'))

# Patch torch's CUDA-restore-location BEFORE we load any jres pickle
# that may contain a CUDA-saved NF.
try:
    from cracked.normalizing_flow import patch_torch_load_to_cpu
    patch_torch_load_to_cpu()
except ImportError:
    pass

from cracked.rate_sphere import reliability_report
from cracked.recommend import recommend_for_isostream


_NEW_KEYS = (
    'data_neff_scipy', 'data_neff_cv_rate', 'data_neff_cv_sky',
    'data_neff_cv_vinf', 'data_neff_nf',
    'nf_rate_per_flow', 'nf_rate_ensemble_mean',
    'nf_rate_ensemble_std', 'nf_rate_ensemble_cv',
    'recommended_method_rate', 'recommended_class_rate',
    'recommend_n_near', 'recommend_coherence_min',
    'recommend_multi_scale_min', 'recommend_local_cond_med',
)


def _build_estimators_from_jres(jres) -> dict[str, tuple[Any, int]]:
    """Reconstruct the production estimators dict from a julia_result."""
    estimators: dict[str, tuple[Any, int]] = {}
    if getattr(jres, 'kdeScipy', None) is not None:
        estimators['scipy'] = (jres.kdeScipy, 14)
    cvkde_full = getattr(jres, 'kde', None)
    if cvkde_full is not None:
        if hasattr(cvkde_full, 'kde_rate'):
            estimators['cv_rate'] = (cvkde_full.kde_rate, 15)
        if hasattr(cvkde_full, 'kde_sky'):
            estimators['cv_sky'] = (cvkde_full.kde_sky, 16)
        if hasattr(cvkde_full, 'kde_vinf'):
            estimators['cv_vinf'] = (cvkde_full.kde_vinf, 17)
    if getattr(jres, 'kdeNF', None) is not None:
        estimators['nf'] = (jres.kdeNF, 18)
    return estimators


def _build_samples_from_rr(rr: dict, labels: list[str]) -> dict[str, tuple]:
    """Reconstruct samples_by_estimator from an existing rateresults dict."""
    out: dict[str, tuple] = {}
    for lbl in labels:
        try:
            out[lbl] = (
                rr[f'resj_{lbl}'],
                rr[f'vsphere_{lbl}'],
                rr[f'costhetasphere_{lbl}'],
                rr[f'phisphere_{lbl}'],
                rr.get(f'eccentricities_{lbl}', np.zeros_like(rr[f'resj_{lbl}'])),
                rr.get(f'thetacs_{lbl}', np.zeros_like(rr[f'resj_{lbl}'])),
                rr.get(f'vinftys_{lbl}', np.zeros_like(rr[f'resj_{lbl}'])),
            )
        except KeyError as e:
            print(f"    [skip {lbl}] missing key {e}")
    return out


def _already_done(rr: dict) -> bool:
    return all(k in rr for k in ('data_neff_scipy', 'recommended_class_rate'))


def backfill_one(rr_path: str, *, dry_run: bool = False, force: bool = False) -> str:
    base = rr_path.replace('_rateresults.pik', '')
    jres_path = base + '.pickle'
    csv_path = base + '.csv'

    if not os.path.isfile(rr_path):
        return f"  MISSING rateresults: {rr_path}"
    if not os.path.isfile(jres_path):
        return f"  MISSING jres pickle: {jres_path}"
    if not os.path.isfile(csv_path):
        return f"  MISSING csv: {csv_path}"

    with open(rr_path, 'rb') as f:
        rr = pickle.load(f)
    if _already_done(rr) and not force:
        return f"  SKIP (already has new keys): {os.path.basename(rr_path)}"

    # Lazy import - only when we know we need it.
    from isostreams_prod import julia_result  # type: ignore
    jres = julia_result.load(jres_path)
    estimators = _build_estimators_from_jres(jres)
    if not estimators:
        return f"  NO estimators on jres: {jres_path}"

    samples = _build_samples_from_rr(rr, list(estimators.keys()))
    xsunj = np.asarray(rr['xsunj'])
    vsunj = np.asarray(rr['vsunj'])
    encounter_r = float(rr.get('encounter_r_pc', 1.0))

    new_keys: dict[str, Any] = {}

    # Reliability report
    rel = reliability_report(estimators, samples, xsun=xsunj, vsun=vsunj, encounter_r=encounter_r, nf_label='nf', verbose=True)
    new_keys.update(rel)

    # Pre-training recommendation - needs the training data from CSV.
    try:
        xvj = np.loadtxt(csv_path, delimiter=',')
        if xvj.ndim == 1:
            xvj = xvj.reshape(1, -1)
        if xvj.shape[1] > 6:
            xvj = xvj[:, :6]
        recs = recommend_for_isostream(xvj[:, :6], xloc=xsunj, vloc=vsunj, tasks=('rate', 'sky_map', '1d_marginal', 'density'))
        for task, rec in recs.items():
            new_keys[f'recommended_method_{task}'] = rec['method']
            new_keys[f'recommended_class_{task}']  = rec['structural_class']
        local = recs['rate']['diagnostics']['local']
        if local is not None:
            for k in ('n_near', 'coherence_min', 'multi_scale_min'):
                new_keys[f'recommend_{k}'] = local[k]
        glob_ = recs['rate']['diagnostics']['global_']
        for k in ('local_trace_ratio', 'local_cond_med',
                   'cov_cond_number', 'narrow_feature_flag',
                   'multi_scale_flag'):
            if k in glob_:
                new_keys[f'recommend_{k}'] = glob_[k]
    except Exception as e:
        print(f"    recommend_for_isostream failed: "
              f"{type(e).__name__}: {e}")

    if dry_run:
        added = [k for k in new_keys if k not in rr or force]
        return (f"  DRY-RUN {os.path.basename(rr_path)}: "
                f"would add {len(added)} keys")

    if force:
        rr.update(new_keys)
    else:
        for k, v in new_keys.items():
            rr.setdefault(k, v)

    with open(rr_path, 'wb') as f:
        pickle.dump(rr, f)
    return (f"  OK {os.path.basename(rr_path)}: "
            f"+{len(new_keys)} keys")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('paths', nargs='+', help='*_rateresults.pik files (globs OK)')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--force', action='store_true', help='overwrite existing diagnostic keys')
    args = ap.parse_args()

    files: list[str] = []
    for p in args.paths:
        if any(c in p for c in '*?['):
            files.extend(sorted(glob.glob(p)))
        else:
            files.append(p)
    if not files:
        print("no files matched")
        return 1

    # Make `isostreams_prod` importable for julia_result. It typically
    # lives in the directory the user ran the script from, not next to
    # this script.
    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, os.path.join(HERE, os.pardir))

    for path in files:
        try:
            print(backfill_one(path, dry_run=args.dry_run, force=args.force))
        except Exception as e:
            print(f"  FAIL {os.path.basename(path)}: "
                  f"{type(e).__name__}: {e}")
            traceback.print_exc()
    return 0


if __name__ == '__main__':
    sys.exit(main())
