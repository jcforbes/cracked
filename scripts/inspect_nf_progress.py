"""Poll the production directory for NF-training progress.

Counts .pickle caches that have a NormalizingFlowKDE attached and
reports their ensemble sizes. Also flags any with a stale 6D!=data
dimensionality (e.g. 9D OU-augmented training data - these should be
discarded by the updated isostreams_prod.py block).

Usage:
    ipython3 scripts/inspect_nf_progress.py            # current dir
    ipython3 scripts/inspect_nf_progress.py /path/to/dir
"""
import os
import sys
import glob

import numpy as np


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else '.'
    fns = [f for f in glob.glob(os.path.join(target, '*.pickle'))
           if not f.endswith('rateresults.pik')]
    fns.sort()

    if not fns:
        print(f"no .pickle caches found in {target}")
        return

    import dill
    total = len(fns)
    with_kde = with_nf = 0
    nf_sizes = []
    nf_dims = []
    bad_dim = []
    for fn in fns:
        try:
            j = dill.load(open(fn, 'rb'))
        except Exception:
            continue
        if getattr(j, 'kde', None) is not None:
            with_kde += 1
        nf = getattr(j, 'kdeNF', None)
        if nf is None:
            continue
        if type(nf).__name__ != 'NormalizingFlowKDE':
            continue
        with_nf += 1
        nf_sizes.append(len(getattr(nf, 'flows', [])))
        d = int(getattr(nf, '_mean', np.zeros(0)).size)
        nf_dims.append(d)
        if d != 6:
            bad_dim.append((os.path.basename(fn), d))

    print(f"{total} .pickle caches in {target}")
    print(f"  with CV kde:  {with_kde}")
    print(f"  with NF:      {with_nf}")
    if nf_sizes:
        print(f"  NF ensemble sizes: "
              f"min={min(nf_sizes)} median={int(np.median(nf_sizes))} "
              f"max={max(nf_sizes)}")
        print(f"  NF dims:           "
              f"min={min(nf_dims)} median={int(np.median(nf_dims))} "
              f"max={max(nf_dims)}")
    if bad_dim:
        print(f"\n  WARNING: {len(bad_dim)} NFs trained on the wrong "
              f"dimensionality (not 6D):")
        for fn, d in bad_dim[:10]:
            print(f"    {fn[:75]:75s}  dim={d}")
        if len(bad_dim) > 10:
            print(f"    ... and {len(bad_dim) - 10} more")


if __name__ == "__main__":
    main()
