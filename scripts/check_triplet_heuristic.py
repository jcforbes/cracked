"""Verify cracked.recommend.recommend_method against the regime map's
empirical winners across all four estimation tasks.

Expected mapping (regime map's rows 0-3):
  scenario      | rate              | sky_map  | 1d_marginal     | density
  smooth iso    | NF                | NF       | NF              | NF
  anisotropic   | NF                | NF       | NF              | NF
  cold+hot      | cvAdapt-narrow    | scipy    | cvAdapt-narrow  | NF
  disk_stream   | cvAdaptive        | scipy    | NF              | NF
  spiky_ball    | cvAdaptive        | scipy    | NF              | cvAdaptive
"""

import os
import sys
import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, os.pardir, "tests"))
sys.path.insert(0, os.path.join(HERE, os.pardir, "src"))

from test_rate_sphere_analytic import (
    make_isotropic_sampler, make_ring_sampler, make_cold_hot_sampler,
    make_disk_stream_sampler, make_spiky_ball_sampler,
)
from cracked.recommend import recommend_method


SCENARIOS = [
    ('smooth isotropic',     make_isotropic_sampler(1.0)),
    ('anisotropic ring',     make_ring_sampler(0.5, 0.05, 0.05)),
    ('multi-scale cold+hot', make_cold_hot_sampler(0.4, 10.0, 0.2)),
    ('narrow disk stream',   make_disk_stream_sampler(R_ring=8000.0, sigma_R=1.0, sigma_z=1.0, sigma_t=0.1, v_circ=220.0, width=25.0, height=12.5, v_sun_peculiar=(-5.0, 5.0, 0.0))),
    ('many narrow spiky',    make_spiky_ball_sampler(25, 100.0, 0.3, 0.3, 0.05, 3.0)[0]),
]

# Empirical winners per (scenario, task) - taken directly from the
# regime map.  Methods are normalised to cracked class names.
EXPECTED = {
    ('smooth isotropic', 'rate'):        'NormalizingFlowKDE',
    ('smooth isotropic', 'sky_map'):     'NormalizingFlowKDE',
    ('smooth isotropic', '1d_marginal'): 'NormalizingFlowKDE',
    ('smooth isotropic', 'density'):     'NormalizingFlowKDE',
    ('anisotropic ring', 'rate'):        'NormalizingFlowKDE',
    ('anisotropic ring', 'sky_map'):     'NormalizingFlowKDE',
    ('anisotropic ring', '1d_marginal'): 'NormalizingFlowKDE',
    ('anisotropic ring', 'density'):     'NormalizingFlowKDE',
    ('multi-scale cold+hot', 'rate'):        'cvAdaptiveKDE',
    ('multi-scale cold+hot', 'sky_map'):     'gaussianKDEWrapper',
    ('multi-scale cold+hot', '1d_marginal'): 'cvAdaptiveKDE',
    ('multi-scale cold+hot', 'density'):     'NormalizingFlowKDE',
    ('narrow disk stream', 'rate'):          'cvAdaptiveKDE',
    ('narrow disk stream', 'sky_map'):       'gaussianKDEWrapper',
    ('narrow disk stream', '1d_marginal'):   'NormalizingFlowKDE',
    ('narrow disk stream', 'density'):       'NormalizingFlowKDE',
    ('many narrow spiky', 'rate'):           'cvAdaptiveKDE',
    ('many narrow spiky', 'sky_map'):        'gaussianKDEWrapper',
    ('many narrow spiky', '1d_marginal'):    'NormalizingFlowKDE',
    ('many narrow spiky', 'density'):        'cvAdaptiveKDE',
}

TASKS = ('rate', 'sky_map', '1d_marginal', 'density')


def main():
    rng = np.random.default_rng(11)
    xloc = np.array([0.0, 0.0, 0.0])

    # Sample each scenario once and re-use for all four tasks.
    samples = []
    for name, sampler in SCENARIOS:
        samples.append((name, sampler(rng, 2000)))

    print("recommend_method vs regime-map empirical winners (N=2000)\n")
    print(f"{'scenario':<22} {'class':<18} "
          f"{'task':<14} {'predicted':<22} {'expected':<22} {'ok'}")
    print("-" * 110)

    correct = 0
    total = 0
    for name, coords in samples:
        for task in TASKS:
            rec = recommend_method(coords, task=task, xloc=xloc, verbose=False)
            pred = rec['method']
            expected = EXPECTED[(name, task)]
            ok = pred == expected
            correct += int(ok)
            total += 1
            mark = 'OK' if ok else '  x'
            print(f"{name:<22} {rec['structural_class']:<18} "
                  f"{task:<14} {pred:<22} {expected:<22} {mark}")

    print(f"\n{correct}/{total} cells correct")
    return correct, total


if __name__ == "__main__":
    main()
