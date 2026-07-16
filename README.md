# cracked

**C**ross-validated **R**ate-of-encounter **A**daptive **C**ovariance-based
**K**ernel **E**stimator of **D**ensity.

A 6D-phase-space adaptive KDE specialised for stream->target encounter-rate
calculations. Per-point Gaussian kernels are built from local sample
covariance (kNN) and rescaled by CV-selected (covfac, covalpha, shrinkage,
scaling) hyperparameters. Includes uniform-sphere and importance-sampled
encounter-rate evaluators with gravitational-focusing cross-sections.

## Install

```bash
pip install -e .
# with test deps:
pip install -e ".[tests]"
```

## Quickstart

```python
import numpy as np
from cracked import (cvAdaptiveKDE, rate_sphere_importance,
                     make_data_driven_is_proposal)

# 6D phase-space data in (pc, pc/Myr) - galactic frame.
# coords.shape == (N, 6) with columns [x, y, z, vx, vy, vz]
coords = ...

# Cross-validated adaptive KDE.
cvkde = cvAdaptiveKDE(coords, nfolds=5, scalings_grid=["auto", "narrow", "narrow_local"], shrinkage_target="local_pooled")

# Importance-sampled encounter rate at the Sun.
xsun = (0.0, 0.0, 0.0)
vsun = (0.0, 0.0, 0.0)
v_mean, v_cov = make_data_driven_is_proposal(coords, xloc=xsun)
out = rate_sphere_importance(cvkde.kde_rate, v_mean, v_cov, xloc=xsun, vloc=vsun, Nboot=100_000, qmaxAU=5.0)
weights = out[0]
rate_per_year = weights.sum() / len(weights)
```

## Tests

```bash
pytest                          # quick tests
pytest tests/test_adaptive_kde.py   # KDE unit/regression tests
pytest tests/test_rate_sphere_analytic.py  # rate validation vs analytic Safronov
pytest tests/test_convergence.py    # convergence diagnostics (slow)
```
