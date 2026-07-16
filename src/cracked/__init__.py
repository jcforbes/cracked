"""CRACKED - Cross-validated Rate-of-encounter Adaptive Covariance-based
Kernel Estimator of Density.

A 6D-phase-space adaptive KDE specialised for stream->target encounter-rate
calculations. Per-point Gaussian kernels are built from local sample
covariance (kNN) and rescaled by CV-selected (covfac, covalpha, shrinkage,
scaling) hyperparameters.

Public API:
  adaptiveKDE          - per-point adaptive KDE on 6D data
  cvAdaptiveKDE        - CV-selected hyperparameters around adaptiveKDE
  cvGaussianKDE        - CV-selected bandwidth + per-axis scaling around scipy
  gaussianKDEWrapper   - thin scipy.stats.gaussian_kde wrapper
  mockScipyKde         - adaptiveKDE forced to scipy-style global covariance

  RATE_Sphere                  - uniform-sphere MC encounter-rate estimator
  rate_sphere_importance       - same, IS on velocity (preferred for narrow features)
  make_data_driven_is_proposal - Gaussian IS proposal from a position-local subset

  G, pcperau           - physical constants (pc/Msun/Myr; pc per AU)
"""
from .kde import (
    adaptiveKDE,
    cvAdaptiveKDE,
    cvGaussianKDE,
    gaussianKDEWrapper,
    mockScipyKde,
)
from .rate_sphere import (
    RATE_Sphere,
    rate_sphere_importance,
    make_data_driven_is_proposal,
    make_production_cv_kde,
    G,
    pcperau,
)

__all__ = [
    "adaptiveKDE",
    "cvAdaptiveKDE",
    "cvGaussianKDE",
    "gaussianKDEWrapper",
    "mockScipyKde",
    "RATE_Sphere",
    "rate_sphere_importance",
    "make_data_driven_is_proposal",
    "make_production_cv_kde",
    "G",
    "pcperau",
]
__version__ = "0.1.0"
