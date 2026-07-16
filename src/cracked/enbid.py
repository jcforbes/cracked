"""Thin wrapper around the Sharma & Steinmetz 2006 EnBiD code.

EnBiD is the canonical adaptive 6D KDE in galactic-dynamics work (used by
Galaxia, Ananke, Aurigaia, and the original DM phase-space studies). It's
written in C++ with an ASCII / Gadget binary I/O. This module shells out
to the compiled `Enbid` binary so it can be slotted into cracked's
`rate_sphere_importance` callable interface.

Use:
    from cracked.enbid import enbidKDE
    kde = enbidKDE(coords, ngb=64)         # build on training data
    rho_at_q = kde(query_points)            # evaluate density at queries

Implementation strategy: EnBiD outputs density only at the input
particles, with adaptive bandwidths derived from local kNN distances.
There's no native "evaluate at query points" interface. The naive
workaround - concatenate queries with training and slice out their
densities - fails catastrophically when queries are densely packed
(e.g. on a tight encounter sphere): each query's kNN are *other queries*,
inflating the local density estimate by orders of magnitude (12000x on
empirical test).

Instead we run EnBiD on training only, then for each query return the
density of its nearest training particle in EnBiD's auto-rescaled
metric. For smooth densities this is a faithful approximation of what
EnBiD "would say" at the query. Failure mode: queries far outside the
training support get whatever the nearest training particle had -
slightly biased near the boundary, fine in the interior.

Hyperparameters exposed (all defaults match the EnBiD distribution's
sample parameterfile4, which is the canonical "kernel + adaptive metric"
recipe):
    ngb=64               - DesNumNgb. 64 is py-EnBiD-ananke's default;
                           Sharma & Steinmetz 2006 used 10. Both are
                           documented literature defaults.
    kernel_type=3        - 0:B-Spline 1:top-hat 2:Bi-weight 3:Epanechikov
                           4:CIC 5:TSC. Epanechikov is the EnBiD default.
    anisotropic=True     - 0:isotropic 1:adaptive-metric. The "AM" option
                           in TypeOfSmoothing=3 corresponds to anisotropic.
    type_of_smoothing=3  - 0:None 1:FiEstAS 2:KerSpNormal 3:KerSpAM
                           4:KerPrNormal 5:KerPrAM. 3 is the kernel-with-
                           adaptive-metric mode the Sharma paper uses.
    spatial_scale=-1.0   - when negative, EnBiD auto-rescales each axis by
                           empirical std (per init.cpp). This is REQUIRED
                           when position and velocity scales differ - at
                           spatial_scale=1.0 with positions in [-5,5] and
                           velocities O(1), EnBiD segfaults in tree build.
                           Use a positive value only if all axes are
                           already on comparable scale (e.g. 6D unit data).
    binary_path          - defaults to <repo_root>/external/Enbid-2.0/Enbid.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def _default_binary_path() -> Path:
    """Locate the compiled Enbid binary in the repo's external/ directory."""
    here = Path(__file__).resolve()
    # cracked source lives at <repo>/src/cracked/enbid.py
    repo_root = here.parent.parent.parent
    candidate = repo_root / "external" / "Enbid-2.0" / "Enbid"
    return candidate


def _build_paramfile(input_path: Path, snapshot_base: str, ngb: int, kernel_type: int, anisotropic: int, type_of_smoothing: int, spatial_scale: float) -> str:
    """EnBiD parameter file text. Mirrors the canonical
    `parameterfiles/myparameterfile4` from the EnBiD distribution."""
    return f"""%  Input and Output
InitCondFile     {input_path}

ICFormat                  0     % O)ASCII 1)Gadget 2)User defined
SnapshotFileBase        {snapshot_base}

SpatialScale            {spatial_scale}
PartBoundary            7
NodeSplittingCriterion  1
CubicCells              1
MedianSplittingOn       0

TypeOfSmoothing      {type_of_smoothing}
DesNumNgb            {ngb}
VolCorr              1

TypeOfKernel           {kernel_type}
KernelBiasCorrection   1
AnisotropicKernel      {anisotropic}
Anisotropy             0
DesNumNgbA             {min(2*ngb, 128)}

TypeListOn        0
PeriodicBoundaryOn 0
% --- EnBiD's paramfile parser re-reads the last line at EOF; the
% --- trailing comment lines absorb the duplicate read without erroring.
"""


class enbidKDE:
    """EnBiD-backed 6D adaptive KDE, callable as a cracked density estimator."""

    def __init__(self, data: np.ndarray, ngb: int = 64, kernel_type: int = 3, anisotropic: bool = True, type_of_smoothing: int = 3, spatial_scale: float = -1.0, binary_path: os.PathLike | None = None):
        self.data = np.ascontiguousarray(data, dtype=float)
        if self.data.ndim != 2 or self.data.shape[1] != 6:
            raise ValueError(f"enbidKDE expects 6D data (N, 6); got shape {self.data.shape}")
        self.ngb = int(ngb)
        self.kernel_type = int(kernel_type)
        self.anisotropic = int(bool(anisotropic))
        self.type_of_smoothing = int(type_of_smoothing)
        self.spatial_scale = float(spatial_scale)
        self.binary_path = Path(binary_path) if binary_path is not None \
                           else _default_binary_path()
        if not self.binary_path.exists():
            raise FileNotFoundError(
                f"Enbid binary not found at {self.binary_path}. Compile from "
                f"external/Enbid-2.0/src via `make` (default DIM6 build).")
        # Match adaptiveKDE's covfac_overall=1.0 convention so external code
        # can pass `covfac` without crashing - we ignore it.
        self.covfac_overall = 1.0
        # Run EnBiD once on training only; cache per-point densities and a
        # KDTree for query-side NN lookup. The scaling for NN lookup mirrors
        # EnBiD's own auto-rescale: per-axis std, normalised by max axis.
        self._train_densities = self._run(self.data)
        from scipy.spatial import cKDTree
        std = self.data.std(axis=0)
        std = np.where(std > 0, std, 1.0)
        self._nn_scales = std / std.max()
        self._tree = cKDTree(self.data / self._nn_scales)

    def _run(self, combined: np.ndarray) -> np.ndarray:
        """Write input, run binary, read density column."""
        with tempfile.TemporaryDirectory(prefix="enbid_") as tmpdir:
            tmpdir = Path(tmpdir)
            input_path = tmpdir / "input.ascii"
            np.savetxt(input_path, combined, fmt="%.8e")
            param_path = tmpdir / "param"
            snapshot_base = "_run"
            param_path.write_text(_build_paramfile(input_path, snapshot_base, ngb=self.ngb, kernel_type=self.kernel_type, anisotropic=self.anisotropic, type_of_smoothing=self.type_of_smoothing, spatial_scale=self.spatial_scale))
            try:
                subprocess.run([str(self.binary_path), str(param_path)], cwd=tmpdir, check=True, capture_output=True, timeout=600)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"Enbid binary failed (exit {e.returncode}). "
                    f"stderr: {e.stderr.decode(errors='replace')[:2000]}")
            est_file = tmpdir / f"input.ascii{snapshot_base}.est"
            if not est_file.exists():
                # Older EnBiD distros may use a different naming convention.
                est_files = list(tmpdir.glob("*.est"))
                if not est_files:
                    raise RuntimeError(f"No .est output found in {tmpdir}")
                est_file = est_files[0]
            return np.loadtxt(est_file)

    def __call__(self, points: np.ndarray, covfac: float = 1.0, covalpha: float = 0.0, returnLog: bool = False, show_contribs: bool = False) -> np.ndarray:
        """Density at `points` (N, 6). The `covfac`/`covalpha`/`show_contribs`
        kwargs are accepted (for interface compatibility) and ignored -
        EnBiD has no analogue.

        Strategy: nearest-training-particle lookup in EnBiD's auto-rescaled
        metric. EnBiD's per-particle density (computed on training only)
        is returned for the nearest training particle to each query."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        if points.ndim != 2 or points.shape[1] != 6:
            raise ValueError(f"expected (M, 6) eval points; got {points.shape}")
        scaled = points / self._nn_scales
        _, nn_idx = self._tree.query(scaled, k=1)
        rho = self._train_densities[nn_idx]
        # EnBiD's output is particle-count density (int rho dV = N_train), not a
        # probability density. Convert by /N so the result matches the
        # cracked KDE convention (int rho dV = 1).
        rho = rho / float(self.data.shape[0])
        if returnLog:
            return np.log(np.maximum(rho, 1.0e-300))
        return rho

    # Stubs to mirror adaptiveKDE's interface where it makes sense.
    @property
    def covalpha_overall(self) -> float:
        return 0.0
