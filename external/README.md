# External dependencies (not packaged)

External codes used as comparison baselines in the test suite. Not bundled
with `cracked` itself — fetch and build locally.

## EnBiD (Sharma & Steinmetz 2006)

The canonical adaptive 6D KDE in galactic-dynamics work. Used by the
plot-comparison tests via `cracked.enbid.enbidKDE`.

### Install

```bash
cd external
curl -L -o enbid-2.0.tar.gz "https://sourceforge.net/projects/enbid/files/latest/download"
tar -xzf enbid-2.0.tar.gz
cd Enbid-2.0/src
# On macOS, point at the Command Line Tools SDK so <stdio.h> resolves:
SDKROOT=$(xcrun --show-sdk-path) make
# Binary lands at external/Enbid-2.0/Enbid
```

The compiled `Enbid` binary is found automatically by `cracked.enbid.enbidKDE`
via the path `<repo_root>/external/Enbid-2.0/Enbid`. If it's missing the
EnBiD column in the test suite is silently skipped.

### Default config

The wrapper sets the canonical "kernel + adaptive metric" recipe from
EnBiD's `parameterfiles/myparameterfile4`:
  - `DesNumNgb=64` (py-EnBiD-ananke / Galaxia default; Sharma 2006 used 10)
  - Epanechikov kernel
  - Anisotropic adaptive metric
  - Type-of-smoothing 3 (Ker Sp AM)

All hyperparameters are exposed on the `enbidKDE` constructor.

### Paper

Sharma & Steinmetz (2006), MNRAS 373:1293 — `https://academic.oup.com/mnras/article/373/4/1293/3101445`.
ASCL entry: `https://www.ascl.net/1109.012`. Source on SourceForge.

## Normalizing flows (planned)

Modern ML alternative for 6D phase-space density estimation; see e.g.
Buckley et al. 2022 (arXiv:2205.01129) and Mapping Dark Matter with
Normalizing Flows and Gaia DR3 (arXiv:2305.13358). Not implemented yet —
see `MEMORY.md` for the follow-up note.
