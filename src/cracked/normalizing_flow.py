"""Normalizing-flow density estimator for 6D phase-space data.

Modeled after Buckley et al. 2022 (arXiv:2205.01129), which fit a Masked
Autoregressive Flow (MAF) to Gaia DR3 6D data to look for dark-matter
substructure. The intuition that motivates NF here:
the data is essentially a sample from a base distribution (close to
Gaussian) that's been transformed by N-body integration. A learned
invertible transform from base -> data exactly captures that mapping.

Wraps `nflows.flows.MaskedAutoregressiveFlow` plus a per-axis standardiser.
Matches cracked's KDE callable interface (`kde(points)` -> density).

The transform is bijective so log-density at query points is *exact*
modulo training; no kernel-bandwidth ambiguity. Cost: training takes
30s-5min depending on N and architecture; query is one forward pass.

Two transform families:
  - transform='affine' (default): Buckley et al. 2022 recipe, affine MAF.
  - transform='spline': piecewise rational-quadratic autoregressive layers
    (Neural Spline Flow; Durkan et al. 2019, arXiv:1906.04032). Same masked
    MADE conditioner, but each 1D transform is a monotone RQ spline with
    `num_bins` knots inside [-tail_bound, tail_bound] (linear tails outside).
    Far more flexible per layer than affine - the candidate fix for the
    affine MAF's failures on narrow spatial features (disk stream, spiky
    ball), where the stiff affine layers can't carve sharp ridges.

Use:
    kde = NormalizingFlowKDE(coords, n_layers=5, hidden_units=64)
    rho = kde(query_points)
    log_rho = kde(query_points, returnLog=True)
    nsf = NormalizingFlowKDE(coords, transform='spline')
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np


def _import_nf():
    """Lazy import so the package doesn't hard-require torch/nflows."""
    try:
        import torch
        from nflows.flows import MaskedAutoregressiveFlow
    except ImportError as e:
        raise ImportError(
            f"NormalizingFlowKDE requires torch + nflows; install via\n"
            f"    pip install torch nflows\n"
            f"Original error: {e}")
    return torch, MaskedAutoregressiveFlow


def _build_spline_flow(features, hidden_features, num_layers, num_blocks_per_layer, use_residual_blocks, activation_fn, num_bins, tail_bound):
    """Neural Spline Flow: mirror of nflows' MaskedAutoregressiveFlow
    construction (ReversePermutation + masked autoregressive layer per
    block), with the affine transform swapped for a piecewise
    rational-quadratic spline (Durkan et al. 2019).

    `tails='linear'` makes the transform identity-with-slope outside
    [-tail_bound, tail_bound] in the standardized frame, so density
    evaluation stays finite for query points beyond the training range -
    important for rate evaluation, which probes low-density tails.
    """
    from nflows.distributions.normal import StandardNormal
    from nflows.flows.base import Flow
    from nflows.transforms.base import CompositeTransform
    from nflows.transforms.permutations import ReversePermutation
    from nflows.transforms.autoregressive import (
        MaskedPiecewiseRationalQuadraticAutoregressiveTransform)

    layers = []
    for _ in range(num_layers):
        layers.append(ReversePermutation(features=features))
        layers.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(features=features, hidden_features=hidden_features, num_bins=num_bins, tails='linear', tail_bound=tail_bound, num_blocks=num_blocks_per_layer, use_residual_blocks=use_residual_blocks, random_mask=False, activation=activation_fn, dropout_probability=0.0, use_batch_norm=False))
    return Flow(transform=CompositeTransform(layers),
                distribution=StandardNormal([features]))


def patch_torch_load_to_cpu():
    """Globally force CUDA tensors in pickled objects to deserialize on CPU.

    Use case: a `julia_result.pickle` cached on a GPU node carries
    NormalizingFlowKDE instances whose torch tensors were saved with
    device='cuda'. Loading via `dill.load(open(fn, 'rb'))` on a CPU-only
    node then fails with `RuntimeError: Attempting to deserialize object
    on a CUDA device but torch.cuda.is_available() is False`. The
    dill->torch internal path calls `torch.load` without an explicit
    `map_location`, so the only intervention point is the torch module's
    default-restore-location.

    Call this ONCE at process startup (e.g. right after importing torch
    in isostreams_prod.py, before any `julia_result.load`). Pickles that
    were already CPU-only pass through unchanged. Future saves via the
    NormalizingFlowKDE.__getstate__ machinery (which moves flows to CPU
    before pickling) don't need this patch.

    Idempotent - safe to call multiple times.
    """
    import torch.serialization as _ts

    def _cpu_restore_location(storage, location):
        return storage.cpu()

    _ts.default_restore_location = _cpu_restore_location


class NormalizingFlowKDE:
    """Masked Autoregressive Flow density estimator for 6D phase-space data."""

    def __init__(
        self,
        data: np.ndarray,
        *,
        # Defaults track Buckley et al. 2022 Appendix A: 5-layer affine MAF
        # with 48-unit residual MADE blocks, GELU activations, two-stage Adam
        # training (lr=1e-3 -> 1e-4 fine-tune), 80/20 split, 50-epoch patience.
        transform: str = "affine",   # 'affine' (Buckley MAF) or 'spline' (NSF)
        num_bins: int = 8,           # spline only: RQ-spline knots per axis
        tail_bound: float = 4.0,     # spline only: spline support, standardized units
        n_layers: int = 5,
        hidden_units: int = 48,
        num_blocks_per_layer: int = 2,
        use_residual_blocks: bool = True,
        activation: str = "gelu",   # 'gelu' (Buckley et al.) or 'relu'
        n_epochs: int = 500,
        batch_size: Optional[int] = None,   # None -> 1/10 of training set
        lr: float = 1.0e-3,
        lr_fine_tune: float = 1.0e-4,
        val_frac: float = 0.2,
        early_stop_patience: int = 50,
        ensemble_size: int = 10,   # Buckley et al. average 5-10; default 10 for stability
        device: Optional[str] = None,
        random_state: Optional[int] = None,
        verbose: bool = False,
        weights: Optional[np.ndarray] = None,
    ):
        torch, MaskedAutoregressiveFlow = _import_nf()
        self._torch = torch
        if transform not in ("affine", "spline"):
            raise ValueError(f"transform must be 'affine' or 'spline'; "
                             f"got {transform!r}")
        self.transform_type = transform
        rng = np.random.default_rng(random_state)

        self.data = np.ascontiguousarray(data, dtype=np.float64)
        if self.data.ndim != 2:
            raise ValueError(f"expected 2D data (N, d); got {self.data.shape}")
        N, d = self.data.shape

        # Standardize per axis so positions (~kpc) and velocities (~km/s)
        # are on the same scale - essential for NF training.
        self._mean = self.data.mean(axis=0)
        self._std = self.data.std(axis=0)
        self._std = np.where(self._std > 0, self._std, 1.0)
        scaled = (self.data - self._mean) / self._std

        # Per-training-point input weights (e.g. Gaia ISO population weights).
        # Used as a WEIGHTED maximum-likelihood objective: the NLL loss becomes
        # Sigma_i w_i*(-log p(x_i)) / Sigma_i w_i instead of the plain mean, so the flow
        # fits the weighted distribution natively (no resampling noise). None ->
        # uniform (unchanged). Normalised to sum N for numerical comparability.
        if weights is None:
            w_full = np.ones(N)
        else:
            w_full = np.asarray(weights, dtype=float).reshape(-1)
            if w_full.shape != (N,):
                raise ValueError(f"weights shape {w_full.shape} != ({N},)")
            if np.any(w_full < 0) or not np.all(np.isfinite(w_full)):
                raise ValueError("weights must be non-negative and finite")
            w_full = w_full / w_full.sum() * N
        self._input_weights = w_full

        # Train / val split.
        idx = rng.permutation(N)
        n_val = max(int(round(val_frac * N)), 1)
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        x_train = torch.from_numpy(scaled[train_idx]).float()
        x_val = torch.from_numpy(scaled[val_idx]).float()
        w_train = torch.from_numpy(w_full[train_idx]).float()
        w_val = torch.from_numpy(w_full[val_idx]).float()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        activation_map = {"gelu": torch.nn.functional.gelu,
                          "relu": torch.nn.functional.relu,
                          "elu":  torch.nn.functional.elu}
        if activation not in activation_map:
            raise ValueError(f"activation must be one of {list(activation_map)}; "
                             f"got {activation!r}")
        activation_fn = activation_map[activation]

        if batch_size is None:
            batch_size = max(1, len(x_train) // 10)

        x_train = x_train.to(self._device)
        x_val = x_val.to(self._device)
        w_train = w_train.to(self._device)
        w_val = w_val.to(self._device)
        self._weighted = weights is not None

        # Train (optionally) an ensemble of MAFs with different inits, average
        # log_prob at query time. Matches Buckley et al.'s variance-reduction trick.
        flows = []
        best_val_per_ensemble = []
        for k in range(ensemble_size):
            if random_state is not None:
                torch.manual_seed(int(random_state) + k)

            if transform == "spline":
                flow = _build_spline_flow(features=d, hidden_features=hidden_units, num_layers=n_layers, num_blocks_per_layer=num_blocks_per_layer, use_residual_blocks=use_residual_blocks, activation_fn=activation_fn, num_bins=num_bins, tail_bound=tail_bound).to(self._device)
            else:
                flow = MaskedAutoregressiveFlow(features=d, hidden_features=hidden_units, num_layers=n_layers, num_blocks_per_layer=num_blocks_per_layer, use_residual_blocks=use_residual_blocks, use_random_masks=False, use_random_permutations=False, activation=activation_fn, dropout_probability=0.0, batch_norm_within_layers=False, batch_norm_between_layers=False).to(self._device)

            # Two-stage Adam (Buckley et al.): lr=1e-3 -> patience-bounded
            # restart at lr_fine_tune for additional epochs.
            best_val = float("inf")
            best_state = None
            for stage_idx, stage_lr in enumerate([lr, lr_fine_tune]):
                optimizer = torch.optim.Adam(flow.parameters(), lr=stage_lr)
                if best_state is not None:
                    flow.load_state_dict(best_state)
                n_since_improve = 0
                for epoch in range(n_epochs):
                    flow.train()
                    perm = torch.randperm(len(x_train), device=self._device)
                    losses = []
                    for i in range(0, len(x_train), batch_size):
                        bidx = perm[i:i + batch_size]
                        batch = x_train[bidx]
                        if self._weighted:
                            wb = w_train[bidx]
                            # Weighted NLL: Sigma w*(-log p) / Sigma w.
                            loss = -(wb * flow.log_prob(batch)).sum() / wb.sum()
                        else:
                            loss = -flow.log_prob(batch).mean()
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        losses.append(float(loss.item()))
                    flow.eval()
                    with torch.no_grad():
                        if self._weighted:
                            val_loss = float((-(w_val * flow.log_prob(x_val)).sum()
                                              / w_val.sum()).item())
                        else:
                            val_loss = float(-flow.log_prob(x_val).mean().item())
                    if val_loss < best_val - 1e-4:
                        best_val = val_loss
                        best_state = {kk: v.detach().clone()
                                      for kk, v in flow.state_dict().items()}
                        n_since_improve = 0
                    else:
                        n_since_improve += 1
                    if verbose and (epoch % 25 == 0 or epoch == n_epochs - 1):
                        print(f"    ens={k} stage={stage_idx} ep={epoch:4d}: "
                              f"train={np.mean(losses):.4f} val={val_loss:.4f} "
                              f"best={best_val:.4f} pat={n_since_improve}/{early_stop_patience}",
                              flush=True)
                    if n_since_improve >= early_stop_patience:
                        if verbose:
                            print(f"    ens={k} stage={stage_idx} early stop "
                                  f"at epoch {epoch}", flush=True)
                        break

            if best_state is not None:
                flow.load_state_dict(best_state)
            flow.eval()
            flows.append(flow)
            best_val_per_ensemble.append(best_val)
        self.flows = flows
        self.best_val_nll = float(np.mean(best_val_per_ensemble))
        # Back-compat alias for the previous single-flow API.
        self.flow = flows[0]
        # Match adaptiveKDE's covfac_overall=1.0 convention for downstream
        # interop (rate_sphere_importance and friends).
        self.covfac_overall = 1.0

    @staticmethod
    def _resolve_base_cov(base_cov, latent_dispersion, d, n_flows):
        """Normalise a base-covariance spec to a (d,d) array or a
        (n_flows,d,d) stack of per-flow covariances.

        Accepts: None (-> (1+s^2)*I from `latent_dispersion`), scalar
        (-> c*I), 1D length-d (-> diag), (d,d), or (n_flows,d,d)."""
        if base_cov is None:
            var = 1.0 + float(latent_dispersion) ** 2
            return var * np.eye(d)
        bc = np.asarray(base_cov, dtype=float)
        if bc.ndim == 0:
            return float(bc) * np.eye(d)
        if bc.ndim == 1:
            if bc.shape[0] != d:
                raise ValueError(f"base_cov diagonal length {bc.shape[0]} != d={d}")
            return np.diag(bc)
        if bc.ndim == 2:
            if bc.shape != (d, d):
                raise ValueError(f"base_cov shape {bc.shape} != ({d},{d})")
            return bc
        if bc.ndim == 3:
            if bc.shape != (n_flows, d, d):
                raise ValueError(f"per-flow base_cov shape {bc.shape} != ({n_flows},{d},{d})")
            return bc
        raise ValueError(f"base_cov has too many dims: {bc.shape}")

    @staticmethod
    def _mvn_logpdf(z, mean, chol):
        """log N(z; mean, C) for z shape (n,d), lower-Cholesky `chol` of C.
        `mean` is (d,) or None (-> 0)."""
        from scipy.linalg import solve_triangular
        d = z.shape[1]
        y = z if mean is None else z - mean
        sol = solve_triangular(chol, y.T, lower=True)        # (d, n)
        quad = np.einsum("ij,ij->j", sol, sol)               # (n,)
        logdet = 2.0 * np.sum(np.log(np.diag(chol)))
        return -0.5 * quad - 0.5 * d * np.log(2.0 * np.pi) - 0.5 * logdet

    def _per_flow_log_prob(self, x_t, latent_dispersion=0.0, base_mean=None, base_cov=None):
        """Per-ensemble-member log-prob at standardized points `x_t`,
        shape (n_ensemble, n_points).

        The base distribution in latent (whitened) space defaults to the
        flow's own N(0, I). Two generalizations replace it:

        `latent_dispersion` (s >= 0): isotropic widened base N(0, (1+s^2)*I).
        The modeled population is "the training population plus extra
        dispersion s in latent space". Through the local Jacobian of the
        learned map, the added data-space spread is J(z)*J(z)^T*s^2 -
        position-dependent, anisotropic, and aligned with the local
        structure the flow learned (e.g. along a stream, not across it).
        s=0 reproduces the flow's own log_prob exactly.

        `base_mean` (n_flows, d) and `base_cov` ((d,d) or (n_flows,d,d)):
        a shifted, anisotropic base N(z_0, Sigma_s). This is the section 5.1
        stream-appearance kernel - `base_mean[k]` is the latent image of the
        stream centre under flow k (per-flow because each ensemble member
        learns a different map), and `base_cov` is the stream's latent
        dispersion. When supplied, `latent_dispersion` is ignored.
        """
        torch = self._torch
        n_flows = len(self.flows)
        standard = (base_mean is None and base_cov is None
                    and float(latent_dispersion) == 0.0)
        with torch.no_grad():
            if standard:
                return np.stack([f.log_prob(x_t).cpu().numpy() for f in self.flows], axis=0)
            d = x_t.shape[1]
            cov = self._resolve_base_cov(base_cov, latent_dispersion, d, n_flows)
            per_flow_cov = (cov.ndim == 3)
            if per_flow_cov:
                chols = [np.linalg.cholesky(cov[k]) for k in range(n_flows)]
            else:
                chol = np.linalg.cholesky(cov)
            bm = None if base_mean is None else np.asarray(base_mean, dtype=float)
            out = []
            for k, f in enumerate(self.flows):
                noise, logabsdet = f._transform(x_t)
                z = noise.cpu().numpy()
                lad = logabsdet.cpu().numpy()
                m = None if bm is None else bm[k]
                ck = chols[k] if per_flow_cov else chol
                base = self._mvn_logpdf(z, m, ck)
                out.append(base + lad)
            return np.stack(out, axis=0)

    def latent_of(self, points: np.ndarray) -> np.ndarray:
        """Per-flow latent coordinates z = T^-1((x - mu)/sigma) of `points`.

        Returns shape (n_ensemble, n_points, d). Each ensemble member maps
        the same physical point to a different latent location, so the
        leading axis is the flow index. This is step 1 of the section 5.1 kernel:
        z_0 = latent_of(stream_centre)."""
        torch = self._torch
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        scaled = (pts - self._mean) / self._std
        x_t = torch.from_numpy(scaled).float().to(self._device)
        out = []
        with torch.no_grad():
            for f in self.flows:
                noise, _ = f._transform(x_t)
                out.append(noise.cpu().numpy())
        return np.stack(out, axis=0)

    def latent_and_logabsdet(self, points: np.ndarray):
        """Per-flow latent coords z and the forward log|det dz/dx_scaled|, in
        ONE forward pass per flow. Returns (z [n_ensemble,n,d],
        logabsdet [n_ensemble,n]).

        The physical field density of flow k is
        exp(logN(z;0,I) + logabsdet + log_jac), log_jac = -Sigma log sigma. Callers
        that need both the latent image (for a stream-kernel Gaussian) and the
        field density at the same points (e.g. the section 5.2 Bayes factor) use this
        to avoid a second forward pass."""
        torch = self._torch
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        scaled = (pts - self._mean) / self._std
        x_t = torch.from_numpy(scaled).float().to(self._device)
        zs, lads = [], []
        with torch.no_grad():
            for f in self.flows:
                noise, lad = f._transform(x_t)
                zs.append(noise.cpu().numpy())
                lads.append(lad.cpu().numpy())
        return np.stack(zs, axis=0), np.stack(lads, axis=0)

    def stream_log_prob(self, points: np.ndarray, center: np.ndarray, Sigma_s, returnLog: bool = True) -> np.ndarray:
        """Stream-appearance kernel density p_stream(points | center, Sigma_s).

        Models a stream whose velocities are a tight, anisotropic clump in
        the flow's latent space: base N(z_0, Sigma_s) with z_0 the per-flow latent
        image of `center` (a physical velocity). The density at `points` is
        the flow's exact change-of-variables density under that shifted base,
        ensemble-averaged over members (averaging rho, not log rho).

        `Sigma_s` is the latent-space stream covariance: scalar (-> c*I),
        length-d diagonal, (d,d) full, or (n_ensemble,d,d) per-flow.

        Even an isotropic Sigma_s = c*I gives anisotropic, locally-aligned
        scatter in velocity space, because the flow's Jacobian maps it to
        J*Sigma_s*J^T. To impose a *physical* orientation instead, pull a
        physical-velocity Sigma_s back through the local Jacobian (see
        `jacobian_at`) and pass the result here."""
        from scipy.special import logsumexp
        torch = self._torch
        z0 = self.latent_of(np.atleast_2d(center))[:, 0, :]   # (n_ensemble, d)
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        scaled = (pts - self._mean) / self._std
        x_t = torch.from_numpy(scaled).float().to(self._device)
        log_ps_scaled = self._per_flow_log_prob(x_t, base_mean=z0, base_cov=Sigma_s)
        log_jac = -float(np.sum(np.log(self._std)))
        log_ps = log_ps_scaled + log_jac
        log_ps = np.where(np.isnan(log_ps), -250.0, log_ps)
        log_p = logsumexp(log_ps, axis=0) - np.log(len(self.flows))
        return log_p if returnLog else np.exp(log_p)

    def draw_stream(self, size: int, center: np.ndarray, Sigma_s, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw `size` samples from the stream kernel p_stream(*|center,Sigma_s).

        Latent base N(z_0, Sigma_s) sampled per flow (z_0 = per-flow latent of
        `center`), then pushed through the inverse transform and
        de-standardized. Ensemble members are mixed by even allocation, like
        `draw`. `Sigma_s` accepts the same shapes as `stream_log_prob`."""
        torch = self._torch
        d = self.data.shape[1]
        n_flows = len(self.flows)
        z0 = self.latent_of(np.atleast_2d(center))[:, 0, :]   # (n_flows, d)
        cov = self._resolve_base_cov(Sigma_s, 0.0, d, n_flows)
        if cov.ndim == 3:
            chols = [np.linalg.cholesky(cov[k]) for k in range(n_flows)]
        else:
            L = np.linalg.cholesky(cov)
            chols = [L] * n_flows
        if rng is None:
            rng = np.random.default_rng()
        per_flow = [size // n_flows] * n_flows
        for k in range(size - sum(per_flow)):
            per_flow[k] += 1
        chunks = []
        with torch.no_grad():
            for k, (f, n) in enumerate(zip(self.flows, per_flow)):
                if n == 0:
                    continue
                eps = rng.standard_normal((n, d))
                z = z0[k] + eps @ chols[k].T
                zt = torch.from_numpy(z).float().to(self._device)
                samp, _ = f._transform.inverse(zt)
                chunks.append(samp.cpu().numpy())
        scaled = np.concatenate(chunks, axis=0)
        idx = rng.permutation(scaled.shape[0])
        return scaled[idx] * self._std + self._mean

    def jacobian_at(self, center: np.ndarray) -> np.ndarray:
        """Per-flow local Jacobian J = dv/dz of the latent->physical map,
        evaluated at the latent image z_0 of `center`.

        v = mu + sigma_sunT(z); the constant mu drops out, so J = d(sigma_sunT(z))/dz.
        Returns (n_ensemble, d, d). Use to pull a physical-velocity stream
        covariance back to the latent base (see `physical_cov_to_latent`):
        an isotropic-in-latent Sigma_s loses physical orientation, while a
        physical Sigma_s pulled back through J preserves ellipsoid tilt / vertex
        deviation."""
        torch = self._torch
        pts = np.atleast_2d(np.asarray(center, dtype=np.float64))
        if pts.shape[0] != 1:
            raise ValueError("jacobian_at expects a single center point")
        z0 = self.latent_of(pts)[:, 0, :]                    # (n_ensemble, d)
        std = torch.from_numpy(self._std).float().to(self._device)
        Js = []
        for k, f in enumerate(self.flows):
            zk = torch.from_numpy(z0[k]).float().to(self._device)

            def fn(z, f=f):
                x_scaled, _ = f._transform.inverse(z.unsqueeze(0))
                return x_scaled.squeeze(0) * std

            J = torch.autograd.functional.jacobian(fn, zk)
            Js.append(J.detach().cpu().numpy())
        return np.stack(Js, axis=0)

    def physical_cov_to_latent(self, center: np.ndarray, Sigma_phys) -> np.ndarray:
        """Pull a physical-velocity stream covariance back to per-flow latent
        covariances: Sigma_latent,k = J_k^-1 Sigma_phys J_k^-^T, with J_k = dv/dz at the
        latent image of `center`. Pass the (n_ensemble,d,d) result as
        `Sigma_s` to `stream_log_prob` / `draw_stream` to impose a specific
        physical stream orientation. `Sigma_phys` is scalar / length-d /
        (d,d) in physical velocity units."""
        J = self.jacobian_at(center)                         # (K, d, d)
        d = J.shape[-1]
        Sig = np.asarray(Sigma_phys, dtype=float)
        if Sig.ndim == 0:
            Sig = float(Sig) * np.eye(d)
        elif Sig.ndim == 1:
            Sig = np.diag(Sig)
        out = []
        for k in range(J.shape[0]):
            Jinv = np.linalg.inv(J[k])
            C = Jinv @ Sig @ Jinv.T
            out.append(0.5 * (C + C.T))                      # symmetrize
        return np.stack(out, axis=0)

    def __call__(self, points: np.ndarray, covfac: float = 1.0, covalpha: float = 0.0, returnLog: bool = False, show_contribs: bool = False, latent_dispersion: float = 0.0) -> np.ndarray:
        """Density at `points` (M, d). `covfac` etc. are accepted for interface
        compatibility but ignored - NF has no bandwidth knob.
        `latent_dispersion`: see `_per_flow_log_prob`."""
        torch = self._torch
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        scaled = (pts - self._mean) / self._std
        x_t = torch.from_numpy(scaled).float().to(self._device)
        # Ensemble averaging: average rho(x) across flows (NOT log rho). Buckley
        # et al. (Eq. 15-16) average phat values, not log phat.
        log_ps_scaled = self._per_flow_log_prob(x_t, latent_dispersion)
        # log of mean exp:  log_mean_p_scaled = logsumexp(log_ps) - log(n).
        from scipy.special import logsumexp
        log_p_scaled = (logsumexp(log_ps_scaled, axis=0)
                        - np.log(len(self.flows)))
        # Jacobian of (x - mu)/sigma -> divide density by Pi sigma -> subtract log Pi sigma.
        log_jac = -float(np.sum(np.log(self._std)))
        log_p = log_p_scaled + log_jac
        # NaN can come from the MADE block hitting numerical instability;
        # replace with a very-low finite floor (treats as "out of support")
        # so downstream callers don't have to special-case NaN. +inf is
        # left as-is - a saturated peak is a real (if degenerate) MAF
        # output, and callers that care should be robust to it. Floor of
        # -250 corresponds to density ~e^-250 ~ 2e-109, effectively zero.
        log_p = np.where(np.isnan(log_p), -250.0, log_p)
        if returnLog:
            return log_p
        return np.exp(log_p)

    @property
    def covalpha_overall(self) -> float:
        return 0.0

    def data_side_neff(self, points, eval_weights=None, covfac: float = 1.0, covalpha: float = 0.0) -> float:
        """NaN - the NF transforms the entire distribution rather than summing
        per-training-particle kernels, so "how many data points contribute"
        doesn't have a per-particle meaning. Returned for interface
        compatibility with adaptiveKDE / gaussianKDEWrapper.data_side_neff."""
        return float('nan')

    # Pickling: move everything to CPU on save so the cached pickle is
    # portable between CUDA and CPU-only machines. dill/torch's default
    # behavior pickles tensors with their .device attached, which raises
    # `Attempting to deserialize object on a CUDA device but
    # torch.cuda.is_available() is False` when loading a GPU-saved
    # pickle on a CPU-only node. __setstate__ re-attaches the running
    # process's torch module and moves flows to whichever device is
    # available at load time.
    def __getstate__(self):
        state = self.__dict__.copy()
        # Drop the torch-module reference; it's re-imported on load.
        state.pop('_torch', None)
        # Move every flow's parameters to CPU before pickling so the
        # resulting bytes contain only CPU tensors. We don't mutate
        # self.flows in place - temporarily copy via state_dict to keep
        # the live object on whatever device it's running on.
        flows = state.get('flows', None)
        if flows is not None:
            cpu_flows = []
            for f in flows:
                cpu_f = type(f).__new__(type(f))     # cheap shallow copy
                cpu_f.__dict__.update(f.__dict__)
                cpu_f.load_state_dict({k: v.detach().cpu() for k, v in f.state_dict().items()})
                cpu_flows.append(cpu_f.cpu())
            state['flows'] = cpu_flows
        # Same treatment for the back-compat single-flow alias.
        if state.get('flow', None) is not None:
            state['flow'] = state['flows'][0] if state.get('flows') else None
        # Store the device as a string so reconstruction doesn't require
        # CUDA on the load side.
        if '_device' in state:
            state['_device'] = str(state['_device'])
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Pickles from before the spline option lack transform_type.
        if 'transform_type' not in state:
            self.transform_type = 'affine'
        # Pickles from before weighted-NLL training lack these.
        if 'transform_type' not in state or '_weighted' not in state:
            self._weighted = getattr(self, '_weighted', False)
        # Re-import torch / nflows in the running process.
        torch, _MAF = _import_nf()
        self._torch = torch
        # Pick whichever device is available now. CPU is the safe default;
        # callers on a GPU box can manually `kde.flows = [f.cuda() for f
        # in kde.flows]` if they want to push back to GPU.
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._device = device
        if getattr(self, 'flows', None) is not None:
            self.flows = [f.to(device) for f in self.flows]
            self.flow = self.flows[0]

    def per_flow_density(self, points: np.ndarray, latent_dispersion: float = 0.0) -> np.ndarray:
        """Density at `points` from EACH ensemble member separately.

        Returns array of shape (n_ensemble, n_points). The standard `__call__`
        averages these via logsumexp to produce the ensemble density. This
        method exposes the per-flow values so callers can probe ensemble
        disagreement - e.g. compute a rate using each flow individually and
        compare the spread to the mean as a proxy for NF reliability at the
        rate-leverage region. `latent_dispersion`: see `_per_flow_log_prob`.
        """
        torch = self._torch
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        scaled = (pts - self._mean) / self._std
        x_t = torch.from_numpy(scaled).float().to(self._device)
        log_ps_scaled = self._per_flow_log_prob(x_t, latent_dispersion)
        log_jac = -float(np.sum(np.log(self._std)))
        log_ps = log_ps_scaled + log_jac
        # Same NaN treatment as __call__.
        log_ps = np.where(np.isnan(log_ps), -250.0, log_ps)
        return np.exp(log_ps)

    def draw(self, size: int, covfac: float = 1.0, rng: np.random.Generator | None = None, latent_dispersion: float = 0.0) -> np.ndarray:
        """Draw `size` samples from the trained flow.

        Each ensemble member contributes proportional samples (Buckley-style
        averaging applies to densities; for sampling we mix by drawing from
        whichever flow each sample is assigned to). `covfac` is ignored
        (interface compatibility with the cracked KDE API).
        `latent_dispersion` s > 0 draws z ~ N(0, (1+s^2)*I) and pushes it
        through the inverse transform - samples from the same widened-base
        population that `__call__(..., latent_dispersion=s)` evaluates."""
        torch = self._torch
        s2 = float(latent_dispersion) ** 2
        # Mix ensemble members by even allocation. NF.sample returns
        # samples in the standardized frame; we de-standardize at the end.
        n_flows = len(self.flows)
        # Assign floor(size/n_flows) to each + handle remainder
        per_flow = [size // n_flows] * n_flows
        for k in range(size - sum(per_flow)):
            per_flow[k] += 1
        chunks = []
        with torch.no_grad():
            for f, n in zip(self.flows, per_flow):
                if n == 0:
                    continue
                if s2 == 0.0:
                    s = f.sample(n).cpu().numpy()
                else:
                    d = self.data.shape[1]
                    noise = (torch.randn(n, d, device=self._device)
                             * float(np.sqrt(1.0 + s2)))
                    samp, _ = f._transform.inverse(noise)
                    s = samp.cpu().numpy()
                chunks.append(s)
        scaled = np.concatenate(chunks, axis=0)
        # Shuffle so the ensemble-member ordering doesn't bias downstream
        # plotting that subsamples the head of the array.
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.permutation(scaled.shape[0])
        scaled = scaled[idx]
        # De-standardize: x = scaled * sigma + mu
        return scaled * self._std + self._mean
