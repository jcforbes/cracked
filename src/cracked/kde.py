"""
adaptive_kde - locally-adaptive 6D KDE with cross-validation, shrinkage, and
multiplicative bias correction. Originally part of isostreams.py; split out as
a standalone module on the way to becoming its own package.

Public API:
  adaptiveKDE          - per-point local-covariance Gaussian KDE
  cvAdaptiveKDE        - wrapper that picks (covfac, covalpha, shrinkage) by k-fold CV
  gaussianKDEWrapper   - thin wrapper around scipy.stats.gaussian_kde with the
                         same `__call__(points, covfac=..., show_contribs=...)` interface
                         as adaptiveKDE
  mockScipyKde         - adaptiveKDE-style class but with the global covariance and
                         Scott's rule, useful for like-with-like scipy comparisons

Helpers:
  cvkde_evaluate_inner - per-point evaluation entry used by ThreadPoolExecutor in
                         adaptiveKDE.__call__
  timer                - small profiling helper used internally
"""
import copy
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
import pdb
import scipy.spatial
import scipy.stats
import scipy.special
from sklearn.model_selection import KFold
from tqdm import tqdm


class timer:
    """Tiny profiling helper used by `adaptiveKDE` when `profile=True`."""
    def __init__(self):
        self.ticks = [time.time()]
        self.labels = []
    def tick(self, label):
        self.ticks.append(time.time())
        self.labels.append(label)
    def timeto(self, label):
        if label in self.labels:
            i = self.labels.index(label)
            return self.ticks[i + 1] - self.ticks[i]
        else:
            return np.nan
    def report(self):
        arr = np.array(self.ticks)
        deltas = arr[1:] - arr[:-1]
        print("Timing report:")
        for i in range(len(self.labels)):
            print(self.labels[i], deltas[i], 100 * deltas[i] / np.sum(deltas), r'%')


# The KDE classes below were split out of isostreams.py on 2026-04-30.
class adaptiveKDE:
    def __init__(self, dataIn, scalings=None, nn=63, covfac=1.0, covalpha=0.0, profile=False, use_multiprocessing=True, shrinkage=0.0, shrinkage_target='global', K_pool=300, weights=None):
        data = copy.deepcopy(dataIn)
        self.use_multiprocessing = use_multiprocessing
        # Per-training-point input weights (e.g. Gaia ISO production weights).
        # Normalised to sum to N so the existing /N normalisation is unchanged
        # and weights=None (uniform) reproduces the original behaviour exactly.
        # Weighting acts on the kernel SUM: fhat(x) = Sigma_k w_k K_k(x) / Sigma_k w_k,
        # implemented in log-space by adding log w_k to each kernel's log-contrib.
        # The per-point KNN local covariance Sigma_k is NOT reweighted (first-order;
        # the neighbour-sample covariance stays geometric).
        N_data = data.shape[0]
        if weights is None:
            self.weights = np.ones(N_data)
        else:
            w = np.asarray(weights, dtype=float).reshape(-1)
            if w.shape != (N_data,):
                raise ValueError(f"weights shape {w.shape} != ({N_data},)")
            if np.any(w < 0) or not np.all(np.isfinite(w)):
                raise ValueError("weights must be non-negative and finite")
            self.weights = w / w.sum() * N_data
        self.log_weights = np.log(np.maximum(self.weights, 1e-300))
        if scalings is None:
            self.scales = np.ones(data.shape[1])
        else:
            self.scales = scalings[:]
            assert len(self.scales)==data.shape[1]
        self.jacfac = 1.0
        self.covfac_overall = covfac
        self.covalpha_overall= covalpha
        if profile:
            tim = timer()
        for i in range(len(self.scales)):
            data[:,i] = data[:,i]/self.scales[i]
            self.jacfac *= self.scales[i] # kde returns density per unit x/scale, so to get density in x^-1... returned_dens = (scale/x) => dens_phys = (1/x) = returned_dens/scale
        # jacfac_marg covers position scales, jacfac_cond covers velocity scales - only meaningful in the 6D phase-space layout used by evaluate_marginal/evaluate_conditional.
        self.jacfac_marg = float(np.prod(self.scales[:3])) if len(self.scales) >= 6 else self.jacfac
        self.jacfac_cond = float(np.prod(self.scales[3:6])) if len(self.scales) >= 6 else 1.0
        self.data = data[:,:]
        if profile:
            tim.tick('rescale data')
        try:
            self.tree = scipy.spatial.KDTree( data, leafsize=8, compact_nodes=True, balanced_tree=True )
        except:
            pdb.set_trace()
        if profile:
            tim.tick('build tree')
        self.nn = nn
        
        neighbor_dist, neighbor_inds = self.tree.query( data, k=self.nn ) # n x m -> n x k.  Don't think we need the distances for anything.
        if profile:
            tim.tick('query tree')

        self.neighbor_inds = neighbor_inds
        self.max_neighbor_dist = neighbor_dist[:,-1] #np.max(neighbor_dist, axis=1) # distance to each points nn'th nearest neighbor
        self.overall_max_dist = np.max(self.max_neighbor_dist) # maximum in the whole sample, i.e. distance associated with "the most isolated point"

        # might have to do a for loop here?
        self.covariances = np.zeros( (data.shape[0], data.shape[1], data.shape[1]) ) # n x m x m 
        self.normals = np.zeros( data.shape[0], dtype=object)
        for i in range(data.shape[0]):
            # KDTree.query returns the queried point as its own closest neighbor, so neighbor_inds[i] already includes data[i].
            selec = data[ self.neighbor_inds[i,:], :]
            self.covariances[i, :,:] = np.cov(selec.T)

            #self.normals[i] = scipy.stats.multivariate_normal( data[i,:], self.covfac*self.covariances[:,:,i] )

        # Cache the shrinkage-INVARIANT pieces so apply_shrinkage() can rebuild
        # the shrinkage-dependent attributes without redoing the tree query or
        # the per-point local covariance loop. This is the cvAdaptiveKDE outer
        # loop's biggest opportunity - a single fold's adaptiveKDE is reused
        # across every shrinkage value in the CV grid.
        self.covariances_local = self.covariances.copy()
        self.global_cov = np.cov(data.T)

        # Shrinkage target: 'global' (default) pools toward the full-data covariance;
        # 'local_pooled' pools toward a position-local pooled covariance (K_pool NN
        # in 3D position space). The latter is the right choice when the data has
        # large-scale spatial structure (e.g. a galactic ring) where Sigma_global picks
        # up the structure's geometry rather than the local-noise scale that
        # shrinkage was meant to stabilise toward.
        self.shrinkage_target = shrinkage_target
        self.K_pool = int(K_pool)
        self.shrinkage_target_per_point = None
        if self.shrinkage_target == 'local_pooled' and data.shape[1] >= 3:
            pos_tree = scipy.spatial.KDTree(data[:, :3], leafsize=8, compact_nodes=True, balanced_tree=True)
            K_eff = min(self.K_pool, data.shape[0])
            _, pool_inds = pos_tree.query(data[:, :3], k=K_eff)
            target_per_pt = np.empty((data.shape[0], data.shape[1], data.shape[1]))
            for i in range(data.shape[0]):
                target_per_pt[i] = np.cov(data[pool_inds[i]].T)
            self.shrinkage_target_per_point = target_per_pt
            if profile:
                tim.tick('local-pooled shrinkage target')

        # Per-axis data span (16-84 percentile) - used by `cvAdaptiveKDE`
        # to skip CV candidates whose effective kernel radius exceeds a
        # multiple of this span (kernels reaching across the dataset are
        # almost certainly wrong; saves the cost of building/evaluating
        # them).
        spans = (np.percentile(data, 84, axis=0)
                 - np.percentile(data, 16, axis=0))
        self.data_axis_span = spans
        self.data_max_span = float(np.max(spans))

        # Multiplicative Laplace-bias correction. Default 0 (no correction). When set
        # to b, every density evaluation is multiplied by exp(-b) (or, in log-space,
        # b is subtracted). Set this from outside via `set_log_bias_correction(b)`
        # or the helper `estimate_log_bias_from_truth(...)` below.
        self.log_bias_correction = 0.0

        # Shrinkage-dependent setup (covariances, Cholesky, eigh, Schur).
        self.apply_shrinkage(shrinkage)

        if profile:
            tim.tick('build normals')
            tim.report()

    def apply_shrinkage(self, shrinkage):
        """Rebuild shrinkage-dependent attributes from the cached `covariances_local`
        and `global_cov`. Use this to sweep over shrinkage values without redoing
        the tree query, neighbour list, or local covariance estimation - the
        speedup that matters for cvAdaptiveKDE's outer shrinkage loop.

        Mixing convention (revised 2026-05-03):
            Sigma_eff(alpha) = (1-alpha)*Sigma_local + alpha*(Silverman^2*Sigma_global)
        where Silverman^2 = N^(-2/(d+4)). At alpha=1 the kernel covariance matches
        scipy.gaussian_kde's default; at alpha=0 it's the original adaptiveKDE
        kernel (covfac applied later by the caller). Earlier convention used
        alpha*Sigma_global directly which had a ~1/Silverman^2 scale mismatch with
        scipy at alpha=1 - fixed here so CV's covfac is calibrated to ~1 across
        the full alpha range rather than fighting the scale mismatch.

        Note: covfac semantics at alpha=0 are NOT Silverman-normalized; the
        natural-bandwidth scaling at alpha=0 is at covfac~Silverman^2. cvAdaptiveKDE
        applies a Silverman renormalisation at the grid/pick boundary so the
        user-facing covfac=1 corresponds to the Silverman scale uniformly.
        """
        self.shrinkage = float(shrinkage)
        if self.shrinkage > 0.0:
            N = self.data.shape[0]
            d = self.data.shape[1]
            if self.shrinkage_target_per_point is not None:
                # Position-local pooled target: per-point Sigma from K_pool spatially-
                # nearest neighbours. Avoids dragging in distant geometry (e.g.
                # the ring on the other side of the galaxy) into the kernel.
                # Silverman^2 uses K_pool (the effective sample size used to
                # estimate the pooled target) rather than N, so at shrinkage=1
                # the kernel volume matches "scipy.gaussian_kde over the K_pool
                # local subset" - the asymptotically right reference for a
                # locally-pooled covariance estimate.
                silverman_sq = self.K_pool ** (-2.0 / (d + 4))
                sigma_target = silverman_sq * self.shrinkage_target_per_point
                self.covariances = ((1.0 - self.shrinkage) * self.covariances_local
                                    + self.shrinkage * sigma_target)
            else:
                silverman_sq = N ** (-2.0 / (d + 4))
                sigma_global_kernel = silverman_sq * self.global_cov
                self.covariances = ((1.0 - self.shrinkage) * self.covariances_local
                                    + self.shrinkage * sigma_global_kernel[None, :, :])
        else:
            self.covariances = self.covariances_local.copy()

        self.choleskys = np.linalg.cholesky(self.covariances)
        vals, vecs = np.linalg.eigh(self.covariances)
        self.logdets = np.sum(np.log(vals), axis=1)
        valsinvs = 1.0 / vals
        self.Us = vecs * np.sqrt(valsinvs)[:, None]
        self.max_from_cov = np.sqrt(np.max(vals))
        # Per-kernel largest eigenvalue of Sigma_j - sets the spatial radius beyond
        # which kernel j contributes negligibly: r_j ~ m*sqrt(lambda_max*covfac*|Sigma_j|^covalpha).
        self.lambda_max_per_kernel = vals[:, -1]
        self.log_lambda_max_per_kernel = np.log(self.lambda_max_per_kernel)

        if self.data.shape[1] == 6:
            xcov = self.covariances[:, :3, :3]
            vcov = self.covariances[:, 3:, 3:]
            crosscov = self.covariances[:, 3:, :3]
            self.onetwotwotwo = np.matmul(crosscov, np.linalg.inv(xcov))
            self.covcond = vcov - np.matmul(self.onetwotwotwo, np.transpose(crosscov, axes=[0, 2, 1]))
            vals, vecs = np.linalg.eigh(self.covcond)
            if not np.all(vals >= 0):
                vals = np.clip(vals, 1.0e-13, None)
            self.logdetsCond = np.sum(np.log(vals), axis=1)
            valsinvs = 1.0 / vals
            self.UsCond = vecs * np.sqrt(valsinvs)[:, None]

            vals, vecs = np.linalg.eigh(xcov)
            self.logdetsMarg = np.sum(np.log(vals), axis=1)
            valsinvs = 1.0 / vals
            self.UsMarg = vecs * np.sqrt(valsinvs)[:, None]

    def get_data(self):
        # get the original data back ( need to de-scale )
        ret = copy.deepcopy(self.data)
        for i in range(len(self.scales)):
            ret[:,i] *= self.scales[i]
        return ret

    def _max_kernel_radius(self, covfac_max=1.0, covalpha_range=None, m_thresh=5.0):
        """Spatial radius (in scaled coords) capturing every kernel whose
        Mahalanobis-distance contribution exceeds exp(-m^2/2) ~ 4e-6 at m=5.

        For kernel j with Sigma_eff_j = (covfac_overall*covfac)*|Sigma_j|^(covalpha_overall+covalpha)*Sigma_j,
        the principal kernel scale is sqrtlambda_max(Sigma_eff_j). Conservative single-radius
        truncation takes max_j over all data, max over (covfac, covalpha) in the
        evaluation range. Variable-length neighbor lists from `query_ball_point`
        are then padded to a uniform shape downstream.
        """
        log_cf = np.log(self.covfac_overall * covfac_max)
        # covalpha_range is (low, high) - kernel j's log scale grows linearly
        # in covalpha * logdets[j], so the worst-case is at one of the endpoints
        # depending on sign(logdets[j]).
        ca_lo, ca_hi = (covalpha_range if covalpha_range is not None
                        else (self.covalpha_overall, self.covalpha_overall))
        ca_lo = ca_lo + self.covalpha_overall
        ca_hi = ca_hi + self.covalpha_overall
        log_lam_lo = ca_lo * self.logdets + self.log_lambda_max_per_kernel
        log_lam_hi = ca_hi * self.logdets + self.log_lambda_max_per_kernel
        log_lam_max = float(np.max(np.maximum(log_lam_lo, log_lam_hi)))
        return float(m_thresh * np.exp(0.5 * (log_cf + log_lam_max)))

    def precompute_query(self, pointsIn, covfac_max=1.0, covalpha_range=None, m_thresh=5.0, n_to_get=None):
        """Compute the per-query Mahalanobis-square matrix at unit covfac/covalpha.

        Returns a dict suitable for `eval_from_cache`. Splitting the call this way lets
        a CV loop sweep over (covfac, covalpha) without redoing the tree query or the
        squared-deviation matrix - both of which are the same for any choice of the
        kernel rescaling factors.

        Truncation is distance-based via `tree.query_ball_point`: include every kernel
        whose centre is within `m_thresh * sqrtlambda_max(Sigma_eff)` of the query, where the max
        is taken over the (covfac_max, covalpha_range) bracket. Variable-length lists
        are padded to (M, K_max) with -1 sentinel; `eval_from_cache` masks padded
        entries before logsumexp.

        `n_to_get` is a legacy kNN override (for `loo_density_at_data` in particular,
        where we need exactly N-1 neighbours per data point). When supplied, the radius
        truncation is bypassed.
        """
        points = copy.deepcopy(pointsIn)
        for i in range(len(self.scales)):
            points[:, i] = points[:, i] / self.scales[i]
        M = points.shape[0]

        if n_to_get is not None:
            # Legacy kNN path - fixed K nearest, no padding needed.
            _, neighbors = self.tree.query(points, n_to_get)
            Us = self.Us[neighbors]
            devs = points[:, None, :] - self.data[neighbors]
            devUs = np.einsum('mni,mnij->mnj', devs, Us)
            mahas = np.sum(np.square(devUs), axis=2)
            logdets_at_n = self.logdets[neighbors]
            return {'mahas': mahas, 'neighbors': neighbors,
                    'logdets_at_n': logdets_at_n, 'M': M, 'valid': None}

        r_max = self._max_kernel_radius(covfac_max=covfac_max, covalpha_range=covalpha_range, m_thresh=m_thresh)
        # When r_max captures essentially all data (e.g. high shrinkage with
        # large Sigma_global eigenvalue, as in disk-stream data along the ring's
        # tangential direction), the (M, K_max, d, d) cache can grow into the
        # tens of GB. Chunk along M so peak memory stays bounded by `mem_budget`.
        # Each chunk gets its own (K_max_chunk, ...) and is concatenated at the end.
        d = self.data.shape[1]
        bytes_per_entry = 8 * (1 + 1 + d + d * d)   # mahas, logdet, devs, Us per (M, K) cell
        # We don't know K_max yet - sample a single query to estimate.
        N_data = self.data.shape[0]
        sample_idx = M // 2
        sample_neighbors = self.tree.query_ball_point(points[sample_idx], r_max)
        K_max_est = max(1, int(len(sample_neighbors) * 1.2 + 8))   # add headroom
        K_max_est = min(K_max_est, N_data)
        mem_budget_bytes = 2 * 1024**3   # 2 GB
        chunk_size = max(1, int(mem_budget_bytes // (bytes_per_entry * K_max_est)))
        chunk_size = min(chunk_size, M)

        if chunk_size >= M:
            return self._precompute_chunk(points, r_max)

        # Process in chunks; concatenate results, padding shorter chunks to the
        # max K across chunks.
        sub_caches = []
        for start in range(0, M, chunk_size):
            end = min(start + chunk_size, M)
            sub_caches.append(self._precompute_chunk(points[start:end], r_max))
        K_max_global = max(sc['mahas'].shape[1] for sc in sub_caches)
        mahas = np.zeros((M, K_max_global), dtype=np.float64)
        logdets_at_n = np.zeros((M, K_max_global), dtype=np.float64)
        neighbors = np.zeros((M, K_max_global), dtype=np.int64)
        valid = np.zeros((M, K_max_global), dtype=bool)
        offset = 0
        for sc in sub_caches:
            n_rows, K_sub = sc['mahas'].shape
            mahas[offset:offset + n_rows, :K_sub] = sc['mahas']
            logdets_at_n[offset:offset + n_rows, :K_sub] = sc['logdets_at_n']
            neighbors[offset:offset + n_rows, :K_sub] = sc['neighbors']
            valid[offset:offset + n_rows, :K_sub] = sc['valid']
            offset += n_rows
        return {'mahas': mahas, 'neighbors': neighbors,
                'logdets_at_n': logdets_at_n, 'M': M, 'valid': valid}

    def _precompute_chunk(self, points, r_max):
        """Build the (M, K_max) cache for a single batch of query points.
        Caller is responsible for chunking M to bound memory.
        """
        M = points.shape[0]
        neighbor_lists = self.tree.query_ball_point(points, r_max)
        lengths = np.fromiter((len(lst) for lst in neighbor_lists), dtype=int, count=M)
        K_max = int(max(1, lengths.max()))
        neighbors = np.full((M, K_max), -1, dtype=np.int64)
        for i, lst in enumerate(neighbor_lists):
            if len(lst) > 0:
                neighbors[i, :len(lst)] = lst
        valid = (neighbors >= 0)
        safe = np.where(valid, neighbors, 0)
        Us = self.Us[safe]                                  # (M, K_max, d, d)
        devs = points[:, None, :] - self.data[safe]          # (M, K_max, d)
        devUs = np.einsum('mni,mnij->mnj', devs, Us)
        mahas = np.sum(np.square(devUs), axis=2)             # (M, K_max)
        logdets_at_n = self.logdets[safe]                    # (M, K_max)
        return {'mahas': mahas, 'neighbors': safe,
                'logdets_at_n': logdets_at_n, 'M': M, 'valid': valid}

    def eval_from_cache(self, cache, covfac=1.0, covalpha=0.0, returnLog=False):
        """Finalise a density estimate from a cache returned by `precompute_query`.

        Equivalent to `__call__` (modulo the truncation in `precompute_query`)
        but skips the tree query and Mahalanobis computation. Note: `cache['mahas']` is
        the unscaled Mahalanobis (i.e. evaluated against Sigma_j, not against covfac_eff*Sigma_j),
        so we apply the per-kernel rescaling here. Padded entries (cache['valid']==False)
        are masked to -inf so they drop out of the logsumexp.
        """
        mahas = cache['mahas']
        logdets_at_n = cache['logdets_at_n']
        valid = cache.get('valid', None)
        dim = self.data.shape[1]
        log2pi = np.log(2.0 * np.pi)
        log_covfac = (np.log(self.covfac_overall * covfac)
                      + (covalpha + self.covalpha_overall) * logdets_at_n)
        # The kernel uses Sigma_eff = exp(log_covfac)*Sigma_j; Mahalanobis under Sigma_eff is
        # mahas/exp(log_covfac); the log-determinant adds dim*log_covfac.
        mahas_eff = mahas / np.exp(log_covfac)
        contribs = -0.5 * (dim * log2pi + mahas_eff + dim * log_covfac + logdets_at_n)
        contribs = contribs + self.log_weights[cache['neighbors']]
        if valid is not None:
            contribs = np.where(valid, contribs, -np.inf)
        est = scipy.special.logsumexp(contribs, axis=1)
        if returnLog:
            return est - np.log(self.jacfac) - np.log(len(self.normals)) - self.log_bias_correction
        return np.exp(est) / self.jacfac / len(self.normals) * np.exp(-self.log_bias_correction)

    def eval_neff_from_cache(self, cache, covfac=1.0, covalpha=0.0):
        """Per-evaluation Kish ESS at each query point: (Sigma w_i)^2 / Sigma w_i^2.

        Mirrors `eval_from_cache` but returns the effective number of kernels
        contributing to each query rather than the density. A median ~ 1
        signals the KDE is acting as a particle counter at these eval points.
        """
        mahas = cache['mahas']
        logdets_at_n = cache['logdets_at_n']
        valid = cache.get('valid', None)
        dim = self.data.shape[1]
        log2pi = np.log(2.0 * np.pi)
        log_covfac = (np.log(self.covfac_overall * covfac)
                      + (covalpha + self.covalpha_overall) * logdets_at_n)
        mahas_eff = mahas / np.exp(log_covfac)
        contribs = -0.5 * (dim * log2pi + mahas_eff + dim * log_covfac + logdets_at_n)
        contribs = contribs + self.log_weights[cache['neighbors']]
        if valid is not None:
            contribs = np.where(valid, contribs, -np.inf)
        log_sum = scipy.special.logsumexp(contribs, axis=1)
        log_sum_sq = scipy.special.logsumexp(2.0 * contribs, axis=1)
        return np.exp(2.0 * log_sum - log_sum_sq)

    def __call__(self, pointsIn, profile=False, covfac=1.0, covalpha=0.0, returnLog=False, show_contribs=False):
        # first find the points we need to worry about in the tree:
        tim = timer()
        if profile:
            tim = timer()
        points = copy.deepcopy(pointsIn)
        if profile:
            tim.tick('copy data')
        
        for i in range(len(self.scales)):
            points[:,i] = points[:,i]/self.scales[i]
        if profile:
            tim.tick('scale data')
            

        #neighbors = self.tree.query_ball_point( points, 2*self.max_from_cov) # neighbors is a list of lists, one list per point (the indexes to data of the neighbors of that point).
        #neighbors = self.tree.query_ball_point( points, self.overall_max_dist) # very conservative!
        # Truncate to nn*10 nearest neighbours: rel-err on density vs full N-1 sum
        # is <1e-4 on a 6D unit gaussian at covfac=1, well below MC noise on rate
        # calculations. Kernel sigma scales as sqrtcovfac so the volume that needs to be
        # captured grows as covfac^(d/2); widen the truncation by that factor when
        # covfac is large, saturated at N-1.
        dim = self.data.shape[1]
        cf_eff = max(1.0, self.covfac_overall * covfac)
        trunc_factor = cf_eff ** (dim / 2.0)
        n_to_get = min(int(self.nn * 10 * trunc_factor), np.shape(self.data)[0] - 1)

        M = points.shape[0]
        est = np.zeros(M)
        nneighbors = np.zeros(M)

        dim = points.shape[1]
        log2pi = np.log(2 * np.pi)
        logcovfac = np.log(self.covfac_overall * covfac) + (covalpha+self.covalpha_overall)*self.logdets



#        # just spitballing here...
#        tim.tick("setup")
#        Us = self.Us * 1.0/np.exp(0.5*logcovfac[:,None,None])
#        tim.tick("Us")
#        devs = points[:,np.newaxis,:] - self.data[:,:]
#        tim.tick("devs")
#        devUs = np.einsum('...i,...ij->...j', devs, Us)
#        tim.tick("devUs")
#        mahas = np.sum(np.square(devUs),axis=2)
#        tim.tick("mahas")
#        est = scipy.special.logsumexp( -0.5 * (dim * log2pi + mahas + logcovfac*dim + self.logdets), axis=1 )
#        tim.tick("logsumexp")
        #tim.report()
        #pdb.set_trace()
    
#        Us = self.Us[neighbors] * 1.0/np.sqrt(covfac)
#        devs = points[:,np.newaxis,:] - ( self.data[:,3:] + np.squeeze(np.matmul(self.onetwotwotwo[:,:,:], (xvec-self.data[:,:3])[:,:,np.newaxis])) )
#        devUs = np.einsum('...i,...ij->...j', devs, Us)
#        mahas = np.sum(np.square(devUs), axis=2)
#        dim = points.shape[1]
#        log2pi = np.log(2*np.pi)
#        est = scipy.special.logsumexp( -0.5 * (dim * log2pi + mahas + dim*np.log(covfac) + self.logdetsCond)[:,pos.flatten()] + logpofz, axis=1 )


        neff = np.zeros(M)
        contrib_matrix = np.zeros(self.data.shape[0])


        if not hasattr(self, 'use_multiprocessing'):
            self.use_multiprocessing = True

        # Chunk along M so peak memory stays bounded. The dominant cost is the
        # `tree.query(pts_chunk, n_to_get)` result - two (chunk, n_to_get) arrays
        # at 8 bytes/entry. When CV picks a high covfac, n_to_get saturates at
        # N-1 and an un-chunked call on a large eval batch (e.g. Nboot~100k for
        # IS rate eval) can allocate ~10 GB per worker process, which times
        # ~12 ProcessPool workers gives the ~100 GB blow-up observed
        # 2026-05-23. The 2 GB budget matches `precompute_query`.
        mem_budget_bytes = 2 * 1024**3
        chunk_size = max(1, int(mem_budget_bytes // (max(n_to_get, 1) * 8 * 2)))
        chunk_size = min(chunk_size, M)

        for start in range(0, M, chunk_size):
            end = min(start + chunk_size, M)
            pts_chunk = points[start:end]
            dists, neighbors = self.tree.query(pts_chunk, n_to_get)
            if profile:
                tim.tick('query tree (chunk)')

            if self.use_multiprocessing:
                # Pass neighbors[i] (the K-NN index array) into the worker so the
                # density sum is restricted to the kdtree truncation (matches the
                # non-MP fallback below and the docstring). Each worker's Us
                # allocation becomes (K, d, d) instead of (N, d, d), which is
                # the actual reason for doing the tree query at all.
                with ThreadPoolExecutor() as exe:
                    results = exe.map(cvkde_evaluate_inner, [(self.Us, logcovfac, self.data, self.logdets, neighbors[i], pts_chunk[i], i, self.log_weights) for i in range(len(neighbors))])

                for result in results:
                    est_i, neff_i, contribs_i, i = result
                    est[start + i] = est_i
                    neff[start + i] = neff_i
                    # Scatter the K-long contribs back into the N-long matrix
                    # at the kNN indices for this query point.
                    contrib_matrix[neighbors[i]] += np.exp(contribs_i)
            else:
                for i in range(len(neighbors)):
                    nneighbors[start + i] = len(neighbors[i])
                    if nneighbors[start + i] > 0:
                        # neighbors[i] is a list of indexes for the neighbors of pts_chunk[i]
                        #distrs = self.normals[neighbors[i]]
                        try:
                            Us = self.Us[neighbors[i]] * 1.0/np.exp(0.5*logcovfac[neighbors[i],None,None]) #* 1/np.sqrt(self.covfac_overall * covfac)
                        except:
                            pdb.set_trace()
                        devs = pts_chunk[i,:] - self.data[neighbors[i]] # surprised this broadcasts(?) correctly
                        devUs = np.einsum('ni,nij->nj', devs, Us)
                        mahas = np.sum(np.square(devUs), axis=1)

                        # Compute and broadcast scalar normalizers.
                        #dim = len(vals[0])
                        contribs = -0.5 * (dim * log2pi + mahas + logcovfac[neighbors[i]]*dim + self.logdets[neighbors[i]])
                        contribs = contribs + self.log_weights[neighbors[i]]
                        contrib_matrix[neighbors[i]] += np.exp(contribs)
                        est[start + i] = scipy.special.logsumexp(contribs)
                        neff[start + i] = np.exp(2*est[start + i] - scipy.special.logsumexp(2*contribs))
                    else:
                        est[start + i] = -100000.0

            # Free chunk allocations before the next iteration (Python's reference
            # counting would do this on reassignment, but explicit del helps when
            # the worker is memory-tight).
            del dists, neighbors
            


            #for j in range(len(distrs)):
                #est += distrs[j].pdf( points ) # there's no way this is right! We should only be contributing to est[i]!
                #est[i] += distrs[j].pdf( points[i] )
        if profile:
            #print("neighbors stats:", np.mean(nneighbors), np.percentile(nneighbors,[5,16,50,84,95]))
            tim.tick('evaluate pdfs')
            tim.report()

        if show_contribs:
            print("Mean, median, min, max of neff: ",np.mean(neff), np.median(neff), np.min(neff), np.max(neff))

            neff_across = np.sum(contrib_matrix)**2/np.sum(contrib_matrix*contrib_matrix)
            print("Effective number of datapoints used in evaluting densities at these ",points.shape[0],"test points: ",neff_across)

        if returnLog:
            return est - np.log(self.jacfac) - np.log(len(self.normals)) - self.log_bias_correction
        else:
            return (np.exp(est) / self.jacfac) / len(self.normals) * np.exp(-self.log_bias_correction)

    def data_side_neff(self, points, eval_weights=None, covfac=1.0, covalpha=0.0):
        """Kish ESS of training-particle contributions to a weighted aggregation.

        For each training particle k, its contribution to the weighted sum
            S = Sigma_i eval_weights[i] * fhat(points[i])
        is
            C_k = (eval_weights[i] * K_k(points[i] - data[k]; Sigma_k)) / (N * jacfac)
            (summed over i)
        and the data-side N_eff is
            (Sigma_k C_k)^2 / Sigma_k C_k^2
        This counts "how many of the N training particles are effectively
        contributing to the weighted aggregation S."

        Complements the two N_effs already in use:
          - per-evaluation-point N_eff (line 541): for each query point, how
            many training particles' kernels dominate the density there.
          - IS Kish ESS (computed downstream from rate_sphere_importance):
            for the rate sum, how many IS samples carry the weight.

        Pass eval_weights = the per-IS-sample importance weights (i.e. the
        `resj_*` arrays from rate_sphere_importance) to get the rate-weighted
        data-side N_eff. Unweighted (eval_weights=None) gives the same metric
        as `show_contribs=True`'s `neff_across` print.
        """
        points = np.asarray(copy.deepcopy(points))
        if points.ndim == 1:
            points = points[np.newaxis, :]
        for i in range(len(self.scales)):
            points[:, i] = points[:, i] / self.scales[i]
        M = points.shape[0]
        if eval_weights is None:
            eval_weights = np.ones(M)
        else:
            eval_weights = np.asarray(eval_weights, dtype=float).reshape(-1)
            if eval_weights.shape != (M,):
                raise ValueError(f"eval_weights shape {eval_weights.shape} != ({M},)")

        dim = points.shape[1]
        log2pi = np.log(2.0 * np.pi)
        log_covfac = np.log(self.covfac_overall * covfac)
        logcovfac = log_covfac + (covalpha + self.covalpha_overall) * self.logdets
        N = self.data.shape[0]
        cf_eff = max(1.0, self.covfac_overall * covfac)
        n_to_get = min(int(self.nn * 10 * cf_eff ** (dim / 2.0)), N - 1)
        contrib_matrix = np.zeros(N)

        # Chunked tree query (same memory budget as __call__).
        mem_budget_bytes = 2 * 1024 ** 3
        chunk_size = max(1, int(mem_budget_bytes // (max(n_to_get, 1) * 8 * 2)))
        chunk_size = min(chunk_size, M)

        for start in range(0, M, chunk_size):
            end = min(start + chunk_size, M)
            pts_chunk = points[start:end]
            w_chunk = eval_weights[start:end]
            _dists, neighbors = self.tree.query(pts_chunk, n_to_get)
            for i in range(len(neighbors)):
                idx = neighbors[i]
                if len(idx) == 0 or w_chunk[i] == 0:
                    continue
                Us = self.Us[idx] / np.exp(0.5 * logcovfac[idx, None, None])
                devs = pts_chunk[i] - self.data[idx]
                devUs = np.einsum('ni,nij->nj', devs, Us)
                mahas = np.sum(np.square(devUs), axis=1)
                contribs = -0.5 * (dim * log2pi + mahas
                                    + logcovfac[idx] * dim
                                    + self.logdets[idx])
                # Each kernel's contribution carries its input weight w_k too,
                # since the weighted density is Sigma_k w_k K_k / N.
                contrib_matrix[idx] += w_chunk[i] * self.weights[idx] * np.exp(contribs)
            del neighbors, _dists

        s1 = float(np.sum(contrib_matrix))
        s2 = float(np.sum(contrib_matrix * contrib_matrix))
        if s2 <= 0:
            return 0.0
        return s1 * s1 / s2

    def set_log_bias_correction(self, b):
        """Set the multiplicative Laplace-bias correction. fhat_corrected(x) = fhat(x) * exp(-b),
        i.e. log fhat_corrected = log fhat - b. Estimate b from data via
        `estimate_log_bias_from_truth(...)` (when truth is known) or a leave-one-out
        scheme (TODO; the spatially-varying correction is not yet implemented)."""
        self.log_bias_correction = float(b)

    def estimate_log_bias_from_truth(self, samples_from_truth, truth_callable, fac=1.0):
        """Estimate b = median over samples drawn from truth of log(fhat(x)*fac / f_truth(x)).
        Setting `self.log_bias_correction = b` then makes the bias-corrected KDE unbiased
        in the mass-weighted-median sense (volume-weighted is shifted away from zero by -b).
        Only usable when `truth_callable` is known; for real applications see notes.
        """
        p_kde = np.atleast_1d(self(samples_from_truth)) * fac
        p_truth = truth_callable(samples_from_truth)
        valid = (p_kde > 0) & (p_truth > 0)
        return float(np.median(np.log(p_kde[valid]) - np.log(p_truth[valid])))

    def loo_density_at_data(self, covfac=1.0, covalpha=0.0, returnLog=False):
        """Leave-one-out KDE evaluation at the training data points.

        For each data point x_i, computes fhat(x_i) *excluding* the i-th kernel -
        which is the dominant self-contribution at the data point itself. The
        first-nearest-neighbor of any data point in the kdtree is itself (distance 0),
        so we drop the first column of the cached Mahalanobis matrix.

        Mainly useful for self-consistent bias estimation via Richardson extrapolation.
        """
        data_physical = self.data * np.asarray(self.scales)[None, :]
        # LOO needs a kNN cache (sorted by distance, self at index 0). Distance-based
        # ball-point queries don't preserve ordering, so use the legacy n_to_get path.
        # Ask for "all" neighbours up to a generous cap; the precompute time scales
        # linearly so we don't bother shrinking it.
        N = self.data.shape[0]
        cache = self.precompute_query(data_physical, n_to_get=min(self.nn * 10, N - 1))
        loo_cache = {
            'mahas': cache['mahas'][:, 1:],
            'neighbors': cache['neighbors'][:, 1:],
            'logdets_at_n': cache['logdets_at_n'][:, 1:],
            'M': cache['M'],
        }
        return self.eval_from_cache(loo_cache, covfac=covfac, covalpha=covalpha, returnLog=returnLog)

    def estimate_log_bias_richardson(self, ratio=0.9):
        """Self-consistent bias estimate via LOO + Richardson extrapolation.

        For a kernel of order p (Gaussian: p=2), the integrated Laplace bias scales
        as h^p. So if we build fhat_h with bandwidth h and fhat_(ratio*h) with bandwidth
        ratio*h, evaluated LOO at data points (which are samples from f),
            E[log fhat_h(X)] - E[log fhat_(ratio*h)(X)] ~ bias(h) - bias(ratio*h)
                                                  = bias(h) * (1 - ratio^p)
        With p=2, solve for bias(h):
            bias(h) ~ median_{x_i in data}(log fhat_LOO_h(x_i) - log fhat_LOO_(ratio*h)(x_i))
                      / (1 - ratio^2).

        h scales as sqrtcovfac in our parametrisation, so h_smaller = ratio*h_full means
        covfac_smaller = ratio^2*covfac_full. The smaller-bandwidth KDE is computed by
        re-evaluating the cache (no need to rebuild Sigma_i).

        Returns the value to feed to `set_log_bias_correction`. Doesn't require truth.

        Empirical caveat (2026-04-30 testing on N=2000 6D unit Gaussian):
        - ratio close to 1 (e.g. 0.9) is preferred - Richardson is exact only as
          ratio -> 1, and finite-ratio higher-order corrections drift the estimate.
          ratio=0.5 gave qualitatively wrong answers in our tests; ratio=0.9 was
          within ~13% of the truth-based estimator for adaptiveKDE.
        - Works less well for mockScipyKde (global-covariance) where the bias
          magnitude is smaller and the signal-to-noise of Richardson degrades.
        - For real applications, prefer ratio=0.9 and treat the estimate as
          approximate. A multi-ratio fit (ratios [0.95, 0.9, 0.85, 0.8] then
          fit bias vs (1-r^2)) would be more robust but is not implemented.
        """
        log_f_full = self.loo_density_at_data(covfac=1.0, returnLog=True)
        log_f_smaller = self.loo_density_at_data(covfac=ratio ** 2, returnLog=True)
        valid = np.isfinite(log_f_full) & np.isfinite(log_f_smaller)
        if not np.any(valid):
            return 0.0
        diff = np.median(log_f_full[valid] - log_f_smaller[valid])
        return float(diff / (1.0 - ratio ** 2))

    def evaluate_conditional(self, pointsIn, xvecIn, profile=False, covfac=1.0, covalpha=0.0):
        # p( v (pointsIn) | x (xvec) )
        assert self.data.shape[1] == 6, (
            "evaluate_conditional requires 6D [pos(3), vel(3)] data; "
            f"this KDE is {self.data.shape[1]}D (the position/velocity split "
            "and Schur-complement conditional are only built for d==6).")
        xvec = copy.deepcopy(xvecIn)
        assert len(xvec)==3 and pointsIn.shape[1]==3
        # first find the points we need to worry about in the tree:
        if profile:
            tim = timer()
        points = copy.deepcopy(pointsIn)
        if profile:
            tim.tick('copy data')
        
        if profile:
            tim.tick('scale data')
            
        pofzgivenx = self.evaluate_marginal( xvec, returnUnsummed=True, covfac=covfac, covalpha=covalpha ) / self.evaluate_marginal( xvec, covfac=covfac, covalpha=covalpha )

        for i in range(3):
            points[:,i] = points[:,i]/self.scales[i+3]
            xvec[i] = xvec[i]/self.scales[i]


        pos = pofzgivenx > 0
        logpofz = np.log(pofzgivenx[pos])
        # this makes more sense if you're using all of the points anyway.

        logcovfac = np.log(self.covfac_overall * covfac) + (covalpha+self.covalpha_overall)*self.logdets
        Us = self.UsCond * 1.0/np.exp(0.5*logcovfac[:,None,None])
        devs = points[:,np.newaxis,:] - ( self.data[:,3:] + np.squeeze(np.matmul(self.onetwotwotwo[:,:,:], (xvec-self.data[:,:3])[:,:,np.newaxis])) )
        devUs = np.einsum('...i,...ij->...j', devs, Us)
        mahas = np.sum(np.square(devUs), axis=2)
        dim = points.shape[1]
        log2pi = np.log(2*np.pi)
        est = scipy.special.logsumexp( -0.5 * (dim * log2pi + mahas + dim*logcovfac + self.logdetsCond)[:,pos.flatten()] + logpofz, axis=1 )
        # est is a density in scaled-v space; pofzgivenx already sums to 1 over z, so the conditional integrates to 1 in scaled-v space (no /N). Convert to physical-v with jacfac_cond, not the full 6D jacfac.
        return np.exp(est)/self.jacfac_cond


#        neighbors = [np.arange( len(self.data) )]*len(points) 
#        #neighbors = self.tree.query_ball_point( points, 2*self.max_from_cov) # neighbors is a list of lists, one list per point (the indexes to data of the neighbors of that point).
#        #neighbors = self.tree.query_ball_point( points, self.overall_max_dist) # very conservative!
#        if profile:
#            tim.tick('query tree')
#        est = np.zeros( points.shape[0] ) # we need a density estimate at each of the input points. We will accumulate these here.
#        nneighbors = np.zeros(len(neighbors))
#        for i in range(len(neighbors)):
#            nneighbors[i] = len(neighbors[i])
#            if nneighbors[i]>0:
#                # neighbors[i] is a list of indexes for the neighbors of point i (where i is an index to points)
#                #distrs = self.normals[neighbors[i]]
#                Us = self.UsCond[neighbors[i]]
#                #pdb.set_trace()
#                devs = points[i,:] - ( self.data[neighbors[i],3:] + np.squeeze(np.matmul(self.onetwotwotwo[neighbors[i],:,:], (xvec-self.data[neighbors[i],:3])[:,:,np.newaxis])) ) #deviation of given velocities (points) from the conditional mean
#                devUs = np.einsum( 'ni,nij->nj', devs, Us )
#                mahas = np.sum(np.square(devUs), axis=1)
#
#                # Compute and broadcast scalar normalizers.
#                #dim = len(vals[0])
#                dim = points.shape[1]
#                log2pi = np.log(2 * np.pi)
#                est[i] = scipy.special.logsumexp( -0.5 * (dim * log2pi + mahas + self.logdetsCond[neighbors[i]]) )
#            else:
#                est[i] = -1000.0
#            
#
#
#            #for j in range(len(distrs)):
#                #est += distrs[j].pdf( points ) # there's no way this is right! We should only be contributing to est[i]!
#                #est[i] += distrs[j].pdf( points[i] )
#        if profile:
#            print("neighbors stats:", np.mean(nneighbors), np.percentile(nneighbors,[5,16,50,84,95]))
#            tim.tick('evaluate pdfs')
#            tim.report()
#        return (np.exp(est) / self.jacfac ) / len(self.normals)

    def draw_marginal(self, covfac=1.0, covalpha=0.0, size=1):
        assert self.data.shape[1] == 6, (
            "draw_marginal returns the position (first 3) marginal and "
            f"requires 6D data; this KDE is {self.data.shape[1]}D.")
        points = self.draw(size=size, covfac=covfac, covalpha=covalpha)
        ret = points[:,:3]
        return ret
    
    #  returnUnsummed here is ~equivalent to evaluating p(z|x), namely the probability that each provided pointsIn is associated with the zth element of the original data.
    def evaluate_marginal(self, pointsIn, profile=False, returnUnsummed=False, covfac=1.0, covalpha=0.0):
        # first find the points we need to worry about in the tree:
        assert self.data.shape[1] == 6, (
            "evaluate_marginal returns the position (first 3) marginal and "
            f"requires 6D data (UsMarg is only built for d==6); this KDE is "
            f"{self.data.shape[1]}D.")
        if profile:
            tim = timer()
        points = np.array(copy.deepcopy(pointsIn))
        if profile:
            tim.tick('copy data')
        
        if len(points.shape)==2:
            assert points.shape[1] == 3
        else:
            assert len(points)==3
            points = points.reshape((1,3))

        jacfac = 1.0
        for i in range(points.shape[1]):
            points[:,i] = points[:,i]/self.scales[i]
            jacfac *= self.scales[i]
        if profile:
            tim.tick('scale data')
            

        # here we run into a bit of a problem because we don't have velocity info for the input points.
        # so unless this becomes a big problem let's just grab all of the points!
        neighbors = [np.arange( len(self.data) )]*len(points) 
        #neighbors = self.tree.query_ball_point( points, 2*self.max_from_cov) # neighbors is a list of lists, one list per point (the indexes to data of the neighbors of that point).
        #neighbors = self.tree.query_ball_point( points, self.overall_max_dist) # very conservative!
        if profile:
            tim.tick('query tree')
        est = np.zeros( points.shape[0] ) # we need a density estimate at each of the input points. We will accumulate these here.
        unsummed = np.zeros( (points.shape[0], len(self.data)) ) # for each point, record the contribution from every point in the dataset
        nneighbors = np.zeros(len(neighbors))
        dim = points.shape[1]
        log2pi = np.log(2 * np.pi)
        logcovfac = np.log(self.covfac_overall * covfac) + (covalpha+self.covalpha_overall)*self.logdets
        #logcovfac = np.log(covfac)
        for i in range(len(neighbors)):
            nneighbors[i] = len(neighbors[i])
            if nneighbors[i]>0:
                # neighbors[i] is a list of indexes for the neighbors of point i (where i is an index to points)
                #distrs = self.normals[neighbors[i]]

                # this stuff from https://gregorygundersen.com/blog/2020/12/12/group-multivariate-normal-pdf/
                Us = self.UsMarg[neighbors[i]] * 1.0/np.exp(0.5*logcovfac[neighbors[i],None,None]) #np.sqrt(covfac)
                devs = points[i,:] - self.data[neighbors[i],:3] # surprised this broadcasts(?) correctly
                devUs = np.einsum( 'ni,nij->nj', devs, Us )
                mahas = np.sum(np.square(devUs), axis=1)

                # Compute and broadcast scalar normalizers.
                #dim = len(vals[0])
                marg_contribs = (-0.5 * (dim * log2pi + mahas + dim*logcovfac[neighbors[i]] + self.logdetsMarg[neighbors[i]])
                                 + self.log_weights[neighbors[i]])
                est[i] = scipy.special.logsumexp( marg_contribs )
                # `unsummed` must include dim*logcovfac so that pofzgivenx = unsummed/sum(unsummed) reflects the same per-point bandwidth that est uses. Omitting it biased pofzgivenx by |Sigma_z|^(-d*covalpha/2) whenever covalpha != 0. Input weights enter here too so pofzgivenx respects them.
                unsummed[i, neighbors[i]] = np.exp( marg_contribs )
            else:
                est[i] = -1000.0
            


            #for j in range(len(distrs)):
                #est += distrs[j].pdf( points ) # there's no way this is right! We should only be contributing to est[i]!
                #est[i] += distrs[j].pdf( points[i] )
        if profile:
            print("neighbors stats:", np.mean(nneighbors), np.percentile(nneighbors,[5,16,50,84,95]))
            tim.tick('evaluate pdfs')
            tim.report()
        if returnUnsummed:
            return (unsummed/jacfac) / len(self.normals)
        ret = (np.exp(est) / jacfac ) / len(self.normals)
        if np.any(np.isnan(ret)):
            pdb.set_trace()
        return ret

    def draw(self, size=1, covfac=1.0, covalpha=0.0, cholesky=True, rng=None):
        # `rng` (numpy Generator) overrides the legacy global np.random state
        # for deterministic CV when callers seed it. Default falls back to
        # global np.random so existing callers are unaffected.
        # Kernel-selection probability ~ input weight (uniform if unweighted).
        # self.weights sums to N, so p = weights/N.
        wp = self.weights / self.weights.sum()
        uniform_w = np.allclose(self.weights, 1.0)
        if rng is None:
            inds = (np.random.choice(len(self.normals), size=size)
                    if uniform_w else
                    np.random.choice(len(self.normals), size=size, p=wp))
            rands = np.random.normal(loc=0, scale=1, size=size * self.data.shape[1]).reshape((size, self.data.shape[1]))
        else:
            inds = (rng.integers(0, len(self.normals), size=size)
                    if uniform_w else
                    rng.choice(len(self.normals), size=size, p=wp))
            rands = rng.standard_normal(size=(size, self.data.shape[1]))
        points = np.zeros( ( size, self.data.shape[1]) )
        log_covfac_this = np.log(covfac*self.covfac_overall) +(covalpha+self.covalpha_overall) * self.logdets
        covfac_this = np.exp(log_covfac_this)
        if cholesky:
            points = self.data[inds] + np.einsum( 'nij,nj->ni', np.sqrt(covfac_this[inds,None,None])*self.choleskys[inds], rands )
        else:
            for i in range(size):
                points[i,:] = scipy.stats.multivariate_normal( self.data[inds[i],:], covfac_this[inds[i]]*self.covariances[inds[i],:,:] ).rvs(size=1)

#                def comp(k):
#                    #return self.data[inds[i]] + np.dot( np.sqrt(covfac)*self.choleskys[inds[i]], np.random.normal(size=6) )
#                    return self.data[inds[i]] + np.einsum( 'ij,j', np.sqrt(covfac)*self.choleskys[inds[i]], np.random.normal(size=6) )
#
#                dbgs = np.zeros( (size, 6) )
#                for ii in range(size):
#                    dbgs[ii,:] = comp(ii)
#
#    
#
#                pdb.set_trace()
#                #print(covfac*self.covariances[inds[i],:,:])
#                #print(np.dot(np.sqrt(covfac)*self.choleskys[inds[i]], np.transpose(np.sqrt(covfac)*self.choleskys[inds[i]]) ) )
#

        for j in range(self.data.shape[1]):
            points[:,j] *= self.scales[j]
        
        return points

    def draw_conditional_on_x(self, xvecIn, size=1, covfac=1.0, covalpha=0.0):
        assert self.data.shape[1] == 6, (
            "draw_conditional_on_x draws velocities given a position and "
            f"requires 6D data; this KDE is {self.data.shape[1]}D.")
        if self.data.shape[1]==6:
            pass
        else:
            assert False # this code is written for 6D case, where we're conditioning on the first 3 elements.

        # For each normal distribution that makes up the KDE estimate, we have to estimate the chance this position xvec is associated with that normal. Then draw one in proportion to that probability, then draw from the appropriate conditional normal distribution.
        xvec = copy.deepcopy(np.array(xvecIn).reshape((1,3)))
        log_covfac_this = np.log(covfac*self.covfac_overall) +(covalpha+self.covalpha_overall) * self.logdets
        covfac_this = np.exp(log_covfac_this)

        pofzgivenx = self.evaluate_marginal( xvec, returnUnsummed=True, covfac=covfac, covalpha=covalpha ) / self.evaluate_marginal( xvec, covfac=covfac, covalpha=covalpha )

        nans = np.isnan(pofzgivenx)
        if np.any(nans):
            pofzgivenx[nans] = 0.0
            print("Replacing", np.sum(nans), "conditional probabilities of assignment with zeros")
            pdb.set_trace()

        points = np.zeros( (size, 3) )
        renorm = np.sum(pofzgivenx.flatten())
        pofzgivenx = pofzgivenx / renorm # occasionally doesn't sum to 1 owing to roundoff error.
        try:
            inds = np.random.choice( len(self.normals), replace=True, p=pofzgivenx.flatten(), size=size)
        except:
            pdb.set_trace()

        # didn't need this when evaluating marginal because marginal already does the scaling. Need it here because now we're working in the scaled space.
        for i in range(3):
            xvec[:,i] = xvec[:,i]/self.scales[i]
        for i in range(size):
            points[i,:] = scipy.stats.multivariate_normal( self.data[inds[i],3:] + np.matmul( self.onetwotwotwo[inds[i]], (xvec-self.data[inds[i],:3]).T).flatten(), np.eye(3)*1.0e-5 + covfac_this[inds[i]]*self.covcond[inds[i]], allow_singular=True ).rvs()

        for j in range( 3 ):
            points[:,j] *= self.scales[j+3]

        return points


# WIP
# use scipy's simple KDE, but wrap it in language compatible with the adaptiveKDE.
class gaussianKDEWrapper:
    def __init__(self, data, weights=None):
        # scipy.stats.gaussian_kde supports per-point weights natively.
        self.weights = None if weights is None else np.asarray(weights, float).reshape(-1)
        self.kde = scipy.stats.gaussian_kde(data.T, weights=self.weights)
    def __call__(self,points, covfac=1.0, show_contribs=False):
        return self.kde(points.T)
    def draw(self, size, covfac=1.0):
        return self.kde.resample(size).T
    def data_side_neff(self, points, eval_weights=None, covfac=1.0):
        """Kish ESS of training-particle contributions to Sigma_i w_i * fhat(points[i]).
        See adaptiveKDE.data_side_neff for the metric definition.
        scipy gaussian_kde has a global covariance, so all training particles
        share the same kernel shape; the contributions are explicit Mahalanobis
        distances. O(M * N) where M = #queries, N = #training particles."""
        data = self.kde.dataset.T   # (N, d)
        inv_cov = np.asarray(self.kde.inv_cov)
        points = np.atleast_2d(np.asarray(points))
        M, d = points.shape
        if eval_weights is None:
            eval_weights = np.ones(M)
        else:
            eval_weights = np.asarray(eval_weights, dtype=float).reshape(-1)
        # Per-training-point input weights (scipy stores them, normalised to
        # sum 1) scale each kernel's contribution to the weighted sum.
        kw = (self.kde.weights if self.weights is not None
              else np.ones(data.shape[0]))
        # Chunked accumulation to bound memory at the M x N kernel-matrix step.
        chunk_size = max(1, int(2 * 1024**3 // (data.shape[0] * 8 * 2)))
        contrib = np.zeros(data.shape[0])
        for start in range(0, M, chunk_size):
            end = min(start + chunk_size, M)
            pts = points[start:end]
            w = eval_weights[start:end]
            # K(x_i - data[k]) ~ exp(-1/2 (x_i - data[k])^T inv_cov (x_i - data[k]))
            diffs = pts[:, None, :] - data[None, :, :]    # (chunk, N, d)
            mahas = np.einsum('mni,ij,mnj->mn', diffs, inv_cov, diffs)
            kernel = np.exp(-0.5 * mahas)                  # normalization cancels in the ratio
            contrib += kw * np.sum(w[:, None] * kernel, axis=0)
        s1 = float(np.sum(contrib)); s2 = float(np.sum(contrib * contrib))
        return s1 * s1 / s2 if s2 > 0 else 0.0


def _detect_narrow_v_sigma(data, axis_offset_start=3, num_v_axes=3, narrow_weight_min=0.05):
    """Per-axis 1D GMM with biased narrow init. Returns the narrow-component sigma
    (in physical units) when its weight >= narrow_weight_min, else falls back
    to the empirical std. Used by cvAdaptiveKDE's `scalings='narrow'` and
    cvGaussianKDE's `scalings_grid='narrow'` to identify cold-component scales
    on multi-scale data without overfitting on monomodal data.
    """
    from sklearn.mixture import GaussianMixture
    out = []
    for axis_offset in range(num_v_axes):
        col = data[:, axis_offset_start + axis_offset].reshape(-1, 1)
        global_std = float(np.std(col))
        chosen = global_std
        try:
            gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0, means_init=np.array([[0.0], [0.0]]), weights_init=np.array([0.2, 0.8]), precisions_init=np.array([[[1.0e4 / global_std**2]], [[1.0 / global_std**2]]]), n_init=1)
            gmm.fit(col)
            sigmas = np.sqrt(gmm.covariances_.flatten())
            i_narrow = int(np.argmin(sigmas))
            if gmm.weights_[i_narrow] >= narrow_weight_min:
                chosen = max(float(sigmas[i_narrow]), 0.01 * global_std)
        except Exception:
            pass
        out.append(chosen)
    return out


class scaledGaussianKDE:
    """scipy.stats.gaussian_kde with per-axis covariance scaling.

    Kernel covariance is `bw^2 * D * Sigma_data * D` where D = diag(scales). Setting
    scales=ones recovers plain scipy.gaussian_kde. Setting scales[i] < 1 narrows
    the kernel along axis i (independent of other axes), allowing per-axis
    bandwidth flexibility that scipy.gaussian_kde alone doesn't support.

    Implementation: build standard scipy.gaussian_kde, then directly override
    `kde.covariance`, `kde.inv_cov`, `_norm_factor`, and `cho_cov` (used by
    resample) with the rescaled covariance. scipy's vectorised evaluation is
    unchanged - same ~N*M flops per call.

    Note: simply rescaling the data (e.g. `data /= scales` then build scipy KDE)
    is mathematically a no-op for the resulting density estimate (the change
    of variables jacobian undoes the rescaling). The covariance override is
    what gives genuine per-axis bandwidth flexibility.
    """
    def __init__(self, data, scales=None, bw_method=None, weights=None):
        self.weights = None if weights is None else np.asarray(weights, float).reshape(-1)
        self.kde = scipy.stats.gaussian_kde(data.T, bw_method=bw_method, weights=self.weights)
        self.scales = np.ones(data.shape[1]) if scales is None else np.asarray(scales, dtype=float)
        if scales is not None:
            D = np.diag(self.scales)
            # Override _data_covariance and _data_cho_cov, then call scipy's
            # _compute_covariance to regenerate derived attributes (covariance,
            # cho_cov, log_det) consistently. Avoids touching read-only
            # cached_property attributes (inv_cov) directly.
            new_data_cov = D @ self.kde._data_covariance @ D
            self.kde._data_covariance = new_data_cov
            self.kde._data_cho_cov = np.linalg.cholesky(new_data_cov)
            self.kde._compute_covariance()

    def __call__(self, points, covfac=1.0, show_contribs=False):
        return self.kde(points.T)

    def draw(self, size, covfac=1.0):
        return self.kde.resample(size).T


class cvGaussianKDE:
    """scipy.stats.gaussian_kde with CV-selected bandwidth and (optional) per-axis
    covariance scaling.

    Kernel covariance is `bw^2 * D * Sigma_data * D`. With `scalings_grid=None` the
    only knob is scalar bandwidth (D=I); supplying `scalings_grid` lets CV pick
    among per-axis scalings options (None=unit, 'auto'=axis-equalising, 'narrow'
    =GMM narrow-component detection on velocity axes, or explicit dimensionless
    multipliers). Mirrors cvAdaptiveKDE's API for fair comparison: nfolds,
    bw_range (log10 around Scott's rule), bootstrap RR, optional ROI focus,
    and rate-weighted CV via `weight_fn`.

    Picks via the same ISE-related score:
        <fhat(X_test) * w(X_test)^2> - 1/2 * <fhat(X_samp) * w(X_samp)^2>
    """
    def __init__(self, dataIn, nfolds=5, nbw=11, bw_range=(-1.0, 1.0), nboot=None, roi=1.0, roiThresh=200, roiCenter=None, weight_fn=None, selector=np.mean, reg=1.0, scalings_grid=None, random_state=None, neff_eval_points=None, neff_floor=None, roi6=None, roiCenter6=None, weights=None):
        data = copy.deepcopy(dataIn)
        N, d = data.shape
        # Per-training-point input weights (e.g. Gaia ISO production weights),
        # distinct from `weight_fn` (the rate-weighting of the CV objective).
        # Normalised to sum to N. Carried into the fold train KDEs and the
        # weighted test-fold score, and into the final estimator build.
        if weights is None:
            w_all = np.ones(N)
        else:
            w_all = np.asarray(weights, dtype=float).reshape(-1)
            if w_all.shape != (N,):
                raise ValueError(f"weights shape {w_all.shape} != ({N},)")
            if np.any(w_all < 0) or not np.all(np.isfinite(w_all)):
                raise ValueError("weights must be non-negative and finite")
            w_all = w_all / w_all.sum() * N
        self.weights = w_all
        kf = KFold(n_splits=nfolds, shuffle=True, random_state=random_state)

        scott_factor = N ** (-1.0 / (d + 4))
        bw_multipliers = np.logspace(bw_range[0], bw_range[1], nbw)
        bws = scott_factor * bw_multipliers

        # Resolve scalings grid. Each entry is either None / 'auto' / 'narrow'
        # / explicit list. Returns a list of (dimensionless multipliers, label).
        # 'auto' equalises kernel sigma across axes (D[i] = 1/sqrtSigma_data[i,i]).
        # 'narrow' equalises positions and uses sigma_narrow_v for velocities.
        sigma_axis = np.sqrt(np.diag(np.cov(data.T)))
        def _resolve(sc_input):
            if sc_input is None:
                return np.ones(d), 'unit'
            if isinstance(sc_input, str) and sc_input == 'auto':
                return 1.0 / sigma_axis, 'auto'
            if isinstance(sc_input, str) and sc_input == 'narrow':
                if d != 6:
                    return 1.0 / sigma_axis, 'narrow(fallback to auto, d!=6)'
                v_narrow = _detect_narrow_v_sigma(data)
                D = np.empty(6)
                D[:3] = 1.0 / sigma_axis[:3]                    # positions: 'auto'
                D[3:] = np.asarray(v_narrow) / sigma_axis[3:]    # velocities: sigma_narrow ratio
                return D, 'narrow'
            if isinstance(sc_input, str) and sc_input == 'narrow_local':
                # Position-local subset around roiCenter; narrow detector on
                # subset's velocities. Captures stream-aligned sigma at the
                # encounter target (essential for ring/stream data).
                if d != 6:
                    return 1.0 / sigma_axis, 'narrow_local(fallback to auto, d!=6)'
                rc = roiCenter if roiCenter is not None else [0.0, 0.0, 0.0]
                rsq = ((data[:, 0] - rc[0]) ** 2
                       + (data[:, 1] - rc[1]) ** 2
                       + (data[:, 2] - rc[2]) ** 2)
                sortr = np.argsort(rsq)
                # See cvAdaptiveKDE._resolve_scalings for the K_LOCAL rationale.
                K_LOCAL = max(300, data.shape[0] // 10)
                K_LOCAL = min(K_LOCAL, data.shape[0] - 1)
                subset = data[sortr[:K_LOCAL]]
                pos_std_local = subset[:, :3].std(axis=0)
                # Floor positions at 1% of global std to avoid degenerate scales
                pos_std_local = np.maximum(pos_std_local, 0.01 * sigma_axis[:3])
                v_narrow = np.asarray(_detect_narrow_v_sigma(subset))
                D = np.empty(6)
                D[:3] = pos_std_local / sigma_axis[:3]
                D[3:] = v_narrow / sigma_axis[3:]
                return D, f'narrow_local(K={K_LOCAL})'
            arr = np.asarray(sc_input, dtype=float)
            return arr, f'custom({arr.round(2)})'

        if scalings_grid is None:
            scalings_grid = [None]
        resolved = [_resolve(s) for s in scalings_grid]
        n_sc = len(resolved)

        nbootThis = nboot if nboot is not None else (200 if roi is None else 2 * N)

        # 6D ROI mode (mirrors cvAdaptiveKDE): per-(scaling, fold) ball under
        # the same per-axis metric the kernel uses (`D_vec`), ratcheted to
        # contain `roi6` particles. Computed inside the sc_idx loop below.
        roi6_active = (roi6 is not None and d == 6)
        roi6_center_arr = None
        if roi6_active:
            if roiCenter6 is not None:
                roi6_center_arr = np.asarray(roiCenter6, dtype=float)
            else:
                rc_pos = roiCenter if roiCenter is not None else [0.0, 0.0, 0.0]
                roi6_center_arr = np.array(list(rc_pos) + [0.0, 0.0, 0.0], dtype=float)
            roiThis = None     # disable 3D ROI codepath
            print(f"cvGaussianKDE 6D ROI: target K={int(roi6)} per (fold, scaling)")
        else:
            # Standard 3D ROI setup
            roiThis = roi
            if roiThis is not None:
                if roiCenter is None:
                    roiCenter = [0.0, 0.0, 0.0]
                rsq = ((data[:, 0] - roiCenter[0]) ** 2
                       + (data[:, 1] - roiCenter[1]) ** 2
                       + (data[:, 2] - roiCenter[2]) ** 2)
                selecroi = rsq < roi * roi
                if np.sum(selecroi) < roiThresh:
                    sortr = np.argsort(rsq)
                    roiThis = np.sqrt(rsq[sortr[roiThresh]])
                    selecroi = rsq < roiThis * roiThis

        def log_wsq(points):
            if weight_fn is None:
                return 0.0
            w = np.asarray(weight_fn(points), dtype=float)
            with np.errstate(divide='ignore'):
                return np.where(w > 0, 2.0 * np.log(w), -np.inf)

        # scores axes: (n_scalings, nbw, nfolds)
        scores = np.zeros((n_sc, nbw, nfolds))
        signs = np.zeros(scores.shape)
        # N_eff floor (mirrors cvAdaptiveKDE): hard structural-smoothness gate.
        # For scipy KDE the per-eval N_eff is `(Sigma exp(-1/2*M_ij))^2 / Sigma exp(-M_ij)`
        # where M_ij is the Mahalanobis distance under the kernel covariance.
        neffs = np.full(scores.shape, np.nan)
        neff_eval_points = (None if neff_eval_points is None
                            else np.asarray(neff_eval_points))
        for i, (train_idx, test_idx) in enumerate(tqdm(kf.split(data), total=nfolds, desc='kfold')):
            train = data[train_idx, :]

            for sc_idx, (D_vec, _label) in enumerate(resolved):
                # Per-(fold, scaling) 6D ROI under D_vec - same metric the
                # kernel covariance uses, so ROI and kernel are coordinated.
                if roi6_active:
                    diffs = (data - roi6_center_arr) * D_vec
                    rsq6 = np.sum(diffs ** 2, axis=1)
                    sortr = np.argsort(rsq6)
                    K = min(int(roi6), data.shape[0])
                    roi6_thresh_sq = rsq6[sortr[K - 1]]
                    selecroi = rsq6 <= roi6_thresh_sq

                test_idx_this = test_idx
                if roi6_active or roiThis is not None:
                    tia = np.zeros(N, dtype=bool)
                    tia[test_idx] = True
                    test_idx_this = tia & selecroi
                test = data[test_idx_this, :]
                if len(test) == 0:
                    scores[sc_idx, :, i] = -1.e30
                    signs[sc_idx, :, i] = -1
                    continue
                log_wsq_test = log_wsq(test)
                w_train = w_all[train_idx]
                # Weighted test average: each test point carries its input weight.
                log_w_test = np.log(np.maximum(w_all[test_idx_this], 1e-300))
                log_w_test_norm = scipy.special.logsumexp(log_w_test)

                for k in range(nbw):
                    if np.allclose(D_vec, 1.0):
                        kde = scipy.stats.gaussian_kde(train.T, bw_method=bws[k], weights=w_train)
                    else:
                        kde = scaledGaussianKDE(train, scales=D_vec, bw_method=bws[k], weights=w_train).kde
                    with np.errstate(divide='ignore'):
                        log_test = np.log(np.maximum(kde(test.T), np.finfo(float).tiny))
                    score = (scipy.special.logsumexp(log_test + log_wsq_test + log_w_test)
                             - log_w_test_norm)

                    samples = kde.resample(nbootThis).T
                    if roi6_active:
                        sd = (samples - roi6_center_arr) * D_vec
                        srsq = np.sum(sd ** 2, axis=1)
                        samples = samples[srsq <= roi6_thresh_sq, :]
                        if len(samples) == 0:
                            scores[sc_idx, k, i] = -1.e30
                            signs[sc_idx, k, i] = -1
                            continue
                    elif roiThis is not None:
                        sample_r = ((samples[:, 0] - roiCenter[0]) ** 2
                                    + (samples[:, 1] - roiCenter[1]) ** 2
                                    + (samples[:, 2] - roiCenter[2]) ** 2)
                        samples = samples[sample_r < roiThis * roiThis, :]
                        if len(samples) == 0:
                            scores[sc_idx, k, i] = -1.e30
                            signs[sc_idx, k, i] = -1
                            continue
                    with np.errstate(divide='ignore'):
                        log_samples = np.log(np.maximum(kde(samples.T), np.finfo(float).tiny))
                    log_wsq_samp = log_wsq(samples)
                    RR = scipy.special.logsumexp(log_samples + log_wsq_samp) - np.log(len(samples))

                    the_score, the_sign = scipy.special.logsumexp([score, RR], b=[1.0 * reg, -0.5], return_sign=True)
                    scores[sc_idx, k, i] = the_score
                    signs[sc_idx, k, i] = the_sign

                    if neff_eval_points is not None:
                        # Per-eval N_eff at the user-supplied eval points.
                        cov_inv = kde.inv_cov
                        train_arr = kde.dataset.T
                        a = np.sum(neff_eval_points @ cov_inv * neff_eval_points, axis=1)
                        c = np.sum(train_arr @ cov_inv * train_arr, axis=1)
                        b = neff_eval_points @ cov_inv @ train_arr.T
                        mahas = a[:, None] + c[None, :] - 2.0 * b
                        log_kernel = -0.5 * mahas
                        log_sum = scipy.special.logsumexp(log_kernel, axis=1)
                        log_sum_sq = scipy.special.logsumexp(2.0 * log_kernel, axis=1)
                        neff_arr = np.exp(2.0 * log_sum - log_sum_sq)
                        neffs[sc_idx, k, i] = float(np.median(neff_arr))

        avg_scores = selector(signs * np.exp(scores), axis=-1)

        # Dual-pick: rate (unconstrained) and shape (floor-constrained).
        rate_best = np.unravel_index(np.nanargmax(avg_scores), avg_scores.shape)
        floor_active = (neff_eval_points is not None
                        and neff_floor is not None
                        and np.any(np.isfinite(neffs)))
        if floor_active:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                neff_med_grid = np.nanmedian(neffs, axis=-1)
            eligible = neff_med_grid >= neff_floor
            n_eligible = int(np.sum(eligible))
            if n_eligible == 0:
                print(f"cvGaussianKDE N_eff floor: NO grid entry meets "
                      f"floor={neff_floor:.0f} (max={np.nanmax(neff_med_grid):.1f}); "
                      f"shape-pick falls back to unconstrained.")
                shape_avg_scores = avg_scores
            else:
                shape_avg_scores = np.where(eligible, avg_scores, -np.inf)
        else:
            shape_avg_scores = avg_scores
            neff_med_grid = None

        shape_best = np.unravel_index(np.nanargmax(shape_avg_scores), shape_avg_scores.shape)
        # Asymptotic bandwidth correction: CV's training fold has N*(k-1)/k
        # data points; the AMISE-optimal bandwidth scales as h ~ N^(-1/(d+4)).
        # `bw` here is a LINEAR bandwidth factor (kernel covariance is
        # bw^2*D*Sigma_data*D, see scaledGaussianKDE), so the linear-scale
        # correction is (k/(k-1))^(-1/(d+4)). (Earlier this used
        # (k/(k-1))^(-2/(d+4)) - that's the COVARIANCE-scale formula and
        # over-shrunk linear bw by a power of 2.)
        asymcorr = (float(nfolds) / (nfolds - 1.0)) ** (-1.0 / (4.0 + d))

        def _build_at(idx):
            sc, k = idx
            D_v, _label = resolved[sc]
            bw = bws[k] * asymcorr
            # Always wrap in scaledGaussianKDE (which handles the (n, d) ->
            # scipy's (d, n) transpose internally and exposes a `.kde` for
            # downstream tools that need the underlying scipy.gaussian_kde).
            return scaledGaussianKDE(data, scales=D_v, bw_method=bw, weights=(self.weights if not np.allclose(self.weights, 1.0) else None))

        # Stash everything pick_at_floor needs (mirrors cvAdaptiveKDE).
        self._build_at = _build_at
        self._kde_cache = {}
        self.neff_med_grid = neff_med_grid if floor_active else None

        if rate_best == shape_best:
            self.kde_shape = _build_at(shape_best)
            self.kde_rate = self.kde_shape
        else:
            self.kde_shape = _build_at(shape_best)
            self.kde_rate = _build_at(rate_best)
        self._kde_cache[0.0] = self.kde_rate
        if floor_active:
            self._kde_cache[float(neff_floor)] = self.kde_shape
        # Backwards-compat alias: callers that did `self.kde(points.T)` need
        # the inner scipy gaussian_kde; callers that did `self.kde(points)`
        # need the scaledGaussianKDE wrapper. The wrapper IS callable on
        # (n, d) and exposes the inner scipy KDE at `.kde`, so we set
        # self.kde to the wrapper (consistent with the rest of the API).
        self.kde = self.kde_shape

        # Expose shape-pick metadata in the existing attribute names.
        sc_best, k_best = shape_best
        D_best, label_best = resolved[sc_best]
        self.bw = bws[k_best] * asymcorr
        self.scales = D_best
        self.scales_label = label_best
        self.avg_scores = avg_scores
        self.bws = bws
        self.bw_multipliers = bw_multipliers
        self.scott_factor = scott_factor
        self.best = shape_best
        self.rate_best = rate_best
        self.shape_best = shape_best
        self.scalings_labels = [lbl for (_d, lbl) in resolved]
        self.neffs = neffs
        self.neff_floor = neff_floor
        self.neff_floor_active = bool(floor_active)
        if floor_active:
            self.neff_n_eligible = int(np.sum(neff_med_grid >= neff_floor))
            self.neff_n_total = int(np.sum(np.isfinite(neff_med_grid)))
            self.neff_med_at_pick = float(neff_med_grid[shape_best])
            self.neff_med_at_rate_pick = float(neff_med_grid[rate_best])
        else:
            self.neff_n_eligible = None
            self.neff_n_total = None
            self.neff_med_at_pick = None
            self.neff_med_at_rate_pick = None
        print(f"cvGaussianKDE: shape bw={self.bw:.3g} "
              f"(Scott={scott_factor:.3g}, multiplier={bw_multipliers[k_best]:.3g}), "
              f"scalings={label_best}; "
              f"rate bw={bws[rate_best[1]] * asymcorr:.3g}, "
              f"scalings={resolved[rate_best[0]][1]}")

    def typical_h_scaled(self):
        """For cvGaussianKDE the kernel covariance is `bw^2 * D * Sigma_data * D`.
        In the scaled basis where `data*D` has roughly unit per-axis
        variance, the per-axis kernel sigma is just `bw`. So a scalar `bw` is
        the right `h_scaled` summary."""
        return float(self.bw)

    def floor_for_dim(self, d_m, target_output_neff=60.0):
        """See `cvAdaptiveKDE.floor_for_dim`."""
        h = self.typical_h_scaled()
        return float(target_output_neff) * h ** (6.0 - float(d_m))

    def pick_for_dim(self, d_m, target_output_neff=60.0):
        """See `cvAdaptiveKDE.pick_for_dim`. d_m=0 is hard-pinned to floor=0."""
        if d_m <= 0:
            return self.pick_at_floor(0.0)
        return self.pick_at_floor(self.floor_for_dim(d_m, target_output_neff))

    def pick_at_floor(self, neff_min):
        """Mirror of `cvAdaptiveKDE.pick_at_floor`. Returns the wrapper KDE
        built at the score-grid argmax subject to median N_eff >= neff_min."""
        if neff_min is None or neff_min <= 0.0:
            return self.kde_rate
        key = float(neff_min)
        if key in self._kde_cache:
            return self._kde_cache[key]
        if self.neff_med_grid is None:
            self._kde_cache[key] = self.kde_rate
            return self.kde_rate
        eligible = self.neff_med_grid >= key
        if not np.any(eligible):
            self._kde_cache[key] = self.kde_rate
            return self.kde_rate
        masked = np.where(eligible, self.avg_scores, -np.inf)
        # Score-collapse fallback (see cvAdaptiveKDE.pick_at_floor for the
        # full rationale): if the floor knocked out all the meaningful picks
        # in the CV score landscape, the argmax of `masked` would pick noise.
        # Detect this via `best_eligible < 0.1 x best_unconstrained` and fall
        # back to the rate-pick instead.
        best_eligible = float(np.nanmax(masked))
        best_unconstrained = float(np.nanmax(np.where(np.isfinite(self.avg_scores), self.avg_scores, -np.inf)))
        if (best_unconstrained > 0
                and best_eligible < 0.1 * best_unconstrained):
            self._kde_cache[key] = self.kde_rate
            return self.kde_rate
        idx = np.unravel_index(np.nanargmax(masked), masked.shape)
        kde = self._build_at(idx)
        self._kde_cache[key] = kde
        return kde

    def __call__(self, points, covfac=1.0, show_contribs=False):
        # self.kde is a scaledGaussianKDE wrapper that takes (n, d) directly.
        return self.kde(points)

    def draw(self, size, covfac=1.0):
        return self.kde.draw(size)

class mockScipyKde(adaptiveKDE):
    def __init__(self, dataIn, scalings=None, nn=63, covfac=1.0, covalpha=0.0, profile=False, weights=None):
        data = copy.deepcopy(dataIn)
        # Input weights (see adaptiveKDE.__init__): normalised to sum to N so
        # the inherited weighted kernel-sum machinery works and weights=None
        # (uniform) is unchanged.
        _N = data.shape[0]
        if weights is None:
            self.weights = np.ones(_N)
        else:
            _w = np.asarray(weights, dtype=float).reshape(-1)
            if _w.shape != (_N,):
                raise ValueError(f"weights shape {_w.shape} != ({_N},)")
            self.weights = _w / _w.sum() * _N
        self.log_weights = np.log(np.maximum(self.weights, 1e-300))
        if scalings is None:
            self.scales = np.ones(data.shape[1])
        else:
            self.scales = scalings[:]
            assert len(self.scales)==data.shape[1]
        self.jacfac = 1.0
        self.covfac_overall = covfac * float(data.shape[0])**(-2.0/(data.shape[1]+4))  # instead of  covfac use the standard thing...
        print("cov factor from scotts rule: ", self.covfac_overall)
        self.covalpha_overall= covalpha
        self.log_bias_correction = 0.0
        if profile:
            tim = timer()
        for i in range(len(self.scales)):
            data[:,i] = data[:,i]/self.scales[i]
            self.jacfac *= self.scales[i] # kde returns density per unit x/scale, so to get density in x^-1... returned_dens = (scale/x) => dens_phys = (1/x) = returned_dens/scale
        self.jacfac_marg = float(np.prod(self.scales[:3])) if len(self.scales) >= 6 else self.jacfac
        self.jacfac_cond = float(np.prod(self.scales[3:6])) if len(self.scales) >= 6 else 1.0
        self.data = data[:,:]
        if profile:
            tim.tick('rescale data')
        try:
            self.tree = scipy.spatial.KDTree( data, leafsize=8, compact_nodes=True, balanced_tree=True )
        except:
            pdb.set_trace()
        if profile:
            tim.tick('build tree')
        self.nn = nn
        
        neighbor_dist, neighbor_inds = self.tree.query( data, k=self.nn ) # n x m -> n x k.  Don't think we need the distances for anything.
        if profile:
            tim.tick('query tree')

        self.neighbor_inds = neighbor_inds
        self.max_neighbor_dist = neighbor_dist[:,-1] #np.max(neighbor_dist, axis=1) # distance to each points nn'th nearest neighbor
        self.overall_max_dist = np.max(self.max_neighbor_dist) # maximum in the whole sample, i.e. distance associated with "the most isolated point"

        # might have to do a for loop here?
        self.covariances = np.zeros( (data.shape[0], data.shape[1], data.shape[1]) ) # n x m x m 
        self.normals = np.zeros( data.shape[0], dtype=object)
        cov = np.cov( data.T )
        self.covariances[:,:,:] = cov[None,:,:]
#        for i in range(data.shape[0]):
#            selec = data[ self.neighbor_inds[i,:], :] # the nn x m datapoints composing the nearest neighbors of this point
#            selec = np.concatenate( [selec, data[i,:].reshape(1,data.shape[1])], axis=0) 
#            self.covariances[i, :,:] = np.cov(selec.T) #self.covfac*np.cov( selec.T )
#
#            #self.normals[i] = scipy.stats.multivariate_normal( data[i,:], self.covfac*self.covariances[:,:,i] )

        self.choleskys = np.linalg.cholesky(self.covariances)# + np.eye(data.shape[1])*1.0e-8) 
        vals,vecs = np.linalg.eigh(self.covariances) # vals each scale as covfac. vals ~ (..., M), where covariances is shaped as N x M x M
        self.logdets = np.sum(np.log(vals), axis=1) 
        valsinvs = 1.0/vals  # valsinvs each scale as 1/covfac
        self.Us = vecs * np.sqrt(valsinvs)[:,None] # careful - broadcasting going on here. vals ~ N x M, so valsinvs ~ N x 1 x M?? vecs ~ N x M x M. So Us just scales as 1/sqrt(covfac)
        self.max_from_cov = np.sqrt(np.max(vals))

        if data.shape[1]==6:
            # split covariances up so we can draw from conditional distributions.
            xcov = self.covariances[:, :3, :3] # Sigma22 in the wiki notation, since the x-part is generally going to be the part we've conditioned on I think. -- scales as covfac
            vcov = self.covariances[:, 3:, 3:] # Sigma11 -- scales as covfac
            crosscov = self.covariances[ :, 3:, :3] # Sigma21 I think. It's Sigma12 in the original configuration, but we've  -- scales as covfac

            self.onetwotwotwo = np.matmul( crosscov, np.linalg.inv(xcov)) # Sigma12 * Sigma22inv -- scales as covfac^0

            self.covcond = vcov - np.matmul( self.onetwotwotwo, np.transpose(crosscov,axes=[0,2,1]) ) # scales as covfac
            # set up stuff to quickly evaluate conditional p(v|x, z) (where  z is the data particle)
            vals,vecs = np.linalg.eigh(self.covcond)
            if not np.all(vals>=0):
                #pdb.set_trace()
                #print("Warning: clipped negative values in eigenvalues for conditional gaussians", np.sum(vals<1.0e-13) )
                vals = np.clip(vals,1.0e-13,None)

            self.logdetsCond = np.sum(np.log(vals), axis=1) # scales as covfac^d
            valsinvs = 1.0/vals
            self.UsCond = vecs * np.sqrt(valsinvs)[:,None] # scales as 1/sqrt(covfac)




            # set up stuff to quickly evaluate marginal int p(x,v)dv = p(x)
            vals,vecs = np.linalg.eigh(xcov)
            self.logdetsMarg = np.sum(np.log(vals), axis=1)
            valsinvs = 1.0/vals
            self.UsMarg = vecs * np.sqrt(valsinvs)[:,None]

        if profile:
            tim.tick('build normals')

            tim.report()
        


def cvkde_evaluate_inner(args):
    """Per-query-point density evaluation entry, called by the ThreadPoolExecutor
    in `adaptiveKDE.__call__`.

    Args (tuple, packed by the caller - see __call__ for the assembly):
      selfUs:      (N, d, d) full per-kernel scaling Cholesky-like factors
      logcovfac:   (N,) per-kernel log-determinant offsets
      selfData:    (N, d) full training data
      selfLogdets: (N,) per-kernel log|Sigma|
      nbrs_i:      (K,) kNN index array for this query point - the kdtree's
                   truncation, used here to restrict the kernel sum to those K
                   neighbours rather than the full N. Summing all N kernels per
                   query instead wastes O(N/K) CPU and allocates an (N, d, d) Us
                   per thread.
      point_i:     (d,) the query point
      i:           int, chunk-local index (used by caller for scatter)
    """
    # log_weights optional (last element) for backward-compat with any caller
    # that still packs the 7-tuple; default to zeros (uniform) if absent.
    if len(args) == 8:
        selfUs, logcovfac, selfData, selfLogdets, nbrs_i, point_i, i, log_weights = args
    else:
        selfUs, logcovfac, selfData, selfLogdets, nbrs_i, point_i, i = args
        log_weights = None
    try:
        Us = selfUs[nbrs_i] * 1.0/np.exp(0.5*logcovfac[nbrs_i, None, None])
    except:
        pdb.set_trace()
    devs = point_i - selfData[nbrs_i]
    devUs = np.einsum('ni,nij->nj', devs, Us)
    mahas = np.sum(np.square(devUs), axis=1)

    dim = selfUs.shape[2]
    log2pi = np.log(2.0*np.pi)
    contribs = -0.5 * (dim * log2pi + mahas + logcovfac[nbrs_i]*dim + selfLogdets[nbrs_i])
    if log_weights is not None:
        contribs = contribs + log_weights[nbrs_i]
    est = scipy.special.logsumexp(contribs)
    neff = np.exp(2*est - scipy.special.logsumexp(2*contribs))

    return est, neff, contribs, i


def _strip_self_from_cache(cache, self_kde_indices=None):
    """Remove self-contributions from a training-data cache for LOO evaluation.

    For a cache built on the training data, row i's neighbour list contains
    data point i (zero distance, full kernel mass). Mask that entry out so
    the LOO density at each x_i excludes the i-th kernel.

    `self_kde_indices` (optional): the i-th cache row's self-index in the KDE's
    training-data ordering. Defaults to `np.arange(M)` (cache built on the FULL
    training set, in order). Pass an explicit array when the cache was built
    on a subset (e.g. ROI-filtered LOO) - then row r corresponds to KDE-index
    `self_kde_indices[r]`, not r.

    Identifies self by neighbours == self_index. The ball-query path doesn't
    sort by distance, so we can't use the [:, 1:] slice that worked with kNN.
    """
    M = cache['M']
    neighbors = cache['neighbors']
    valid = cache.get('valid', None)
    if self_kde_indices is None:
        self_idx = np.arange(M)[:, None]
    else:
        self_idx = np.asarray(self_kde_indices, dtype=neighbors.dtype)[:, None]
    is_self = (neighbors == self_idx)
    if valid is not None:
        new_valid = valid & ~is_self
    else:
        # Legacy kNN path: build a fresh mask from is_self.
        new_valid = ~is_self
    return {
        'mahas': cache['mahas'],
        'logdets_at_n': cache['logdets_at_n'],
        'neighbors': neighbors,
        'M': M,
        'valid': new_valid,
    }


## WIP
#class cvSimpleKDE:
#    def __init__(self, dataIn, nfolds=5):
#        data = copy.deepcopy(dataIn)
#        kf = KFold(n_splits=nfolds, shuffle=True)
#
#        assert data.shape[1]==6
#
#        def simplevscale(vf):
#            return np.array([ 1.0, 1.0, 1.0, vf, vf, vf])
#
#        covfacs = np.logspace(-1.2,1.2,ncovfacs)
#        vfacs = [ 0.001, 0.0031, 0.01, 0.031, 0.1, 0.31, 1.0, 3.1, 10.0  ]
#        scores = np.zeros( (len(covfacs), len(vfacs), nfolds) )
#        signs = np.zeros( scores.shape )
#        errorsRR = np.zeros( scores.shape )
#        RRs = np.zeros( scores.shape )
#
#
#
#        for i, (train_index, test_index) in enumerate(tqdm(kf.split(data), position=0, desc='kfold', total=nfolds)):
#            train_data = data[train_index,:]
#
#            test_index_this = test_index
#            test_data = data[test_index_this,:]
#            for iii in tqdm(range(len(vfacs)), position=2,leave=False, desc='vfac'):
#                # covfac is 1 here - we don't want to rebuild the tree and re-evaluate the covariances every time. We can just scale them at evaluation.
#                scalings = simplevscale( vfacs[iii] )
#                tim = timer()
#                thisFoldKDE = gaussian_kde( train_data, scalings=scalings, covfac=1.0 )
#                tim.tick('init')
#                for kk in tqdm(range(len(covfacs)), position=3, desc='covfacs', leave=False):
#                    validate_densities = thisFoldKDE( test_data, covfac=covfacs_this[kk], returnLog=True )
#                    tim.tick('evaluate')
#                    score = scipy.special.logsumexp(validate_densities) - np.log(len(test_data))
#                    #score = np.sum(np.log10(np.clip(validate_densities,1.0e-299,None))) # this is the SECOND term in the cross-validation scheme
#                    # we also need to subtract 0.5 R(fhat) = 0.5 Integral( fhat^2(x) dx ) approx 0.5 * mean ( fhat(x) ) where x is drawn from fhat itself.
#                    nsamples_total = nbootThis
#                    samples = thisFoldKDE.draw(size=nsamples_total,covfac=covfacs_this[kk], cholesky=True)
#                    tim.tick('draw')
#                    
#                    if not roiThis is None:
#                        sample_r = (samples[:,0]-roiCenter[0])**2 + (samples[:,1]-roiCenter[1])**2 + (samples[:,2]-roiCenter[2])**2
#                        selec_sample = sample_r < roiThis*roiThis
#                        samples = samples[selec_sample,:]
#                        #print("Sample size for ROI RR: ", np.sum(selec_sample))
#                        if np.sum(selec_sample)==0:
#                            pdb.set_trace()
#                    tim.tick('downselect')
#                    nsamples = len(samples)
#                    #samples_slow = thisFoldKDE.draw(size=nsamples,covfac=covfacs_this[kk], cholesky=False)
#                    #figchol = corner.corner( samples_cholesky )
#                    #plt.savefig('samples_cholesky_debug.png', dpi=300)
#                    #plt.close(figchol)
#                    #figslow = corner.corner( samples_slow )
#                    #plt.savefig('samples_slow_debug.png', dpi=300)
#                    #plt.close(figslow)
#
#                    #RR = np.mean( thisFoldKDE(samples) ) # double check this!
#                    logpsamples = thisFoldKDE(samples, returnLog=True, covfac=covfacs_this[kk])
#                    tim.tick('evaluate again')
#                    psamples = np.exp(logpsamples)
#                    error_on_expRR = np.std(psamples)/np.sqrt(nsamples)
#                    RR = scipy.special.logsumexp( logpsamples ) - np.log( nsamples ) # this is the (log of the) mean of "psamples"
#                    the_score, the_sign = scipy.special.logsumexp([score,RR],b=[1.0*reg,-0.5], return_sign=True) 
#                    scores[kk,iii,i] = the_score
#                    signs[kk,iii,i] = the_sign
#                    errorsRR[kk,iii,i] = error_on_expRR
#                    RRs[kk,iii,i] = RR
#                    tim.tick('record')
#                    tim.report()
#                    pdb.set_trace()
#        #avg_scores = scipy.special.logsumexp(scores, axis=-1)
#        avg_scores = selector( signs*np.exp(scores), axis=-1) # was np.sum (i.e. averages). Find the best "worst case" scenario.
#        best = np.unravel_index( np.nanargmax(avg_scores), avg_scores.shape)
#        print("Best results at", best)
#        print("Favorite covfac and nn: ", covfacs[best[0]], nns[best[1]], vfacs[best[2]] )
#        print("All scores: ")
#        print(avg_scores)
#
##        print(" ")
##        print("k-fold relative errors on RR")
##        print( np.std(np.exp(RRs),axis=-1)/np.mean(np.exp(RRs),axis=-1))
##        print(" ")
##        print("mean RR error relative to final score")
##        #print( np.mean(errorsRR,axis=-1)/np.mean(np.exp(RRs),axis=-1))
##        print( np.mean(errorsRR,axis=-1)/np.abs(avg_scores))
#        if not np.any(np.isfinite(avg_scores)):
#            pdb.set_trace()
#        asymcorr = (float(nfolds)/float(nfolds-1.))**(-2.0/(4.0+data.shape[1]))
#        print("asymptotic correction factor: ", asymcorr)
#        self.kde = adaptiveKDE( data, scalings=scalings, nn=nns[best[1]], covfac=covfacs[best[0]]*asymcorr )
#
#        self.avg_scores = avg_scores
#
#
#
#    def __call__(self, points, covfac=1.0):
#        return self.kde(points,covfac=covfac)



class cvAdaptiveKDE:
    #def __init__(self, dataIn, nfolds=5, broad=True, kappa=1.0, nu=1.0, nboot=None, ncovfacs=7, roi=None, roiThresh=10, roiCenter=None, reg=1.0, selector=np.sum ):
    def __init__(self, dataIn, nfolds=5, broad=True, kappa=1.0, nu=1.0, nboot=None, ncovfacs=11, roi=1.0, roiThresh=200, roiCenter=None, reg=1.0, selector=np.mean, ncovalphas=4, nn=None, scalings=None, scalings_grid=None, nshrinkages=5, shrinkage_grid=None, covfac_range=(-1.5, 0.5), covalpha_range=(-0.5, 0.5), rr_method='ISE', weight_fn=None, shrinkage_target='global', K_pool=None, max_kernel_span_factor=5.0, random_state=None, neff_eval_points=None, neff_floor=None, roi6=None, roiCenter6=None, natural_covfac=None, natural_covalpha=None, stability_lambda=0.0, weights=None,
                  # Rate-weighted ESS metrics for the sky-map / v_inf picks.
                  # Pass per-eval-point arrays alongside neff_eval_points and
                  # CV will compute Kish ESS of the rate-weighted histograms
                  # during the inner loop (cheap - uses the cache already built
                  # for the median-N_eff measurement).
                  sky_bin_costheta=None, sky_bin_phi=None,
                  vinf_bin_coord=None, rate_weight_geom_factor=None,
                  sky_bins=(12, 24), vinf_bins=20):
        # Normalise rr_method so both the new 'ISE' (clearer name - the
        # objective IS an ISE approximation; the bootstrap is just how the
        # RR term is estimated) and the historical 'bootstrap' resolve to
        # the same internal flag. LOO is the alternative.
        rr_lower = str(rr_method).lower()
        if rr_lower in ('ise', 'bootstrap'):
            _rr_method_normalized = 'ise'
        elif rr_lower == 'loo':
            _rr_method_normalized = 'loo'
        else:
            raise ValueError(f"rr_method must be 'ISE'/'bootstrap' or 'LOO'; got {rr_method!r}")
        data = copy.deepcopy( dataIn )

        # Per-training-point input weights (e.g. Gaia ISO production weights),
        # distinct from `weight_fn` (the rate-weighting of the CV objective).
        # Normalised to sum to N. Carried into the per-fold adaptiveKDE builds
        # (train weights), the weighted test-fold score, the weighted-draw RR
        # term, and the final estimator build. None -> uniform (unchanged).
        _N_all = data.shape[0]
        if weights is None:
            self.weights = np.ones(_N_all)
        else:
            _w = np.asarray(weights, dtype=float).reshape(-1)
            if _w.shape != (_N_all,):
                raise ValueError(f"weights shape {_w.shape} != ({_N_all},)")
            if np.any(_w < 0) or not np.all(np.isfinite(_w)):
                raise ValueError("weights must be non-negative and finite")
            self.weights = _w / _w.sum() * _N_all

        # ------- nn, K_pool scheduling (2026-05-19)
        # Classical KDE consistency conditions require nn -> inf and nn/N -> 0 as
        # N -> inf for a sample-point estimator with per-point local pilot. Fixed
        # nn (e.g. nn=10) gives a fixed-variance per-point cov estimate and the
        # estimator is not asymptotically consistent - visible as anti-converging
        # density-bias scatter and sky-TV at large N in the convergence sweep.
        # Schedule nn = max(10, floor(sqrtN)) and K_pool = max(50, floor(sqrtN)) by default;
        # caller can override either.
        N_data = dataIn.shape[0]
        d_data = dataIn.shape[1]
        if nn is None:
            nn = max(10, int(N_data ** 0.5))
        if K_pool is None:
            K_pool = max(50, int(N_data ** 0.5))

        # ------- natural covfac/covalpha reparameterization (2026-05-19)
        # The CV grid sweeps USER-FACING (covfac, covalpha) where (1, 0)
        # corresponds to Abramson volume-equalized kernels at the Silverman
        # bandwidth scale against |Sigma_global|. Specifically:
        #   covfac_raw      = covfac_user      * silverman^2 * |Sigma_data|^(1/d)_geom
        #   covalpha_raw    = covalpha_user    - 1/d
        # The raw values are what adaptiveKDE consumes; user-facing values are
        # what get reported via cv.{rate,shape}_{covfac,covalpha}_user and in
        # the CV grid axes. The natural anchor makes covfac_user=1 land at
        # the scipy.gaussian_kde bandwidth volume relative to |Sigma_data|.
        # covalpha_user=0 = Abramson volume-equalizing exponent.
        #
        # 2026-05-22: switched the natural-volume anchor from |Sigma|^(1/d) (full
        # determinant) to (Pi diag(Sigma))^(1/d) (geometric mean of per-axis
        # variances). The full-determinant form collapses to ~0 for data on
        # a low-dimensional manifold (e.g. a galactic ring in 6D phase space
        # has highly correlated v_x/v_y -> |Sigma_scaled|^(1/d) ~ 0.004 on disk
        # streams) - making cf_user=0.01 mean a delta-like kernel on `auto`
        # / `narrow` scalings, while it means a sensible kernel on
        # `narrow_local`. The geometric-mean-diag form is robust to
        # correlation structure: nat_cf ~ silverman^2 on any auto-like
        # scaling, so cf_user has the same physical meaning across scalings.
        silverman_sq = N_data ** (-2.0 / (d_data + 4))
        if natural_covfac is None:
            cov_full = np.cov(dataIn.T)
            diag_data = np.diagonal(cov_full)
            # Geometric mean of per-axis variances via log-space mean for
            # numerical stability.
            log_diag = np.log(np.maximum(diag_data, 1e-300))
            natural_covfac = silverman_sq * np.exp(np.mean(log_diag))
        if natural_covalpha is None:
            natural_covalpha = -1.0 / d_data
        self.natural_covfac = float(natural_covfac)
        self.natural_covalpha = float(natural_covalpha)
        self.silverman_sq = float(silverman_sq)
        # Seed both the KFold splits and the bootstrap-RR draws so a given
        # (data, random_state) pair gives bit-deterministic CV picks. Without
        # a seed, sklearn's KFold(shuffle=True) and `np.random` defaults
        # produce different fold partitions and bootstrap samples on every
        # call, which is the dominant source of run-to-run rate-recovery
        # noise observed in the validation suite.
        kf = KFold(n_splits=nfolds, shuffle=True, random_state=random_state)
        # Local rng for the per-iteration draw() bootstraps (see eval RR loop).
        rr_rng = np.random.default_rng(random_state)

        nbootThis = nboot
        if nbootThis is None:
            if roi is None:
                nbootThis = 200
            else:
                nbootThis = data.shape[0]*2


        if broad:
            nns = [15, 31, 63 ]
            vfac = [  0.1, 0.31, 1.0]
            zfac = [0.031, 0.1, 0.31 ]
            vzfac = [ 0.1, 0.31, 1.0]
            #covfacs = [ 0.3, 0.5, 1.0, 2.0]
            covfacs = [1.0]
        else:
            nns = [31]
            vfac = [0.03]
            covfacs = [1.0]
        
        # cvAdaptiveKDE works in any dimension d (the CV + adaptive-kernel math
        # is dimension-general). d==6 is the phase-space default; d==3 is used
        # for velocity-only estimation under a constant-spatial-density
        # assumption (HBL Gaia ISO). NOTE for d != 6: pass scalings='auto' or an
        # explicit d-list, and DON'T use scalings 'narrow'/'narrow_local' or the
        # spatial roi/roi6/shrinkage_target='local_pooled' options - those assume
        # the 6D [pos(3), vel(3)] layout. (The None default and 'auto' are
        # dimension-safe; see _resolve_scalings.)
        assert data.shape[1] >= 1

        def simplevscale(vf):
            return np.array([ 1.0, 1.0, 1.0, vf, vf, vf])


        counter = 0
        print("cross-validating KDE")

#        for kk in tqdm(range(len(covfacs)), position=0, desc='covfacs'):
#            for ii in tqdm(range(len(nns)), position=1,leave=False, desc='nns'):
#                for jj in tqdm(range(len(vfac)), position=2, leave=False, desc='vfac'):
#                    for qq in tqdm(range(len(zfac)), position=3, leave=False, desc='zfac'):
#                        for rr in tqdm(range(len(vzfac)), position=4, leave=False, desc='vzfac'):
#                            for i, (train_index, test_index) in enumerate(tqdm(kf.split(data), position=5, leave=False, desc='kfold', total=5)):
#                                scalings = simplevscale(vfac[jj])
#                                scalings[2] *= zfac[qq]
#                                scalings[5] *= vzfac[rr]
#                                tim = timer()
#                                thisFoldKDE = adaptiveKDE( data[train_index,:], scalings=scalings, nn=nns[ii], covfac=covfacs_this[kk])
#                                tim.tick('train KDE')
#                                validate_densities = thisFoldKDE( data[test_index,:] )
#                                tim.tick('test KDE')
#                                score = np.sum(np.log10(np.clip(validate_densities,1.0e-99,None)))
#                                
#                                scores[kk,ii,jj,rr,qq,i] = score
#                                #print('kk,ii,jj,i; score', kk,ii,jj,i,score)
#                                #tim.report()
#        avg_scores = np.mean(scores,axis=-1)
#        best = np.unravel_index(np.nanargmax(avg_scores), avg_scores.shape)
#        print("Best results at ii,jj:" , best )
#        print("All scores:", avg_scores, avg_scores.shape)
#        print("Best score is", avg_scores[best[0],best[1],best[2],best[3],best[4]])
#        print("Favorite fvac, nn, and covfac: ", vfac[best[2]], nns[best[1]], covfacs[best[0]] )
#
#        scalings = simplevscale(vfac[best[2]])
#        scalings[2] *= zfac[best[3]]
#        scalings[5] *= vzfac[best[4]]
#        self.kde = adaptiveKDE( data, scalings=simplevscale(vfac[best[2]]), nn=nns[best[1]], covfac=covfacs[best[0]])

        # Scalings determine the kdtree distance metric: smaller scaling on a dimension
        # makes that dimension matter more in "nearest neighbor". Default uses
        # epicyclic frequencies (appropriate for stream physics in a galactic disk).
        # Pass `scalings='auto'` for empirical per-axis std (treats all 6D axes
        # comparably - useful for non-epicyclic structure like rings or bimodal
        # velocity distributions).
        # `scalings_grid` (optional) is a list of options to CV over. Each element
        # is None / 'auto' / explicit 6-list, same convention as `scalings`. When
        # not provided, defaults to a singleton grid of `[scalings]` (backward
        # compatible). The cosine-shear test demonstrated that no single auto-derived
        # scaling works across scenarios - gridding lets CV pick the right metric
        # per dataset.
        def _resolve_scalings(sc_input):
            if sc_input is None:
                if data.shape[1] == 6:
                    return [1.0, 1.0, nu/kappa, kappa, kappa, nu], 'unit'
                # Non-6D (e.g. velocity-only 3D): unit metric of the right length.
                return list(np.ones(data.shape[1])), 'unit'
            elif isinstance(sc_input, str) and sc_input == 'auto':
                return list(data.std(axis=0)), 'auto'
            elif isinstance(sc_input, str) and sc_input == 'narrow':
                # Position: empirical std (same as 'auto' for x). Velocity: detect a
                # narrow sub-population in each velocity axis via the helper -
                # 2-component 1D GMM with biased narrow init + 5% weight filter.
                pos_std = list(data[:, :3].std(axis=0))
                v_narrow = _detect_narrow_v_sigma(data)
                return pos_std + v_narrow, 'narrow'
            elif isinstance(sc_input, str) and sc_input == 'narrow_local':
                # Like 'narrow' but the GMM (and position-std) are computed on a
                # position-local SUBSET of the data, anchored at roiCenter. This
                # captures stream-aligned sigma at the encounter target - essential
                # for ring/stream data where the narrow velocity direction varies
                # with phi and the GLOBAL marginals don't reveal sigma_t directly.
                rc = roiCenter if roiCenter is not None else [0.0, 0.0, 0.0]
                rsq = ((data[:, 0] - rc[0]) ** 2
                       + (data[:, 1] - rc[1]) ** 2
                       + (data[:, 2] - rc[2]) ** 2)
                sortr = np.argsort(rsq)
                # K_LOCAL schedule: max(300, N//10).
                # Conservative bound is 300, principled bound is N/10 (keeps
                # spatial extent of the local subset roughly constant as N grows
                # for uniform 3D data, and provides >=50 narrow-component
                # particles for the GMM at N>=10k assuming the 5% weight floor).
                # Fixed K=300 was the cause of cf-bimodality at large N on
                # disk_stream: ~15 narrow particles -> ~25% noise on sigma_narrow
                # -> cf picks bouncing in [0.02, 0.16]. (2026-05-24.)
                K_LOCAL = max(300, data.shape[0] // 10)
                K_LOCAL = min(K_LOCAL, data.shape[0] - 1)
                subset = data[sortr[:K_LOCAL]]
                pos_std = list(subset[:, :3].std(axis=0))
                # Floor at 1% of global std to avoid degenerate near-zero scales
                pos_std = [max(s, 0.01 * float(np.std(data[:, i])))
                           for i, s in enumerate(pos_std)]
                v_narrow = _detect_narrow_v_sigma(subset)
                return pos_std + v_narrow, f'narrow_local(K={K_LOCAL})'
            else:
                resolved = list(sc_input)
                return resolved, f'custom({np.array(resolved).round(2)})'

        if scalings_grid is None:
            scalings_grid = [scalings]
        resolved_scalings_list = []
        scalings_labels = []
        for sc_input in scalings_grid:
            resolved, label = _resolve_scalings(sc_input)
            resolved_scalings_list.append(resolved)
            scalings_labels.append(label)

        # Grid in USER-FACING units (post-2026-05-19 reparameterization, with
        # per-scaling natural fix 2026-05-20, geometric-mean-diag fix 2026-05-22).
        # covfac_range/covalpha_range are interpreted in user-facing log10/linear
        # units; we translate to raw adaptiveKDE values via the natural anchors.
        # The "natural covfac" must be computed PER SCALING because adaptiveKDE
        # applies the covfac in the SCALED coordinate system (data divided by
        # `scales`). For the user-facing kernel volume to be invariant across
        # scaling choices, the raw covfac passed to adaptiveKDE must scale with
        # the per-axis spread of the scaled data:
        #   nat_cf(sc) = silverman^2 * (Pi diag(Sigma_scaled(sc)))^(1/d)
        # i.e. silverman^2 times the geometric mean of per-axis variances. We
        # avoid the full determinant |Sigma|^(1/d) because it collapses to ~0 for
        # data on a low-dim manifold (e.g. galactic ring -> highly correlated
        # v_x/v_y -> |Sigma_scaled|^(1/d) ~ 0.004 on `auto` for disk_stream, making
        # cf_user=0.01 mean a delta-like kernel only on that scaling). The
        # geometric-mean-diag form is robust to correlation structure, so
        # cf_user has the same physical meaning across scalings.
        # Default ranges:
        #   covfac_range = (-1.5, 0.5) - user covfac in [0.03, 3.2]
        #   covalpha_range = (-0.5, 0.5) - user covalpha +/-0.5 around Abramson natural
        covfacs_user = np.logspace(covfac_range[0], covfac_range[1], ncovfacs)
        covalphas_user = np.linspace(covalpha_range[0], covalpha_range[1], ncovalphas)
        # Per-scaling natural covfac.
        natural_covfacs_per_scaling = np.empty(len(resolved_scalings_list))
        for sc_idx_init, scales_this in enumerate(resolved_scalings_list):
            scales_arr = np.asarray(scales_this, dtype=float)
            data_scaled = data / scales_arr[None, :]
            diag_scaled = np.diagonal(np.cov(data_scaled.T))
            log_diag = np.log(np.maximum(diag_scaled, 1e-300))
            natural_covfacs_per_scaling[sc_idx_init] = (
                silverman_sq * np.exp(np.mean(log_diag)))
        # covfacs[sc_idx, k] - raw covfac to pass to adaptiveKDE for scaling
        # sc_idx and user-facing covfac k. covalpha doesn't depend on scaling
        # (Abramson exponent is dimensionless), so it stays 1D.
        covfacs = (covfacs_user[None, :]
                   * natural_covfacs_per_scaling[:, None])
        covalphas = covalphas_user + self.natural_covalpha
        self.covfacs_user = covfacs_user
        self.covalphas_user = covalphas_user
        self.natural_covfacs_per_scaling = natural_covfacs_per_scaling
        # nn default = 10 - small enough that local Sigma_i captures fine structure when
        # scalings make KNN structurally local (cosine shear: nn=10+tight_x -> 100%);
        # large nn (formerly default max(50, N//10)) is recoverable via shrinkage=1
        # in CV, so the smaller default is strictly more flexible.
        if nn is None:
            nns = [10]
        else:
            nns = [nn]
        vfacs = [0.1]
        # Shrinkage interpolates per-point Sigma_i toward the global sample covariance -
        # 0 = fully local (original adaptiveKDE), 1 = fully global (scipy-like).
        # Adding it to the CV grid lets the optimum sit anywhere on the local<->global axis.
        if shrinkage_grid is None:
            shrinkages = np.linspace(0.0, 1.0, nshrinkages)
        else:
            shrinkages = np.asarray(shrinkage_grid)
        # scores axes: (covfacs, covalphas, nns, vfacs, shrinkages, scalings, nfolds)
        # covfacs is now 2D (n_sc, ncovfacs) post-per-scaling fix; use the
        # user-grid length for the score-array's covfac axis.
        n_sc = len(resolved_scalings_list)
        scores = np.zeros((len(covfacs_user), len(covalphas), len(nns),
                           len(vfacs), len(shrinkages), n_sc, nfolds))
        signs = np.zeros(scores.shape)
        errorsRR = np.zeros(scores.shape)
        RRs = np.zeros(scores.shape)
        # N_eff floor: median per-eval N_eff at neff_eval_points for each (cf,
        # ca, nn, vf, sh, sc, fold). After CV, picks are restricted to grid
        # entries clearing neff_floor, so the chosen kernel is structurally
        # smooth at the rate-evaluation locus rather than just minimising ISE
        # at the training-data positions. NaN-filled when the floor is unused.
        neffs = np.full(scores.shape, np.nan)
        # Rate-weighted Kish ESS at the sky-bin (cos theta, phi) and v_inf-bin
        # aggregations. Populated alongside neffs in the inner loop when the
        # caller supplies the per-eval geometry inputs (rate_weight_geom_factor,
        # sky_bin_costheta, sky_bin_phi, vinf_bin_coord). NaN-filled otherwise.
        # The Kish ESS of a histogram H is (Sigma H)^2 / Sigma H^2 - measures how
        # concentrated the rate-weight is across bins; high = uniform coverage.
        sky_ess_arr  = np.full(scores.shape, np.nan)
        vinf_ess_arr = np.full(scores.shape, np.nan)
        neff_eval_points = (None if neff_eval_points is None
                            else np.asarray(neff_eval_points))
        _rate_ess_active = (rate_weight_geom_factor is not None
                            and sky_bin_costheta is not None
                            and sky_bin_phi is not None
                            and vinf_bin_coord is not None
                            and neff_eval_points is not None)
        if _rate_ess_active:
            sky_bin_costheta = np.asarray(sky_bin_costheta, dtype=float)
            sky_bin_phi = np.asarray(sky_bin_phi, dtype=float)
            vinf_bin_coord = np.asarray(vinf_bin_coord, dtype=float)
            rate_weight_geom_factor = np.asarray(rate_weight_geom_factor, dtype=float)
            _vinf_lo = float(np.nanmin(vinf_bin_coord))
            _vinf_hi = float(np.nanmax(vinf_bin_coord)) + 1e-12
            _vinf_edges = np.linspace(_vinf_lo, _vinf_hi, int(vinf_bins) + 1)

        # Rate-weighted CV: instead of minimising ISE = int(f-fhat)^2 dx, minimise the
        # rate-weighted version int(f-fhat)^2*w(x)^2 dx where w is the rate-leverage weight
        # (concentrated at low |v| via the gravitational focusing factor). This adds
        # 2*log w(X) to each density evaluation before logsumexp; bound particles
        # with w=0 contribute nothing (correct, since they don't contribute to rate).
        # Cost: one weight evaluation per test/sample point (negligible vs kernel sums).
        def compute_log_wsq(points):
            if weight_fn is None:
                return 0.0    # additive identity -> no change to score
            w = np.asarray(weight_fn(points), dtype=float)
            with np.errstate(divide='ignore'):
                return np.where(w > 0, 2.0 * np.log(w), -np.inf)


        # 6D ROI mode: dynamic-radius scaled-Euclidean ball that expands
        # until it contains exactly `roi6` particles. The metric is the same
        # `scalings` the kdtree uses, so for each scaling option in the CV
        # grid the ROI is computed under that scaling (inside the sc_idx
        # loop below). Center: position from roiCenter (or origin), velocity
        # from roiCenter6 (or zeros) - typically the rate-leverage region.
        # When active, supersedes the 3D ROI machinery below.
        roi6_active = (roi6 is not None and data.shape[1] == 6)
        roi6_center_arr = None
        if roi6_active:
            if roiCenter6 is not None:
                roi6_center_arr = np.asarray(roiCenter6, dtype=float)
            else:
                rc_pos = roiCenter if roiCenter is not None else [0.0, 0.0, 0.0]
                roi6_center_arr = np.array(list(rc_pos) + [0.0, 0.0, 0.0], dtype=float)
            roiThis = None    # disable 3D ROI codepath
            print(f"6D ROI: target K={int(roi6)} particles per (fold, scaling) "
                  f"under each kdtree metric")
        else:
            roiThis = roi
            if roiThis is None:
                pass
            else:
                # Default roiCenter to the origin when caller specified roi but not center.
                if roiCenter is None:
                    roiCenter = [0.0, 0.0, 0.0]
                rsquared = (data[:,0]-roiCenter[0])**2 + (data[:,1]-roiCenter[1])**2 + (data[:,2]-roiCenter[2])**2
                selecroi = rsquared < roi*roi
                nroi = np.sum(selecroi)
                if nroi < roiThresh:
                    # we have a problem! Not enough particles in the region of interest.
                    # Expand to the radius needed to encompass roiThresh nearest data points.
                    sortr = np.argsort(rsquared)
                    rsquaredCrit = rsquared[sortr[roiThresh]] # the rsquared we need to go to to get enough particles
                    roiThis = np.sqrt(rsquaredCrit)
                    selecroi = rsquared < rsquaredCrit
                else:
                    pass

            print("ROI: ", roiThis)

        for i, (train_index, test_index) in enumerate(tqdm(kf.split(data), position=0, desc='kfold', total=nfolds)):
            train_data = data[train_index,:]

            for sc_idx in tqdm(range(n_sc), position=1, leave=False, desc='scalings'):
                scalings_this = resolved_scalings_list[sc_idx]
                # Per-scaling raw covfac grid (post-2026-05-20 fix): user-facing
                # covfac is multiplied by this scaling's natural anchor to give
                # the raw covfac passed to adaptiveKDE.
                covfacs_this = covfacs[sc_idx]

                # ROI selection: depends on the kdtree metric (= scalings_this)
                # under 6D mode, and on the global 3D ball under classic mode.
                # Computed inside the sc_idx loop so per-scaling ROIs are
                # properly coordinated with the per-scaling kernels.
                if roi6_active:
                    Dvec = 1.0 / np.asarray(scalings_this, dtype=float)
                    diffs = (data - roi6_center_arr) * Dvec
                    rsq6 = np.sum(diffs ** 2, axis=1)
                    sortr = np.argsort(rsq6)
                    K = min(int(roi6), data.shape[0])
                    roi6_thresh_sq = rsq6[sortr[K - 1]]
                    selecroi = rsq6 <= roi6_thresh_sq
                elif roiThis is not None:
                    pass    # selecroi already computed pre-loop above
                # else: no ROI; selecroi is undefined and not referenced.

                test_index_this = test_index
                if roi6_active or (roiThis is not None):
                    test_ind_array = np.zeros(data.shape[0], dtype=bool)
                    test_ind_array[test_index] = True
                    test_index_this = test_ind_array & selecroi
                    if np.sum(test_index_this) == 0:
                        scores[:, :, :, :, :, sc_idx, i] = -1.e30
                        signs[:, :, :, :, :, sc_idx, i] = -1
                        continue
                test_data = data[test_index_this, :]
                log_wsq_test = compute_log_wsq(test_data)
                # Per-training-point input weights for this fold.
                w_train_fold = self.weights[train_index]
                log_w_test = np.log(np.maximum(self.weights[test_index_this], 1e-300))
                log_w_test_norm = scipy.special.logsumexp(log_w_test)

                # Floor eval points: when the user supplied a fixed set
                # they're used as-is; otherwise (auto-mode) we'd regenerate
                # per (fold, scaling) - but that's done at the caller for now,
                # and the fixed neff_eval_points work fine in practice since
                # the floor measures kernel overlap at sphere positions and
                # the kernel widths track the test-data anyway.
                neff_pts_this = neff_eval_points
                for ii in tqdm(range(len(nns)), position=2, leave=False, desc='nns'):
                    for iii in tqdm(range(len(vfacs)), position=3, leave=False, desc='vfac'):
                        # Build the fold's adaptiveKDE ONCE (with shrinkage=0). For each shrinkage
                        # value we then just call .apply_shrinkage(alpha) - which recomputes the
                        # shrinkage-dependent attributes (covariances, Cholesky, eigh, Schur)
                        # from the cached `covariances_local` and `global_cov`. This skips the
                        # tree query and per-point local covariance loop on subsequent shrinkages.
                        thisFoldKDE = adaptiveKDE(train_data, scalings=scalings_this, nn=nns[ii], covfac=1.0, covalpha=0.0, shrinkage=0.0, shrinkage_target=shrinkage_target, K_pool=K_pool, weights=w_train_fold)
                        # For LOO RR: precompute the unscaled training data once (doesn't change
                        # with shrinkage). Used to build the LOO cache inside the shrinkage loop.
                        train_data_unscaled = thisFoldKDE.data * np.asarray(thisFoldKDE.scales)[None, :]
                        # Apply the same ROI filter used by the bootstrap path -
                        # i.e. when roi6/roi is active, restrict the LOO eval
                        # locus to training points inside the ROI. Without this,
                        # the LOO branch was evaluating the full per-fold train
                        # set (no throttle) while the bootstrap path filtered
                        # samples down to the ROI subset; LOO ran ~15x slower
                        # as a result on roi6-enabled production CVs (2026-05-25).
                        if roi6_active or (roiThis is not None):
                            train_roi_mask = selecroi[train_index]
                            train_data_loo = train_data_unscaled[train_roi_mask]
                            log_wsq_train = compute_log_wsq(train_data_loo)
                            # KDE-index of each ROI-subset row, used by
                            # _strip_self_from_cache to identify the self-row.
                            train_loo_self_idx = np.where(train_roi_mask)[0]
                        else:
                            train_data_loo = train_data_unscaled
                            log_wsq_train = compute_log_wsq(train_data_unscaled)
                            train_loo_self_idx = None    # full set: row r <-> KDE-index r
                        nsamples_loo = max(len(train_data_loo) - 1, 1)
                        # CV uses a slightly looser ball-query threshold than production:
                        # m=3 captures ~99% of kernel mass and gives ~20x smaller caches
                        # than m=5 in 6D. The residual ~1% bias is uniform-ish across the
                        # (cf, ca) grid and doesn't affect CV's *relative* ranking; the
                        # final production KDE is built outside the loop and uses the
                        # default m=5 for full precision.
                        cv_m_thresh = 3.0
                        # (B) Skip CV candidates whose effective kernel reaches more than
                        # `max_kernel_span_factor` x the data's largest 16-84 percentile
                        # span. A kernel that wide isn't doing density estimation, it's
                        # smearing the data into one blob - almost certainly wrong, and
                        # the cost of even computing its score is large because every
                        # query saturates the cache to full N.
                        max_kernel_radius = max_kernel_span_factor * thisFoldKDE.data_max_span
                        for sh in tqdm(range(len(shrinkages)), position=4, leave=False, desc='shrink'):
                            tim = timer()
                            thisFoldKDE.apply_shrinkage(shrinkages[sh])
                            tim.tick('apply_shrinkage')
                            for ik in range(len(covalphas)):
                                for kk in range(len(covfacs_user)):
                                    # Guardrail: skip if the worst-case kernel radius
                                    # would exceed the data span by `max_kernel_span_factor`.
                                    r_check = thisFoldKDE._max_kernel_radius(covfac_max=covfacs_this[kk], covalpha_range=(covalphas[ik], covalphas[ik]), m_thresh=cv_m_thresh)
                                    if r_check > max_kernel_radius:
                                        scores[kk, ik, ii, iii, sh, sc_idx, i] = -1.e30
                                        signs[kk, ik, ii, iii, sh, sc_idx, i] = -1
                                        errorsRR[kk, ik, ii, iii, sh, sc_idx, i] = 0
                                        RRs[kk, ik, ii, iii, sh, sc_idx, i] = 0
                                        continue
                                    # Per-iteration tight cache: build at this (cf, ca)'s
                                    # specific radius rather than caching once at covfac_max.
                                    # When the grid spans many decades of cf, the "build once"
                                    # cache saturates at full N for the high-cf candidates and
                                    # makes every eval pay that cost; per-iteration sizing
                                    # keeps low-cf evals fast.
                                    test_cache = thisFoldKDE.precompute_query(test_data, covfac_max=covfacs_this[kk], covalpha_range=(covalphas[ik], covalphas[ik]), m_thresh=cv_m_thresh)
                                    validate_densities = thisFoldKDE.eval_from_cache(test_cache, covalpha=covalphas[ik], covfac=covfacs_this[kk], returnLog=True)
                                    # Weighted test average: each test point carries
                                    # its input weight (data term intfhat*f_true).
                                    score = (scipy.special.logsumexp(validate_densities + log_wsq_test + log_w_test)
                                             - log_w_test_norm)
                                    # Free the test-cache before building the next cache so
                                    # the three per-iteration caches (test, neff, sample) aren't
                                    # simultaneously alive. Each can be ~2 GB at saturated K_max
                                    # (high covfac), so eager del reduces per-worker peak
                                    # roughly 3x during CV. (2026-05-23)
                                    del test_cache, validate_densities

                                    # Median per-eval N_eff at the user-supplied
                                    # eval points (e.g. encounter sphere). Used
                                    # post-loop as a hard floor on grid eligibility.
                                    if neff_eval_points is not None:
                                        neff_cache = thisFoldKDE.precompute_query(neff_eval_points, covfac_max=covfacs_this[kk], covalpha_range=(covalphas[ik], covalphas[ik]), m_thresh=cv_m_thresh)
                                        neff_arr = thisFoldKDE.eval_neff_from_cache(neff_cache, covalpha=covalphas[ik], covfac=covfacs_this[kk])
                                        neffs[kk, ik, ii, iii, sh, sc_idx, i] = float(np.median(neff_arr))
                                        # Rate-weighted sky/v_inf ESS: re-use the same
                                        # cache, get log fhat at the eval points, multiply
                                        # by the precomputed geometry factor, and bin.
                                        if _rate_ess_active:
                                            log_f = thisFoldKDE.eval_from_cache(neff_cache, covalpha=covalphas[ik], covfac=covfacs_this[kk], returnLog=True)
                                            w = np.exp(log_f) * rate_weight_geom_factor
                                            wsum = float(w.sum())
                                            if wsum > 0:
                                                H_sky, _, _ = np.histogram2d(sky_bin_costheta, sky_bin_phi, weights=w, bins=[int(sky_bins[0]), int(sky_bins[1])], range=[[-1.0, 1.0], [-np.pi, np.pi]])
                                                sky_denom = float((H_sky ** 2).sum())
                                                sky_ess_arr[kk, ik, ii, iii, sh, sc_idx, i] = (
                                                    (wsum ** 2) / max(sky_denom, 1e-300))
                                                H_vinf, _ = np.histogram(vinf_bin_coord, weights=w, bins=_vinf_edges)
                                                vinf_denom = float((H_vinf ** 2).sum())
                                                vinf_ess_arr[kk, ik, ii, iii, sh, sc_idx, i] = (
                                                    (wsum ** 2) / max(vinf_denom, 1e-300))
                                        del neff_cache, neff_arr

                                    if _rr_method_normalized == 'loo':
                                        # Empty-ROI guard (analogous to bootstrap's len(samples)==0).
                                        if len(train_data_loo) == 0:
                                            scores[kk, ik, ii, iii, sh, sc_idx, i] = -1.e30
                                            signs[kk, ik, ii, iii, sh, sc_idx, i] = -1
                                            errorsRR[kk, ik, ii, iii, sh, sc_idx, i] = 0
                                            RRs[kk, ik, ii, iii, sh, sc_idx, i] = 0
                                            continue
                                        train_cache = thisFoldKDE.precompute_query(train_data_loo, covfac_max=covfacs_this[kk], covalpha_range=(covalphas[ik], covalphas[ik]), m_thresh=cv_m_thresh)
                                        # LOO requires self at column 0 - but ball-query doesn't preserve
                                        # ordering. Detect and remove the self-contribution (zero-distance
                                        # entry per row) explicitly. When train_data_loo is the in-ROI
                                        # subset of train_data, _strip_self matches each row's true
                                        # self-index via neighbour-equality, so this remains correct.
                                        train_loo_cache = _strip_self_from_cache(train_cache, self_kde_indices=train_loo_self_idx)
                                        logpsamples = thisFoldKDE.eval_from_cache(train_loo_cache, covalpha=covalphas[ik], covfac=covfacs_this[kk], returnLog=True)
                                        del train_cache, train_loo_cache
                                        log_wsq_eval = log_wsq_train
                                        nsamples = nsamples_loo
                                    else:
                                        samples = thisFoldKDE.draw(size=nbootThis, covfac=covfacs_this[kk], covalpha=covalphas[ik], cholesky=True, rng=rr_rng)
                                        if roi6_active:
                                            sd = (samples - roi6_center_arr) * Dvec
                                            srsq = np.sum(sd ** 2, axis=1)
                                            samples = samples[srsq <= roi6_thresh_sq, :]
                                            if len(samples) == 0:
                                                scores[kk, ik, ii, iii, sh, sc_idx, i] = -1.e30
                                                signs[kk, ik, ii, iii, sh, sc_idx, i] = -1
                                                errorsRR[kk, ik, ii, iii, sh, sc_idx, i] = 0
                                                RRs[kk, ik, ii, iii, sh, sc_idx, i] = 0
                                                continue
                                        elif roiThis is not None:
                                            sample_r = ((samples[:, 0] - roiCenter[0]) ** 2
                                                        + (samples[:, 1] - roiCenter[1]) ** 2
                                                        + (samples[:, 2] - roiCenter[2]) ** 2)
                                            samples = samples[sample_r < roiThis * roiThis, :]
                                            if len(samples) == 0:
                                                scores[kk, ik, ii, iii, sh, sc_idx, i] = -1.e30
                                                signs[kk, ik, ii, iii, sh, sc_idx, i] = -1
                                                errorsRR[kk, ik, ii, iii, sh, sc_idx, i] = 0
                                                RRs[kk, ik, ii, iii, sh, sc_idx, i] = 0
                                                continue
                                        sample_cache = thisFoldKDE.precompute_query(samples, covfac_max=covfacs_this[kk], covalpha_range=(covalphas[ik], covalphas[ik]), m_thresh=cv_m_thresh)
                                        logpsamples = thisFoldKDE.eval_from_cache(sample_cache, covalpha=covalphas[ik], covfac=covfacs_this[kk], returnLog=True)
                                        del sample_cache
                                        log_wsq_eval = compute_log_wsq(samples)   # bootstrap samples -> per-iter
                                        nsamples = len(samples)

                                    psamples = np.exp(logpsamples)
                                    error_on_expRR = np.std(psamples) / np.sqrt(nsamples)
                                    RR = scipy.special.logsumexp(logpsamples + log_wsq_eval) - np.log(nsamples)
                                    the_score, the_sign = scipy.special.logsumexp([score, RR], b=[1.0 * reg, -0.5], return_sign=True)
                                    scores[kk, ik, ii, iii, sh, sc_idx, i] = the_score
                                    signs[kk, ik, ii, iii, sh, sc_idx, i] = the_sign
                                    errorsRR[kk, ik, ii, iii, sh, sc_idx, i] = error_on_expRR
                                    RRs[kk, ik, ii, iii, sh, sc_idx, i] = RR
                            tim.tick('inner CV sweep')
        per_fold_values = signs * np.exp(scores)
        avg_scores = selector(per_fold_values, axis=-1)

        # Stability tiebreaker (1-SE rule, 2026-05-21).
        # On a roughly-flat CV objective landscape (cf/alpha tradeoff means many
        # grid points have nearly equal true ISE), pure argmax(avg_score) is
        # dominated by MC noise - different trials pick wildly different
        # (cf, alpha, scaling) tuples with similar avg_score. The 1-SE rule
        # (Breiman) penalises picks by their across-fold standard error,
        # selecting the most reliable signal among similar-scoring picks:
        #
        #     combined_score = avg_score - stability_lambda * SEM(score)
        #     SEM = std_across_folds / sqrt(nfolds)
        #
        # `stability_lambda=0` (default) reproduces the original behaviour.
        # `stability_lambda=1` is the canonical 1-SE rule. Larger values
        # bias more aggressively toward stable picks.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            std_scores = np.nanstd(per_fold_values, axis=-1, ddof=1)
        sem_scores = std_scores / np.sqrt(max(nfolds, 1))
        if stability_lambda != 0.0:
            combined_scores = avg_scores - stability_lambda * sem_scores
        else:
            combined_scores = avg_scores

        # Dual-pick scheme: rate-faithful and shape-faithful estimators come
        # from the same CV grid via two different argmaxes.
        #   rate_best = argmax(combined_score)              (unconstrained)
        #   shape_best = argmax(combined_score | N_eff >= floor)
        # Rationale: the rate `int fhat*w dv` is unbiased on average even when
        # kernels barely overlap (particle-counter regime), so the
        # unconstrained ISE-CV pick recovers it on average - at the cost of
        # high MC variance. Differentials (sky map, marginal histograms,
        # log(fhat/f) at samples) are local and become unreliable at low
        # per-eval N_eff. The floor-constrained pick gives a smooth fhat for
        # those, at the cost of suppressing the rate via oversmoothing of
        # any narrow rate-leverage feature. Storing both lets the caller
        # report rate from rate_best and everything else from shape_best.
        rate_best = np.unravel_index(np.nanargmax(combined_scores), combined_scores.shape)

        neff_med_grid = np.nan
        floor_active = (neff_eval_points is not None
                        and neff_floor is not None
                        and np.any(np.isfinite(neffs)))
        # Sky / v_inf ESS grids - median across folds. Available regardless
        # of `neff_floor` (used by pick_at_sky_floor / pick_at_vinf_floor).
        import warnings as _w
        with np.errstate(invalid='ignore'):
            with _w.catch_warnings():
                _w.simplefilter("ignore", RuntimeWarning)
                sky_ess_grid = (np.nanmedian(sky_ess_arr, axis=-1)
                                if _rate_ess_active else None)
                vinf_ess_grid = (np.nanmedian(vinf_ess_arr, axis=-1)
                                 if _rate_ess_active else None)
        if floor_active:
            with np.errstate(invalid='ignore'):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    neff_med_grid = np.nanmedian(neffs, axis=-1)
            eligible = neff_med_grid >= neff_floor
            n_eligible = int(np.sum(eligible))
            n_total = int(np.sum(np.isfinite(neff_med_grid)))
            print(f"N_eff floor: {n_eligible}/{n_total} grid entries clear "
                  f"median N_eff >= {neff_floor:.0f} at sphere eval points")
            if n_eligible == 0:
                print(f"  WARNING: no grid entry meets the N_eff floor; "
                      f"shape-pick falls back to the unconstrained pick. "
                      f"(max median N_eff was {np.nanmax(neff_med_grid):.1f}.)")
                shape_avg_scores = combined_scores
            else:
                shape_avg_scores = np.where(eligible, combined_scores, -np.inf)
        else:
            shape_avg_scores = combined_scores

        shape_best = np.unravel_index(np.nanargmax(shape_avg_scores), shape_avg_scores.shape)

        print(f"Rate pick:  cf={covfacs[rate_best[5], rate_best[0]]:.3g} "
              f"(user={covfacs_user[rate_best[0]]:.3g}) "
              f"ca={covalphas[rate_best[1]]:.3g} sh={shrinkages[rate_best[4]]:.3g} "
              f"scaling={scalings_labels[rate_best[5]]}")
        print(f"Shape pick: cf={covfacs[shape_best[5], shape_best[0]]:.3g} "
              f"(user={covfacs_user[shape_best[0]]:.3g}) "
              f"ca={covalphas[shape_best[1]]:.3g} sh={shrinkages[shape_best[4]]:.3g} "
              f"scaling={scalings_labels[shape_best[5]]}")
        if floor_active and not np.isnan(neff_med_grid).all():
            print(f"  med N_eff: rate-pick={neff_med_grid[rate_best]:.1f}, "
                  f"shape-pick={neff_med_grid[shape_best]:.1f} "
                  f"(floor {neff_floor:.0f})")
        # Maintain back-compat name `best` = shape-pick (the floor-respecting
        # one), since prior code path already filtered avg_scores by the
        # floor before picking; this keeps cv.best meaning the same thing.
        best = shape_best

#        print(" ")
#        print("k-fold relative errors on RR")
#        print( np.std(np.exp(RRs),axis=-1)/np.mean(np.exp(RRs),axis=-1))
#        print(" ")
#        print("mean RR error relative to final score")
#        #print( np.mean(errorsRR,axis=-1)/np.mean(np.exp(RRs),axis=-1))
#        print( np.mean(errorsRR,axis=-1)/np.abs(avg_scores))
        if not np.any(np.isfinite(avg_scores)):
            pdb.set_trace()
        # Asymptotic covfac correction: CV trains on N*(k-1)/k data; the
        # AMISE-optimal kernel covariance scales as Sigma ~ N^(-2/(d+4)), so the
        # CV-picked covfac (sized for the smaller fold) is shrunk by
        # `asymcorr_cov = (k/(k-1))^(-2/(d+4))` for the final estimator.
        #
        # Shrinkage adjustment: when `shrinkage = alpha > 0`, the kernel includes
        # the term `alpha * silverman^2 * Sigma_global` whose Silverman^2 uses
        # `self.data.shape[0]`. That N changes from N*(k-1)/k during CV to
        # N at final-build time, so Silverman^2 self-adjusts the alpha-component
        # by exactly asymcorr_cov. Applying asymcorr_cov to covfac too
        # double-corrects that part. The shrinkage-aware factor below mixes
        # local (uncorrected by Silverman, needs full asymcorr) and global
        # (self-corrected by Silverman, needs none) contributions:
        #     correction(alpha) = asymcorr_cov / (1 - alpha*(1 - asymcorr_cov))
        # which reduces to asymcorr_cov at alpha=0 and 1.0 at alpha=1.
        asymcorr_cov = (float(nfolds)/float(nfolds-1.))**(-2.0/(4.0+data.shape[1]))
        print("asymptotic correction factor (cov-scale, alpha=0): ", asymcorr_cov)

        def _build_at(idx):
            alpha = shrinkages[idx[4]]
            correction = asymcorr_cov / (1.0 - alpha * (1.0 - asymcorr_cov))
            return adaptiveKDE(data,
                               scalings=resolved_scalings_list[idx[5]],
                               nn=nns[idx[2]],
                               # Per-scaling raw covfac (post-2026-05-20 fix):
                               # covfacs[sc_idx, k] = covfacs_user[k] * natural[sc_idx]
                               covfac=covfacs[idx[5], idx[0]] * correction,
                               covalpha=covalphas[idx[1]],
                               shrinkage=alpha,
                               shrinkage_target=shrinkage_target,
                               K_pool=K_pool, weights=self.weights)

        # When the two picks coincide, build only once and alias.
        # Stash everything pick_at_floor needs. The build helper is also
        # stored as a closure-capturing lambda to avoid re-computing
        # asymcorr / pulling grids into the call-time scope.
        self._build_at = _build_at
        self._kde_cache = {}
        self.neff_med_grid = neff_med_grid if floor_active else None
        # Rate-weighted ESS grids (median across folds). None when the caller
        # didn't supply the geometry inputs needed to compute them.
        self.sky_ess_grid = sky_ess_grid
        self.vinf_ess_grid = vinf_ess_grid
        self._sky_bins = sky_bins
        self._vinf_bins = int(vinf_bins)
        self._asymcorr = asymcorr_cov
        self._covfacs = covfacs
        self._covalphas = covalphas
        self._nns = nns
        self._vfacs = vfacs
        self._shrinkages = shrinkages
        self._resolved_scalings_list = resolved_scalings_list

        if rate_best == shape_best:
            self.kde_shape = _build_at(shape_best)
            self.kde_rate = self.kde_shape
        else:
            self.kde_shape = _build_at(shape_best)
            self.kde_rate = _build_at(rate_best)
        # Seed the floor-cache with the two precomputed picks. pick_at_floor()
        # for the same (or coincident) floor value returns the cached object.
        self._kde_cache[0.0] = self.kde_rate
        if floor_active:
            self._kde_cache[float(neff_floor)] = self.kde_shape
        # Backwards-compat alias: existing callers using `cv.kde` get the
        # shape-faithful (floor-respecting) version, matching the post-floor
        # behaviour they had been seeing.
        self.kde = self.kde_shape

        self.avg_scores = avg_scores
        self.std_scores = std_scores
        self.sem_scores = sem_scores
        self.combined_scores = combined_scores
        # Per-fold raw arrays (added 2026-05-24 for the RR-variance diagnostic).
        # Useful for inspecting whether MC noise in the RR bootstrap term is
        # what's making CV pick near-delta kernels on flat-density scenarios.
        self.RRs_per_fold = RRs                     # (n_cf, n_ca, n_nn, n_vf, n_sh, n_sc, n_folds)
        self.scores_per_fold = signs * np.exp(scores)
        self.errorsRR_per_fold = errorsRR
        self.stability_lambda = stability_lambda
        self.scalings_labels = scalings_labels
        self.best = best                    # = shape_best (back-compat alias)
        self.rate_best = rate_best
        self.shape_best = shape_best
        self.neffs = neffs
        self.neff_floor = neff_floor
        self.neff_floor_active = bool(floor_active)
        # User-facing pick values (covfac=1 ~ silverman^2*|Sigma_data|^(1/d) in the
        # picked-scaling coords, covalpha=0 ~ Abramson volume-equalize). The
        # asymptotic correction is the train-on-(N-1)/N correction, not
        # part of the natural anchor - back it out before user-facing
        # translation. Correction depends on the per-pick shrinkage:
        # correction(alpha) = asymcorr_cov / (1 - alpha*(1 - asymcorr_cov))
        # (matches _build_at exactly).
        def _correction_at(idx):
            alpha = shrinkages[idx[4]]
            return asymcorr_cov / (1.0 - alpha * (1.0 - asymcorr_cov))
        self.natural_covfac_rate = float(natural_covfacs_per_scaling[rate_best[5]])
        self.natural_covfac_shape = float(natural_covfacs_per_scaling[shape_best[5]])
        self.rate_covfac_user = float(self.kde_rate.covfac_overall / _correction_at(rate_best) / self.natural_covfac_rate)
        self.rate_covalpha_user = float(self.kde_rate.covalpha_overall - self.natural_covalpha)
        self.shape_covfac_user = float(self.kde_shape.covfac_overall / _correction_at(shape_best) / self.natural_covfac_shape)
        self.shape_covalpha_user = float(self.kde_shape.covalpha_overall - self.natural_covalpha)
        if floor_active:
            self.neff_n_eligible = int(np.sum(neff_med_grid >= neff_floor))
            self.neff_n_total = int(np.sum(np.isfinite(neff_med_grid)))
            self.neff_med_at_pick = float(neff_med_grid[shape_best])
            self.neff_med_at_rate_pick = float(neff_med_grid[rate_best])
        else:
            self.neff_n_eligible = None
            self.neff_n_total = None
            self.neff_med_at_pick = None
            self.neff_med_at_rate_pick = None

    def typical_h_scaled(self):
        """Single-scalar summary of the production KDE's kernel size relative
        to data spread, in the scaled coordinates the kdtree uses.

        Definition (revised 2026-05-21):

            h_per_axis_i = sqrt(cf_eff_i * diag(Sigma_eff_i)) / data_std_per_axis
            h_per_pt_i   = min over axes of h_per_axis_i
            h_typ        = median over points of h_per_pt_i
            return       max(h_typ, 1/N^(1/d))

        The min-across-axes captures the *worst-bridged* axis for a typical
        kernel - the axis where adjacent training points are most likely to
        fall outside the kernel support. Old `trace(Sigma)/d` averaged across
        axes and missed pathological cases where the kernel was near-delta
        on a single compressed axis (e.g. `narrow` scaling on stream-like
        data compresses v_y by 1500x, but trace-averaging hid this).

        The clamp at `1/N^(1/d)` is the per-axis inter-point spacing for
        unit-spread scaled data. Kernels narrower than this can't bridge
        typical training-point spacing on the worst axis, so the floor
        formula in `floor_for_dim` should not keep shrinking below this.

        Used by `floor_for_dim` to translate an output-dimensional N_eff
        target into a 6D floor: at h<1 (typical), low-d outputs need much
        smaller 6D floors than high-d outputs because the marginalising
        integral averages over the orthogonal axes.
        """
        kde = self.kde_shape
        N = kde.data.shape[0]
        d = kde.data.shape[1]
        Sigmas = kde.covariances    # already shrinkage-mixed; scaled coords
        if hasattr(kde, 'logdets'):
            cf_eff = kde.covfac_overall * np.exp(kde.covalpha_overall * kde.logdets)
        else:
            cf_eff = np.full(Sigmas.shape[0], kde.covfac_overall)
        diags = np.diagonal(Sigmas, axis1=1, axis2=2)    # (N, d)
        kernel_sigma_per_axis = np.sqrt(np.maximum(cf_eff[:, None] * diags, 1e-300))
        data_std_per_axis = np.maximum(kde.data.std(axis=0, ddof=1), 1e-300)        # (d,)
        h_per_axis = kernel_sigma_per_axis / data_std_per_axis[None, :]
        h_per_pt_min = h_per_axis.min(axis=1)
        h_typ = float(np.median(h_per_pt_min))
        h_floor_pp = 1.0 / max(N, 1) ** (1.0 / float(d))
        return max(h_typ, h_floor_pp)

    def floor_for_dim(self, d_m, target_output_neff=60.0):
        """Translate an output-dimensionality target into a 6D N_eff floor
        via the asymptotic dimensional scaling

            N_eff^(6D) ~ N_eff_target_output * h^(6 - d_m)

        where h is `typical_h_scaled`. The default `target_output_neff=60`
        is calibrated so that a sky-map output (effective d_m~5 once
        rate-weight localisation is accounted for) lands at the empirical
        floor~30 we found benchmark-competitive on cold+hot at h~0.5.
        Users with very different N or feature scales should re-anchor
        against their own benchmark.

        Effective d_m is the user's responsibility. The dimensionality of
        the output's *integrand support*, not just the histogram-axis
        count, is what enters: rate-weighted "1D" outputs (e.g.
        costheta histograms weighted by w(v)~1/v^2) have d_m_eff ~ 4 not 1
        because the focusing weight concentrates the integrand on a 3D
        v-region, leaving only ~2 axes truly integrated.
        """
        h = self.typical_h_scaled()
        return float(target_output_neff) * h ** (6.0 - float(d_m))

    def pick_for_dim(self, d_m, target_output_neff=60.0):
        """Pick the production KDE for an output of effective dimensionality
        `d_m`. Convenience wrapper over `floor_for_dim` + `pick_at_floor`.

        Special case: d_m=0 (the rate scalar) hard-pins to floor=0 per the
        multi-pick scheme (paper section sec:dual-pick). The rate is unbiased on
        average regardless of kernel overlap, so no kernel-overlap floor
        applies. Returning `pick_at_floor(0)` ensures we get back
        `self.kde_rate` even when `floor_for_dim(0) = 60*h^6` is tiny but
        positive (which would otherwise route through the masking codepath
        and incorrectly filter rate_best out when its neff_med is NaN -
        e.g. because the kernel-radius guardrail fired during CV at the
        rate_best position).
        """
        if d_m <= 0:
            return self.pick_at_floor(0.0)
        return self.pick_at_floor(self.floor_for_dim(d_m, target_output_neff))

    def pick_at_floor(self, neff_min):
        """Return an `adaptiveKDE` built at the best (cf, ca, nn, vf, sh,
        scaling) hyperparameter tuple subject to `median N_eff >= neff_min`
        across CV folds (median measured at the user-supplied
        `neff_eval_points`). When `neff_min <= 0` returns the unconstrained
        rate-pick; when no entry passes `neff_min` falls back to the
        rate-pick. Cached, so repeated calls at the same floor are free.

        Per-task floor selection: see `pick_for_dim` for the dimensional
        formula that translates an output's effective dimensionality into
        a recommended 6D floor.
        """
        if neff_min is None or neff_min <= 0.0:
            return self.kde_rate
        key = float(neff_min)
        if key in self._kde_cache:
            return self._kde_cache[key]
        if self.neff_med_grid is None:
            # No floor was active during CV -> no neff grid to mask against;
            # all picks coincide with the rate-pick.
            self._kde_cache[key] = self.kde_rate
            return self.kde_rate
        eligible = self.neff_med_grid >= key
        if not np.any(eligible):
            self._kde_cache[key] = self.kde_rate
            return self.kde_rate
        # Use combined_scores (avg - lambda*SEM) so per-task picks respect the
        # stability tiebreaker the rate/shape picks already used. Falls back
        # to avg_scores when stability_lambda=0 (combined == avg).
        score_for_pick = getattr(self, 'combined_scores', self.avg_scores)
        masked = np.where(eligible, score_for_pick, -np.inf)
        # Score-collapse fallback (2026-05-22): if the floor rejects all the
        # picks where CV could meaningfully differentiate (because the kernel-
        # radius guardrail nuked N_eff on the high-alpha arm of the grid, say),
        # the argmax of `masked` selects the loudest noise floor among picks
        # with effectively-zero ISE-CV scores. Empirically observed on
        # disk_stream at N=1000 where: rate-pick at (ca=0.2, narrow_local,
        # cf=1.78) had a real score with NaN N_eff (guardrail fired during
        # eval), so the floor rejected it; the only floor-eligible region
        # was (ca<=0, narrow/narrow_local) with scores ~1e-14 -> noise-driven
        # near-delta pick -> catastrophic sky-TV. Detect this and fall back
        # to kde_rate, which at least represents a real CV optimum.
        # Heuristic: if best eligible score is <10% of best unconstrained
        # score, the floor has knocked out the meaningful picks.
        best_eligible = float(np.nanmax(masked))
        best_unconstrained = float(np.nanmax(np.where(np.isfinite(score_for_pick), score_for_pick, -np.inf)))
        if (best_unconstrained > 0
                and best_eligible < 0.1 * best_unconstrained):
            self._kde_cache[key] = self.kde_rate
            return self.kde_rate
        idx = np.unravel_index(np.nanargmax(masked), masked.shape)
        kde = self._build_at(idx)
        self._kde_cache[key] = kde
        return kde



    def _pick_at_ess_floor(self, ess_grid, ess_min, cache_key):
        """Shared body for pick_at_sky_floor and pick_at_vinf_floor. Same
        score-collapse fallback semantics as pick_at_floor - falls back to
        kde_rate when (a) the grid is unavailable, (b) no entry passes the
        floor, or (c) the best eligible score is < 10% of the unconstrained
        best (CV grid collapsed into noise after the floor mask)."""
        if ess_min is None or ess_min <= 0.0:
            return self.kde_rate
        if cache_key in self._kde_cache:
            return self._kde_cache[cache_key]
        if ess_grid is None:
            self._kde_cache[cache_key] = self.kde_rate
            return self.kde_rate
        eligible = ess_grid >= ess_min
        if not np.any(eligible):
            self._kde_cache[cache_key] = self.kde_rate
            return self.kde_rate
        score_for_pick = getattr(self, 'combined_scores', self.avg_scores)
        masked = np.where(eligible, score_for_pick, -np.inf)
        best_eligible = float(np.nanmax(masked))
        best_unconstrained = float(np.nanmax(np.where(np.isfinite(score_for_pick), score_for_pick, -np.inf)))
        if (best_unconstrained > 0
                and best_eligible < 0.1 * best_unconstrained):
            self._kde_cache[cache_key] = self.kde_rate
            return self.kde_rate
        idx = np.unravel_index(np.nanargmax(masked), masked.shape)
        kde = self._build_at(idx)
        self._kde_cache[cache_key] = kde
        return kde

    def pick_at_sky_floor(self, min_sky_ess):
        """Pick the highest-CV-score grid entry whose median (across folds)
        Kish ESS on the (cos theta, phi) sky histogram clears `min_sky_ess`.

        The sky-ESS measures rate-weight concentration across the 12x24
        encounter-sphere bins (max value = 288 if rate-weight is uniform).
        Use this instead of `pick_for_dim(d_m=5)` when sky-map fidelity is
        the goal - the per-eval N_eff floor measures kernel coverage at
        sphere-probe points and doesn't capture concentration in the actual
        sky-aggregation that drives the sky-map TV diagnostic.

        Requires `sky_bin_costheta/phi` + `rate_weight_geom_factor` to have
        been supplied at construction time; otherwise falls back to kde_rate.
        """
        return self._pick_at_ess_floor(self.sky_ess_grid, min_sky_ess,
                                        cache_key=('sky_ess', float(min_sky_ess)))

    def pick_at_vinf_floor(self, min_vinf_ess):
        """Pick the highest-CV-score grid entry whose median (across folds)
        Kish ESS on the v_inf histogram clears `min_vinf_ess`.

        Analogous to `pick_at_sky_floor` but for the 1D v_inf marginal
        (default 20 bins). Requires `vinf_bin_coord` + `rate_weight_geom_factor`
        to have been supplied at construction time.
        """
        return self._pick_at_ess_floor(self.vinf_ess_grid, min_vinf_ess,
                                        cache_key=('vinf_ess', float(min_vinf_ess)))

    def __call__(self, points, covfac=1.0, show_contribs=False):
        return self.kde(points,covfac=covfac)

    def data_side_neff(self, points, eval_weights=None, covfac=1.0, covalpha=0.0):
        """Delegate to the underlying adaptiveKDE (the current self.kde -
        usually kde_shape; pass kde_rate / kde_sky / kde_vinf directly if
        you want the data-side N_eff for a different pick).
        See adaptiveKDE.data_side_neff for definition."""
        return self.kde.data_side_neff(points, eval_weights=eval_weights, covfac=covfac, covalpha=covalpha)




