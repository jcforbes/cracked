"""Regime map: which KDE method wins where, with task-specific panels.

Axes:
  x = data structural complexity (categorical, ordinal):
        smooth -> anisotropic -> multi-scale -> narrow -> many-narrow
  y = output dimensionality of the task:
        the test data (preview)
        6D density fhat vs f
        3D velocity DF
        2D sky map
        1D marginal
        0D rate

Each cell shows the analytic-DF task target.  d=3 and d=2 heatmaps
get truth-contour overlays for shape clarity; d=1 gets a smooth truth
curve; d=6 shows the truth log-density distribution with the winning
estimator's bias-shifted reconstruction overlaid; d=0 shows a bar chart
of R/R_an across the four method families with the winner emphasized.
Border + badge mark the winner from diagnostic_table.md.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpath_effects
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter, gaussian_filter1d

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, os.pardir, "tests"))
sys.path.insert(0, os.path.join(HERE, os.pardir, "src"))
from test_rate_sphere_analytic import (
    make_isotropic_sampler, make_ring_sampler, make_cold_hot_sampler,
    make_disk_stream_sampler, make_spiky_ball_sampler,
    make_isotropic_df, make_ring_df, make_cold_hot_df,
    make_disk_stream_df, make_spiky_ball_df,
    N0,
)
from cracked import G, pcperau

# Rate-integrand constants (matching the test suite)
QMAX_AU = 5.0
QMAX_PC = QMAX_AU * pcperau
GM_TARGET = G * 1.0
R_SPHERE = 0.1
V_ESC_SQ = 2.0 * GM_TARGET / R_SPHERE


# Scenarios - each carries its analytic sampler AND density callable.
SCENARIOS = [
    dict(label="smooth\nisotropic", sampler=make_isotropic_sampler(1.0), df=make_isotropic_df(N0, 1.0), axis_pos=(-5.0, 5.0), axis_vel=(-3.0, 3.0), pos_focus=None),
    dict(label="anisotropic\n(velocity ring)", sampler=make_ring_sampler(0.5, 0.05, 0.05), df=make_ring_df(N0, 0.5, 0.05, 0.05), axis_pos=(-5.0, 5.0), axis_vel=(-0.9, 0.9), pos_focus=None),
    dict(label="multi-scale\n(cold + hot)", sampler=make_cold_hot_sampler(0.4, 10.0, 0.2), df=make_cold_hot_df(N0, 0.4, 10.0, 0.2), axis_pos=(-5.0, 5.0), axis_vel=(-15.0, 15.0), pos_focus=None),
    dict(label="narrow feature\n(disk stream)", sampler=make_disk_stream_sampler(R_ring=8000.0, sigma_R=1.0, sigma_z=1.0, sigma_t=0.1, v_circ=220.0, width=25.0, height=12.5, v_sun_peculiar=(-5.0, 5.0, 0.0)), df=make_disk_stream_df(1.0, R_ring=8000.0, sigma_R=1.0, sigma_z=1.0, sigma_t=0.1, v_circ=220.0, width=25.0, height=12.5, v_sun_peculiar=(-5.0, 5.0, 0.0)), axis_pos=(-1.6e4, 1.6e4), axis_vel=(-300.0, 300.0), pos_focus=50.0),
    dict(label="many narrow\n(spiky ball)", sampler=make_spiky_ball_sampler(25, 100.0, 0.3, 0.3, 0.05, 3.0)[0], df=make_spiky_ball_df(1.0, 25, 100.0, 0.3, 0.3, 0.05, 3.0), axis_pos=(-60.0, 60.0), axis_vel=(-4.0, 4.0), pos_focus=2.0),
]


# Per-scenario rate bars (R/R_an in %).  Each bar = best variant of
# that method family from diagnostic_table.md; for cvGauss that's
# whatever scalings_grid=[None,auto,narrow] picked; for cvAdapt it's
# min(default, narrow=smart) distance to 100.
RATES_PCT = {
    'smooth\nisotropic':           [80,  78, 79, 83],
    'anisotropic\n(velocity ring)': [86,  85, 95, 103],
    'multi-scale\n(cold + hot)':    [36, 213, 104, 87],
    'narrow feature\n(disk stream)': [1,   1, 72,  47],
    'many narrow\n(spiky ball)':    [0,   16, 68,  29],
}
METHODS_BARS = ['scipy', 'cvgauss', 'cvadapt', 'nf']

# Per-cell median log10(fhat/f) bias for the d=6 row's winner - used to
# build the bias-shifted fhat histogram.  Pulled directly from
# diagnostic_table.md log10(fhat/f) median column for each scenario's
# winning estimator.
D6_BIAS = {
    'smooth\nisotropic':           -0.05,  # NF
    'anisotropic\n(velocity ring)': -0.10, # NF
    'multi-scale\n(cold + hot)':   -0.16,  # NF
    'narrow feature\n(disk stream)': -0.78, # NF (least-bad failing)
    'many narrow\n(spiky ball)':   -0.81,  # cvAdapt (least-bad failing)
}


TASKS = [
    dict(d=6, label="6D density", sub=r"$\hat f(\vec x, \vec v)$ at arbitrary points", winners=['nf', 'nf', 'nf', 'nf', 'cvadapt']),
    dict(d=3, label="3D velocity DF", sub=r"$\hat f_v(\vec v)$", winners=['nf', 'nf', 'nf', 'nf', 'cvadapt']),
    dict(d=2, label="2D sky map", sub=r"$\mathrm{d}\mathcal{R}/\mathrm{d}\Omega(\cos\theta, \varphi)$", winners=['nf', 'nf', 'scipy', 'scipy', 'scipy']),
    dict(d=1, label="1D marginal", sub=r"$\mathrm{d}\mathcal{R}/\mathrm{d}\log v_\infty$", winners=['nf', 'nf', 'cvadapt', 'nf', 'nf']),
    dict(d=0, label="0D rate", sub=r"$\mathcal{R} = \int f\,v\,\sigma_{\rm geom}\,dv$", winners=['nf', 'nf', 'cvadapt', 'cvadapt', 'cvadapt']),
]


COL_SCIPY   = "#2b6cb0"
COL_CVGAUSS = "#38a169"
COL_CVADAPT = "#c53030"
COL_NF      = "#805ad5"
COL_NONE    = "#a0aec0"
WINNER_COLOR = dict(scipy=COL_SCIPY, cvgauss=COL_CVGAUSS, cvadapt=COL_CVADAPT, nf=COL_NF, none=COL_NONE)
WINNER_LABEL = dict(scipy="scipy", cvgauss="cvGauss-narrow", cvadapt="cvAdaptive", nf="NF ensemble", none="(no clear winner)")


def rate_weights(v_xyz):
    """w(v) = v * sigma_geom(v) for unbound particles; 0 for bound."""
    v2 = np.sum(v_xyz ** 2, axis=1)
    speed = np.sqrt(np.maximum(v2, 1e-30))
    focusing = 1.0 + 2.0 * GM_TARGET / (QMAX_PC * np.maximum(v2, 1e-30))
    w = speed * focusing
    w[v2 < V_ESC_SQ] = 0.0
    return w


def project_sky(v_xyz):
    speed = np.maximum(np.sqrt(np.sum(v_xyz ** 2, axis=1)), 1e-12)
    return v_xyz[:, 2] / speed, np.arctan2(v_xyz[:, 1], v_xyz[:, 0])


def overlay_contours(ax, h2d, extent, levels_frac=(0.20, 0.50, 0.80)):
    """Smooth-and-overlay truth contour lines on a histogram cell."""
    smooth = gaussian_filter(h2d.T, sigma=1.0)
    vmax = smooth.max()
    if vmax <= 0:
        return
    lv = np.array(levels_frac) * vmax
    lv = np.unique(lv[lv > 0])
    if len(lv) == 0:
        return
    xs = np.linspace(extent[0], extent[1], smooth.shape[1])
    ys = np.linspace(extent[2], extent[3], smooth.shape[0])
    X, Y = np.meshgrid(xs, ys)
    ax.contour(X, Y, smooth, levels=lv, colors='#7fffd4', linewidths=0.7, alpha=0.9)


def render_test_data(ax, samples, axis_pos, rng):
    """Top row: position scatter + velocity arrows colored by v_z."""
    pos = samples[:, :2]
    in_view = ((pos[:, 0] >= axis_pos[0]) & (pos[:, 0] <= axis_pos[1])
               & (pos[:, 1] >= axis_pos[0]) & (pos[:, 1] <= axis_pos[1]))
    sub = samples[in_view]
    if len(sub) > 140:
        idx = rng.choice(len(sub), 140, replace=False)
        sub = sub[idx]
    if len(sub) == 0:
        ax.text(0.5, 0.5, "-", ha='center', va='center', transform=ax.transAxes, color='#888', fontsize=10)
        ax.set_xlim(axis_pos); ax.set_ylim(axis_pos)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        return
    span = axis_pos[1] - axis_pos[0]
    v_arrow = np.hypot(sub[:, 3], sub[:, 4])
    v95 = max(float(np.percentile(v_arrow, 95)), 1e-12)
    a_per_v = span * 0.10 / v95
    c95 = max(float(np.percentile(np.abs(sub[:, 5]), 95)), 1e-12)
    ax.scatter(sub[:, 0], sub[:, 1], s=11, c='#1a365d', alpha=0.7, lw=0)
    ax.quiver(sub[:, 0], sub[:, 1], sub[:, 3] * a_per_v, sub[:, 4] * a_per_v, sub[:, 5], cmap='RdBu_r', clim=(-c95, +c95), angles='xy', scale_units='xy', scale=1.0, width=0.010, alpha=0.85)
    ax.set_aspect('equal')
    ax.set_xlim(axis_pos); ax.set_ylim(axis_pos)
    ax.set_xticks([]); ax.set_yticks([])


def render_d6(ax, samples, df_callable, winner_bias, winner_color):
    """d=6: log10(f) histogram (truth) + bias-shifted log10(fhat)."""
    f = df_callable(samples)
    log_f = np.log10(np.maximum(f, 1e-300))
    log_f = log_f[np.isfinite(log_f)]
    if len(log_f) < 10:
        ax.text(0.5, 0.5, "-", ha='center', va='center', transform=ax.transAxes, color='#888', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        return
    lo = float(np.percentile(log_f, 1))
    hi = float(np.percentile(log_f, 99))
    span = max(hi - lo, 0.5)
    lo -= 0.1 * span; hi += 0.1 * span
    if winner_bias < 0:
        lo += winner_bias
    else:
        hi += winner_bias
    bins = np.linspace(lo, hi, 35)
    centers = 0.5 * (bins[1:] + bins[:-1])
    h_f, _ = np.histogram(log_f, bins=bins)
    h_fh, _ = np.histogram(log_f + winner_bias, bins=bins)
    norm = max(h_f.max(), 1)
    h_f = h_f / norm
    h_fh = h_fh / norm
    ax.fill_between(centers, 0.0, h_f, color='#444', alpha=0.55, lw=0)
    ax.plot(centers, h_f, color='#222', lw=0.9)
    ax.plot(centers, h_fh, color=winner_color, lw=1.7)
    # Bias annotation
    bias_txt = f"bias\n{winner_bias:+.2f} dex"
    ax.text(0.03, 0.95, bias_txt, transform=ax.transAxes, ha='left', va='top', fontsize=7.0, color=winner_color, fontweight='bold', bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none', alpha=0.85))
    ax.set_xlim(lo, hi); ax.set_ylim(0.0, 1.10)
    ax.set_xticks([]); ax.set_yticks([])


def render_d3(ax, samples, axis_vel):
    """d=3: 2D velocity heatmap with truth contour overlay."""
    v = samples[:, 3:5]
    h, _, _ = np.histogram2d(v[:, 0], v[:, 1], bins=32, range=[axis_vel, axis_vel])
    nz = h[h > 0]
    vmax = float(np.percentile(nz, 99)) * 1.05 if len(nz) else 1.0
    ax.imshow(h.T, extent=(*axis_vel, *axis_vel), origin='lower', cmap='magma', aspect='equal', vmin=0.0, vmax=vmax, interpolation='nearest')
    overlay_contours(ax, h, (*axis_vel, *axis_vel))
    ax.set_xticks([]); ax.set_yticks([])


def render_d2(ax, samples, weights):
    """d=2: rate-weighted sky map with truth contour overlay."""
    costh, phi = project_sky(samples[:, 3:6])
    h, _, _ = np.histogram2d(costh, phi, bins=[18, 28], range=[(-1, 1), (-np.pi, np.pi)], weights=weights)
    nz = h[h > 0]
    vmax = float(np.percentile(nz, 99)) * 1.05 if len(nz) else 1.0
    ax.imshow(h.T, extent=(-1, 1, -np.pi, np.pi), origin='lower', cmap='magma', aspect='auto', vmin=0.0, vmax=vmax, interpolation='nearest')
    overlay_contours(ax, h, (-1, 1, -np.pi, np.pi))
    ax.set_xticks([]); ax.set_yticks([])


def render_d1(ax, samples, weights):
    """d=1: rate-weighted log10(vinf) histogram + smooth truth curve."""
    speed = np.linalg.norm(samples[:, 3:6], axis=1)
    safe = speed > 0
    if safe.sum() == 0:
        ax.text(0.5, 0.5, "-", ha='center', va='center', transform=ax.transAxes, color='#888', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        return
    log_v = np.log10(np.maximum(speed[safe], 1e-3))
    w = weights[safe]
    if w.sum() > 0:
        order = np.argsort(log_v)
        cum = np.cumsum(w[order]) / w.sum()
        lo = log_v[order][np.searchsorted(cum, 0.01)]
        hi = log_v[order][np.searchsorted(cum, 0.99)]
        if hi <= lo:
            hi = lo + 1.0
    else:
        lo, hi = float(log_v.min()), float(log_v.max())
    counts, edges = np.histogram(log_v, bins=40, range=(lo, hi), weights=w)
    centers = 0.5 * (edges[1:] + edges[:-1])
    if counts.max() > 0:
        counts = counts / counts.max()
    truth = gaussian_filter1d(counts, sigma=1.8)
    ax.fill_between(centers, 0.0, counts, color='#888', alpha=0.55, linewidth=0)
    ax.plot(centers, truth, color='#7fffd4', lw=1.6, path_effects=[mpath_effects.withStroke(linewidth=2.6, foreground='#111')])
    ax.set_xlim(lo, hi); ax.set_ylim(0.0, 1.10)
    ax.set_xticks([]); ax.set_yticks([])


def render_d0(ax, rates_pct, winner_idx):
    """d=0: bar chart of R/R_an across method families."""
    methods = METHODS_BARS
    colors = [WINNER_COLOR[m] for m in methods]
    x = np.arange(len(methods))
    clip = 220.0
    drawn = np.minimum(rates_pct, clip)
    bars = ax.bar(x, drawn, color=colors, alpha=0.5, edgecolor='#222', linewidth=0.5)
    bars[winner_idx].set_alpha(1.0)
    bars[winner_idx].set_linewidth(2.0)
    ax.axhline(100.0, color='#222', linestyle='--', linewidth=0.7, alpha=0.7)
    for r, b in zip(rates_pct, bars):
        if r > clip:
            ax.text(b.get_x() + b.get_width() / 2, clip * 0.95, f"{int(r)}", ha='center', va='top', fontsize=7, color='white', fontweight='bold')
        elif r < 6 and r > 0:
            ax.text(b.get_x() + b.get_width() / 2, 8, f"{r}", ha='center', va='bottom', fontsize=7, color='#222', fontweight='bold')
    ax.set_xlim(-0.6, len(methods) - 0.4)
    ax.set_ylim(0, clip * 1.05)
    ax.set_xticks([]); ax.set_yticks([])


def draw_pictogram(ax, kind):
    """Pictogram for the left-side row label."""
    rng = np.random.default_rng(3 + hash(str(kind)) % 1000)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if kind == 'test':
        # Scatter + arrows
        xs = rng.uniform(-1, 1, 20)
        ys = rng.uniform(-1, 1, 20)
        ax.scatter(xs, ys, s=10, c='#1a365d', alpha=0.7, lw=0)
        for i in range(0, len(xs), 2):
            ax.annotate('', xy=(xs[i] + 0.2, ys[i] + 0.05), xytext=(xs[i], ys[i]), arrowprops=dict(arrowstyle='->', color='#c00', lw=0.7))
        ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.set_aspect('equal')
    elif kind == 0:
        ax.bar([0], [0.78], color="#888", edgecolor="#222", width=0.5)
        ax.text(0, 0.92, r"$\mathcal{R}$", ha='center', va='bottom', fontsize=13, fontweight='bold')
        ax.set_xlim(-0.6, 0.6); ax.set_ylim(0, 1.15)
    elif kind == 1:
        xs = np.linspace(0, 1, 50)
        ys = np.exp(-((xs - 0.5) / 0.20) ** 2)
        ax.fill_between(xs, 0, ys, color="#888", alpha=0.85)
        ax.plot(xs, ys, color="#222", lw=1.0)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.15)
    elif kind == 2:
        h = rng.normal(0.6, 0.2, size=(6, 10))
        h += 0.7 * np.exp(-(np.arange(6)[:, None] - 4) ** 2 / 1.2)
        ax.imshow(h, aspect='auto', origin='lower', cmap='viridis', extent=[-1, 1, -np.pi, np.pi])
    elif kind == 3:
        t = np.linspace(0, 2*np.pi, 60)
        ax.scatter(np.cos(t) + 0.05*rng.standard_normal(60), np.sin(t) + 0.05*rng.standard_normal(60), s=8, c=np.cos(t), cmap='RdBu_r', alpha=0.7)
        ax.set_aspect('equal'); ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
    elif kind == 6:
        # Two overlaid density curves (fhat vs f)
        xs = np.linspace(0, 1, 80)
        f = np.exp(-((xs - 0.6) / 0.18) ** 2)
        fhat = np.exp(-((xs - 0.52) / 0.22) ** 2)  # biased + smoothed
        ax.fill_between(xs, 0, f, color="#888", alpha=0.55)
        ax.plot(xs, f, color="#222", lw=1.0)
        ax.plot(xs, fhat, color="#805ad5", lw=1.4)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.15)


def collect_samples(scenarios, N=20000, rng_seed=11):
    rng = np.random.default_rng(rng_seed)
    out = []
    for sc in scenarios:
        full = sc['sampler'](rng, N)
        if sc['pos_focus'] is not None:
            r = np.linalg.norm(full[:, :3], axis=1)
            focus = full[r <= sc['pos_focus']]
            if len(focus) < 200:
                idx = np.argsort(r)[:max(500, len(full) // 50)]
                focus = full[idx]
        else:
            focus = full
        out.append((full, focus))
    return out


def main():
    n_cols = len(SCENARIOS)
    n_body_rows = len(TASKS)
    n_rows = n_body_rows + 1  # +1 for the "test data" header row

    samples_per_scenario = collect_samples(SCENARIOS)

    fig = plt.figure(figsize=(3.0 + 2.4 * n_cols, 2.6 * n_rows + 1.8))
    gs = fig.add_gridspec(n_rows, n_cols + 1, left=0.04, right=0.985, top=0.83, bottom=0.05, width_ratios=[1.0] + [1.55] * n_cols, wspace=0.18, hspace=0.30)

    fig.text(0.5, 0.967, "Which KDE for which task? - regime map", ha='center', va='center', fontsize=15, fontweight='bold')
    fig.text(0.5, 0.940,
              "Each cell shows the analytic-DF task target for that "
              "(data, task) combination.  d=2/d=3/d=1 overlay smoothed "
              "truth contours/curves; d=6 overlays $\\hat f$ shifted by "
              "the winning estimator's measured log10 bias.",
              ha='center', va='center', fontsize=9.5, color='#444',
              style='italic')

    callout_y = 0.890
    fig.add_artist(Rectangle((0.20, callout_y - 0.022), 0.60, 0.035, facecolor='#f7f2fb', edgecolor=COL_NF, linewidth=1.5, transform=fig.transFigure))
    fig.text(0.50, callout_y - 0.004,
              "generate samples from $\\hat f$:  "
              "-> NormalizingFlow ensemble (only method with a true "
              "generator; cv*KDE.draw is OK for few samples but lives on "
              "training support)",
              ha='center', va='center', fontsize=10,
              color='#5a2a8a', fontweight='bold')

    # Column headers
    for c, sc in enumerate(SCENARIOS):
        l = 0.04 + (1.0 + (c + 0.5) * 1.55) / (1.0 + n_cols * 1.55) * (0.985 - 0.04)
        fig.text(l, 0.857, sc['label'], ha='center', va='center', fontsize=11, fontweight='bold')

    fig.text(0.5 + 0.5 * (0.04 + (1.0 / (1.0 + n_cols * 1.55)) * (0.985 - 0.04)), 0.020, "data structural complexity  ->", ha='center', va='center', fontsize=11, fontweight='bold', color='#222')
    fig.text(0.012, 0.45, "task output dimension  v", ha='center', va='center', fontsize=11, fontweight='bold', color='#222', rotation=90)

    # -- Header row: "the test" - position scatter + velocity arrows
    ax_picto = fig.add_subplot(gs[0, 0])
    draw_pictogram(ax_picto, 'test')
    bbox = ax_picto.get_position()
    fig.text(bbox.x0 - 0.010, bbox.y0 + bbox.height * 0.5, "data", ha='right', va='center', fontsize=12, fontweight='bold', color='#222')
    fig.text(bbox.x0 + bbox.width * 0.5, bbox.y0 - 0.005, "the test data", ha='center', va='top', fontsize=11, fontweight='bold')
    fig.text(bbox.x0 + bbox.width * 0.5, bbox.y0 - 0.030, r"$\vec x$ scatter, $\vec v$ arrows", ha='center', va='top', fontsize=8.5, color='#333')
    for c, sc in enumerate(SCENARIOS):
        ax = fig.add_subplot(gs[0, c + 1])
        full, _ = samples_per_scenario[c]
        render_test_data(ax, full, sc['axis_pos'], np.random.default_rng(23 * c + 5))
        for spine in ax.spines.values():
            spine.set_color('#444')
            spine.set_linewidth(1.6)

    # -- Body rows: each task
    for r, task in enumerate(TASKS):
        gs_row = r + 1
        ax_picto = fig.add_subplot(gs[gs_row, 0])
        draw_pictogram(ax_picto, task['d'])
        bbox = ax_picto.get_position()
        fig.text(bbox.x0 - 0.010, bbox.y0 + bbox.height * 0.5, f"d={task['d']}", ha='right', va='center', fontsize=15, fontweight='bold', color='#222')
        fig.text(bbox.x0 + bbox.width * 0.5, bbox.y0 - 0.005, task['label'], ha='center', va='top', fontsize=11, fontweight='bold')
        fig.text(bbox.x0 + bbox.width * 0.5, bbox.y0 - 0.030, task['sub'], ha='center', va='top', fontsize=8.5, color='#333')

        for c, sc in enumerate(SCENARIOS):
            ax = fig.add_subplot(gs[gs_row, c + 1])
            full, focus = samples_per_scenario[c]
            cell_rng = np.random.default_rng(101 + 23 * c + 7 * r)
            winner = task['winners'][c]
            wc = WINNER_COLOR[winner]

            if task['d'] == 6:
                bias = D6_BIAS[sc['label']]
                render_d6(ax, full, sc['df'], bias, wc)
                if c == 0:
                    ax.set_xlabel(r"$\log_{10} f$", fontsize=8, labelpad=1)
            elif task['d'] == 3:
                render_d3(ax, full, sc['axis_vel'])
                if c == 0:
                    ax.set_xlabel(r"$v_x$", fontsize=8, labelpad=1)
                    ax.set_ylabel(r"$v_y$", fontsize=8, labelpad=1)
            elif task['d'] == 2:
                w = rate_weights(focus[:, 3:6])
                render_d2(ax, focus, w)
                if c == 0:
                    ax.set_xlabel(r"$\cos\theta$", fontsize=8, labelpad=1)
                    ax.set_ylabel(r"$\varphi$", fontsize=8, labelpad=1)
            elif task['d'] == 1:
                w = rate_weights(focus[:, 3:6])
                render_d1(ax, focus, w)
                if c == 0:
                    ax.set_xlabel(r"$\log_{10}\,v_\infty$", fontsize=8, labelpad=1)
            elif task['d'] == 0:
                rates = RATES_PCT[sc['label']]
                winner_idx = METHODS_BARS.index(winner) if winner in METHODS_BARS else 0
                render_d0(ax, rates, winner_idx)
                if c == 0:
                    ax.set_xticks(np.arange(len(METHODS_BARS)))
                    ax.set_xticklabels(['sci', 'cvG', 'cvA', 'NF'], fontsize=7)
                    for tl, m in zip(ax.get_xticklabels(), METHODS_BARS):
                        tl.set_color(WINNER_COLOR[m])
                        tl.set_fontweight('bold')
                    ax.set_yticks([100])
                    ax.set_yticklabels(['100%'], fontsize=7, color='#222')

            for spine in ax.spines.values():
                spine.set_color(wc)
                spine.set_linewidth(3.2)
            ax.text(0.97, 0.97, WINNER_LABEL[winner], transform=ax.transAxes, ha='right', va='top', fontsize=8.0, color=wc, fontweight='bold', bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85))

    fig.savefig("method_regime_map.pdf")
    fig.savefig("method_regime_map.png", dpi=130)
    print("wrote method_regime_map.pdf and method_regime_map.png")


if __name__ == "__main__":
    main()
