"""Cartoon decision guide for choosing a KDE method.

Branches on (task -> data structure -> N) and lands on a concrete
recommendation. Distilled from the cracked test-suite empirical
findings: stream/spike scenarios bias the recommendation toward
cvAdaptive; multi-scale (cold+hot) bias toward cvGaussian with
narrow scaling; smooth-monomodal regimes get scipy as the cheap
baseline; NF earns its keep when you also need to sample from the
estimator or when feature-counting + capacity scaling matter.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# Palette
COL_TASK     = "#cfe2f3"   # blue
COL_DATA     = "#fff2cc"   # yellow
COL_N        = "#f4cccc"   # red/pink
COL_METHOD   = "#d9ead3"   # green
COL_ARROW    = "#444444"
COL_EDGE     = "#666666"


def box(ax, x, y, w, h, text, *, facecolor, fontsize=10, fontweight="normal"):
    """Rounded-rect box anchored at top-left (x, y) with width w, height h."""
    p = FancyBboxPatch((x, y - h), w, h, boxstyle="round,pad=0.02,rounding_size=0.04", linewidth=1.2, edgecolor=COL_EDGE, facecolor=facecolor)
    ax.add_patch(p)
    ax.text(x + w / 2, y - h / 2, text, ha="center", va="center", fontsize=fontsize, fontweight=fontweight, wrap=True)
    return (x, y, x + w, y - h)   # bbox: l, t, r, b


def arrow(ax, fr, to, *, label=None, label_offset=(0.0, 0.0), label_side='right'):
    """Arrow between two anchor points. `fr`/`to` are (x, y) tuples."""
    a = FancyArrowPatch(fr, to, arrowstyle="-|>", mutation_scale=12, color=COL_ARROW, linewidth=1.2, shrinkA=2, shrinkB=2)
    ax.add_patch(a)
    if label:
        mx = 0.5 * (fr[0] + to[0]) + label_offset[0]
        my = 0.5 * (fr[1] + to[1]) + label_offset[1]
        ha = 'left' if label_side == 'right' else 'right'
        ax.text(mx, my, label, fontsize=8, color="#222222", ha=ha, va="center", style='italic', bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85))


def bottom_anchor(bbox):
    """Bottom-center of a box bbox = (l, t, r, b)."""
    l, t, r, b = bbox
    return (0.5 * (l + r), b)


def top_anchor(bbox):
    l, t, r, b = bbox
    return (0.5 * (l + r), t)


def main():
    fig, ax = plt.subplots(figsize=(13.5, 10))
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    # --- Title
    ax.text(6.75, 9.7, "Choosing a 6D density estimator for encounter-rate / stream work", ha="center", va="center", fontsize=15, fontweight="bold")
    ax.text(6.75, 9.3,
            "Decision tree, distilled from the cracked test suite. Branch labels "
            "describe the property of YOUR data/task.",
            ha="center", va="center", fontsize=9.5, color="#444444", style='italic')

    # --- Root: task
    b_task = box(ax, 4.75, 8.8, 4.0, 0.6, "What's the primary task?", facecolor=COL_TASK, fontsize=11, fontweight="bold")

    # === Branch 1: smooth/density estimation (left) ===
    # === Branch 2: rate / sky-map calculation (center) ===
    # === Branch 3: need to draw samples (right) ===

    # --- LEFT: density-estimation branch
    b_smooth_q = box(ax, 0.4, 7.7, 3.6, 0.65, "Smooth, single-scale\nvelocity dispersion?", facecolor=COL_DATA, fontsize=10)
    arrow(ax, bottom_anchor(b_task), top_anchor(b_smooth_q), label="density at\narbitrary points", label_offset=(-1.8, 0.20), label_side='right')

    b_scipy = box(ax, 0.4, 6.7, 1.7, 0.85, "scipy_kde\n(default)\n- fastest, ~ms eval", facecolor=COL_METHOD, fontsize=9)
    arrow(ax, (1.25, 7.05), top_anchor(b_scipy), label="yes", label_offset=(-0.3, 0.1), label_side='right')

    b_cvg_smooth = box(ax, 2.25, 6.7, 1.75, 0.85, "cvGaussianKDE\nscalings_grid=\n[None, auto, narrow]", facecolor=COL_METHOD, fontsize=9)
    arrow(ax, (3.15, 7.05), top_anchor(b_cvg_smooth), label="anisotropic\nor multi-scale", label_offset=(0.0, 0.0), label_side='left')

    # --- CENTER: rate / sky-map
    b_struct_q = box(ax, 4.75, 7.7, 4.0, 0.65, "Narrow features in data?\n(streams, spikes, sigma_min << sigma_global)", facecolor=COL_DATA, fontsize=10)
    arrow(ax, bottom_anchor(b_task), top_anchor(b_struct_q), label="encounter rate\nor sky map", label_offset=(0.05, 0.20), label_side='left')

    b_no_narrow = box(ax, 4.75, 6.7, 1.85, 0.85, "cvGaussianKDE\nscalings_grid=\n[None, auto, narrow]", facecolor=COL_METHOD, fontsize=9)
    arrow(ax, (5.6, 7.05), top_anchor(b_no_narrow), label="no", label_offset=(-0.25, 0.1), label_side='right')

    # Has narrow features -> check N
    b_n_q = box(ax, 6.85, 6.7, 1.9, 0.85, "Number of\nparticles N?", facecolor=COL_N, fontsize=10)
    arrow(ax, (7.85, 7.05), top_anchor(b_n_q), label="yes", label_offset=(0.05, 0.1), label_side='left')

    b_cv_small = box(ax, 4.6, 5.0, 1.95, 1.05, "cvAdaptiveKDE\n(default scalings)\n+ accept ~50% rate\nrecovery\n- low N kills SNR", facecolor=COL_METHOD, fontsize=9)
    arrow(ax, (7.0, 5.85), (5.6, 4.95), label="N <~ 5k", label_offset=(-0.4, 0.05), label_side='right')

    b_cv_large = box(ax, 6.85, 5.0, 1.95, 1.05, "cvAdaptiveKDE\n(default scalings,\nroi6=None)\n+ make_production_\ncv_kde factory", facecolor=COL_METHOD, fontsize=9, fontweight="bold")
    arrow(ax, (7.85, 5.85), top_anchor(b_cv_large), label="N >~ 10k", label_offset=(0.05, 0.05), label_side='left')

    # --- RIGHT: sample-drawing
    b_sample_q = box(ax, 9.4, 7.7, 3.7, 0.65, "Generative? or paper-level\nnarrow-feature ablation?", facecolor=COL_DATA, fontsize=10)
    arrow(ax, bottom_anchor(b_task), top_anchor(b_sample_q), label="generate samples", label_offset=(1.8, 0.20), label_side='left')

    b_nf = box(ax, 9.4, 6.7, 1.75, 0.85, "NormalizingFlowKDE\n(MAF, ensemble=10)\n- Buckley et al.", facecolor=COL_METHOD, fontsize=9)
    arrow(ax, (10.3, 7.05), top_anchor(b_nf), label="yes", label_offset=(-0.3, 0.1), label_side='right')

    b_cv_draw = box(ax, 11.35, 6.7, 1.75, 0.85, "cvAdaptiveKDE\nor cvGaussianKDE\n.draw(N)", facecolor=COL_METHOD, fontsize=9)
    arrow(ax, (12.25, 7.05), top_anchor(b_cv_draw), label="just need\ndraws", label_offset=(0.05, 0.0), label_side='left')

    # --- Footnotes
    foot_y = 3.6
    foot_h = 0.30
    foot_x0 = 0.4
    foot_w = 12.7
    ax.text(foot_x0, foot_y, "Cross-cutting notes:", fontsize=11, fontweight="bold", color="#222222")
    notes = [
        "-  Compute cost:  scipy ~ 1 s build, ms eval;   cvGaussianKDE ~ 30 s CV;   "
        "cvAdaptiveKDE ~ 2-10 min CV;   NormalizingFlow ~ 5-25 min train (10x ensemble).",
        "-  For sky maps & rate-weighted angular outputs use the production factory's "
        "cv.kde_sky / cv.kde_vinf picks - they target rate-weighted Kish ESS, not global ISE.",
        "-  Multi-scale data (e.g. cold + hot stellar populations): cvGaussianKDE with "
        "scalings='narrow' is often the simplest and most accurate choice; cvAdaptive's local "
        "Sigma_i doesn't always earn its keep.",
        "-  Spiky-ball / many-narrow-features regime (N_features >~ 10) at N >~ 10k: NF ensemble "
        "and cvAdaptive both work; cvAdaptive cheaper at small N, NF wins when capacity scaling matters.",
        "-  N <~ 1k:   no method recovers narrow-feature rate well; understand the bias floor "
        "(median KDE-rate / true-rate ~ 0.4 in synthetic benchmarks) before trusting numbers.",
        "-  EnBiD (Sharma & Steinmetz 2006) is the standard reference in galactic-dynamics; "
        "useful as a sanity baseline but tends to inflate density on dense query grids.",
    ]
    for k, n in enumerate(notes):
        ax.text(foot_x0, foot_y - 0.35 - 0.35 * k, n, fontsize=9.0, color="#222222", va='top')

    # Legend
    legend_y = 0.55
    for i, (col, name) in enumerate([(COL_TASK, "Task"), (COL_DATA, "Data property"), (COL_N, "Sample size"), (COL_METHOD, "Method recommendation")]):
        x0 = 0.4 + 3.2 * i
        ax.add_patch(FancyBboxPatch((x0, legend_y - 0.18), 0.4, 0.30, boxstyle="round,pad=0.02,rounding_size=0.04", linewidth=1.0, edgecolor=COL_EDGE, facecolor=col))
        ax.text(x0 + 0.55, legend_y - 0.04, name, fontsize=9.5, va="center", ha="left")

    plt.tight_layout()
    fig.savefig("method_decision_guide.pdf")
    fig.savefig("method_decision_guide.png", dpi=130)
    print("wrote method_decision_guide.pdf and method_decision_guide.png")


if __name__ == "__main__":
    main()
