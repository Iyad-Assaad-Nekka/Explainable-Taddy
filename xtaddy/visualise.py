"""
X-TADDY: Visualisation Utilities
=================================
Functions for plotting spatial and temporal Shapley attribution bar charts,
reproducing Figure 1 of the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional, List
from .explainer import EdgeExplanation


# Colours consistent with the paper figure
CORAL  = '#D85A30'
BLUE   = '#185FA5'
LIGHT  = '#f5f5f3'


def plot_explanation(
    expl:        EdgeExplanation,
    k:           int             = 5,
    W:           int             = 2,
    edge_label:  str             = '(u, v)',
    snap_label:  str             = 't*',
    save_path:   Optional[str]   = None,
    figsize:     tuple           = (10, 4),
):
    """
    Plot spatial and temporal Shapley attribution bar charts for one edge.

    This reproduces Fig. 1 of the X-TADDY paper.

    Parameters
    ----------
    expl : EdgeExplanation
        Output of XTADDYExplainer.explain_edge().
    k : int
        Number of neighbours in TADDY (default 5).
    W : int
        Window size (default 2).
    edge_label : str
        Label for the target edge (displayed in figure title).
    snap_label : str
        Label for the snapshot (displayed in figure title).
    save_path : str or None
        If provided, save the figure to this path (e.g. 'fig.pdf').
    figsize : tuple
        Figure size in inches.
    """
    fig, (ax_sp, ax_tp) = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor('white')

    # ── Spatial attribution ──────────────────────────────────────────────
    phi_sp  = expl.phi_spatial
    n_nodes = k + 2
    labels  = ['u', 'v'] + [f'n{i}' for i in range(1, k + 1)]
    colors  = [CORAL if v >= 0 else BLUE for v in phi_sp]

    bars = ax_sp.barh(
        labels[:n_nodes], phi_sp[:n_nodes],
        color=colors[:n_nodes], alpha=0.85, height=0.6,
    )
    ax_sp.axvline(0, color='#888780', linewidth=0.6, linestyle='--')
    ax_sp.set_xlabel('Shapley value φ', fontsize=11, color='#3d3d3a')
    ax_sp.set_title('Spatial attribution — neighbor tokens',
                    fontsize=12, fontweight='bold', color='#1a1a18', pad=8)
    ax_sp.tick_params(axis='both', labelsize=10, colors='#3d3d3a')
    ax_sp.spines['top'].set_visible(False)
    ax_sp.spines['right'].set_visible(False)
    ax_sp.spines['left'].set_color('#d3d1c7')
    ax_sp.spines['bottom'].set_color('#d3d1c7')
    ax_sp.set_facecolor(LIGHT)

    # Annotate bar values
    for bar, val in zip(bars, phi_sp[:n_nodes]):
        xpos = val + 0.002 if val >= 0 else val - 0.002
        ha   = 'left' if val >= 0 else 'right'
        ax_sp.text(xpos, bar.get_y() + bar.get_height() / 2,
                   f'{val:+.3f}', va='center', ha=ha,
                   fontsize=9, color='#2c2c2a')

    # ── Temporal attribution ─────────────────────────────────────────────
    phi_tp = expl.phi_temporal
    step_labels = ['t (current)'] + [f't−{l} (prev.)' for l in range(1, W)]
    tp_colors   = [CORAL if v >= 0 else BLUE for v in phi_tp]

    bars_tp = ax_tp.bar(
        step_labels, phi_tp,
        color=tp_colors, alpha=0.85, width=0.5,
    )
    ax_tp.axhline(0, color='#888780', linewidth=0.6, linestyle='--')
    ax_tp.set_ylabel('Shapley value φ', fontsize=11, color='#3d3d3a')
    ax_tp.set_title('Temporal attribution — lookback steps',
                    fontsize=12, fontweight='bold', color='#1a1a18', pad=8)
    ax_tp.tick_params(axis='both', labelsize=10, colors='#3d3d3a')
    ax_tp.spines['top'].set_visible(False)
    ax_tp.spines['right'].set_visible(False)
    ax_tp.spines['left'].set_color('#d3d1c7')
    ax_tp.spines['bottom'].set_color('#d3d1c7')
    ax_tp.set_facecolor(LIGHT)

    for bar, val in zip(bars_tp, phi_tp):
        ypos = val + 0.005 if val >= 0 else val - 0.005
        va   = 'bottom' if val >= 0 else 'top'
        ax_tp.text(bar.get_x() + bar.get_width() / 2, ypos,
                   f'{val:+.3f}', va=va, ha='center',
                   fontsize=9, color='#2c2c2a')

    # ── Legend ───────────────────────────────────────────────────────────
    pos_patch = mpatches.Patch(color=CORAL, alpha=0.85,
                               label='positive (raises anomaly score)')
    neg_patch = mpatches.Patch(color=BLUE,  alpha=0.85,
                               label='suppressive (lowers anomaly score)')
    fig.legend(
        handles=[pos_patch, neg_patch],
        loc='lower center', ncol=2, fontsize=10,
        frameon=False, labelcolor='#2c2c2a',
        bbox_to_anchor=(0.5, -0.04),
    )

    fig.suptitle(
        f'X-TADDY explanation for edge {edge_label} at snapshot {snap_label}  '
        f'(score = {expl.true_score:.3f}, fidelity = {expl.fidelity:.4f})',
        fontsize=11, color='#1a1a18', y=1.02,
    )

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white')
        print(f'Figure saved to {save_path}')

    return fig, (ax_sp, ax_tp)


def plot_mean_attribution(
    explanations: List[EdgeExplanation],
    k:            int           = 5,
    W:            int           = 2,
    title:        str           = 'Mean X-TADDY Attribution',
    save_path:    Optional[str] = None,
    figsize:      tuple         = (10, 4),
):
    """
    Plot mean spatial and temporal Shapley attributions across multiple edges.

    Parameters
    ----------
    explanations : list of EdgeExplanation
    k, W : TADDY hyperparameters
    title : figure suptitle
    save_path : optional save path
    figsize : figure size
    """
    sp_mat = np.stack([e.phi_spatial  for e in explanations], axis=0)
    tp_mat = np.stack([e.phi_temporal for e in explanations], axis=0)
    phi_sp = sp_mat.mean(axis=0)
    phi_tp = tp_mat.mean(axis=0)
    sp_err = sp_mat.std(axis=0)
    tp_err = tp_mat.std(axis=0)

    # Reuse the single-edge plot logic with mean values
    from copy import deepcopy
    mock = deepcopy(explanations[0])
    mock.phi_spatial  = phi_sp
    mock.phi_temporal = phi_tp
    mock.true_score   = float(np.mean([e.true_score for e in explanations]))
    mock.fidelity     = float(np.mean([e.fidelity   for e in explanations]))

    fig, axes = plot_explanation(
        mock, k=k, W=W,
        edge_label=f'(mean over {len(explanations)} edges)',
        save_path=save_path, figsize=figsize,
    )
    fig.suptitle(title + f'  (n={len(explanations)} edges)',
                 fontsize=11, color='#1a1a18', y=1.02)
    return fig, axes
