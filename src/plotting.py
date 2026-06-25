"""
Matplotlib heatmap helpers for visualising per-head scores and classifications.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure


def plot_score_heatmap(
    scores: np.ndarray,
    title: str = "Head Scores",
    xlabel: str = "Head",
    ylabel: str = "Layer",
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    annot: bool = True,
    figsize: tuple[int, int] = (12, 6),
) -> Figure:
    """Plot a 2-D array of per-head scalar scores as a colour-mapped heatmap.

    Args:
        scores:  Array of shape ``[n_layers, n_heads]``.
        title:   Figure title.
        xlabel:  X-axis label (default "Head").
        ylabel:  Y-axis label (default "Layer").
        cmap:    Matplotlib colormap name.
        vmin:    Colour scale minimum (symmetric around 0 if both None).
        vmax:    Colour scale maximum.
        annot:   If True, write the numeric value inside each cell.
        figsize: ``(width, height)`` in inches.

    Returns:
        The ``matplotlib.figure.Figure`` object (call ``plt.show()`` or
        ``fig.savefig(...)`` on the return value).
    """
    n_layers, n_heads = scores.shape

    abs_max = np.nanmax(np.abs(scores))
    if vmin is None and vmax is None:
        vmin, vmax = -abs_max, abs_max

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(scores, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([str(h) for h in range(n_heads)])
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([str(l) for l in range(n_layers)])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    plt.colorbar(im, ax=ax)

    if annot:
        for row in range(n_layers):
            for col in range(n_heads):
                val = scores[row, col]
                if not np.isnan(val):
                    text_color = "white" if abs(val) > (abs_max * 0.6) else "black"
                    ax.text(col, row, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color=text_color)

    fig.tight_layout()
    return fig


def plot_head_heatmap(
    labels: np.ndarray,
    category_colors: dict[str, str] | None = None,
    title: str = "Head Classification",
    figsize: tuple[int, int] = (12, 6),
) -> Figure:
    """Plot a 2-D grid of head classifications using a discrete colour per category.

    Args:
        labels:            Object array of shape ``[n_layers, n_heads]`` where each
                           cell is a list of label strings (from ``classify_all_heads``).
                           If a head has multiple labels the *first* one is used for
                           colouring; all are written as text.
        category_colors:   Mapping from label string to a matplotlib colour string.
                           Defaults to a sensible preset if ``None``.
        title:             Figure title.
        figsize:           ``(width, height)`` in inches.

    Returns:
        The ``matplotlib.figure.Figure`` object.
    """
    default_colors: dict[str, str] = {
        "induction": "#4c72b0",
        "prev_token": "#55a868",
        "copy": "#c44e52",
        "copy_suppression": "#8172b2",
        "other": "#cccccc",
    }
    if category_colors is not None:
        default_colors.update(category_colors)

    n_layers, n_heads = labels.shape

    # Build an RGB image
    img = np.ones((n_layers, n_heads, 3))
    for row in range(n_layers):
        for col in range(n_heads):
            cell = labels[row, col]
            primary = cell[0] if cell else "other"
            color = default_colors.get(primary, default_colors["other"])
            img[row, col] = mcolors.to_rgb(color)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, aspect="auto")

    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([str(h) for h in range(n_heads)])
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([str(l) for l in range(n_layers)])
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(title)

    # Short labels for the cell text. copy and copy_suppression must differ, since
    # a naive first-3-chars would render both as "cop".
    abbrev = {
        "induction": "ind",
        "prev_token": "pre",
        "copy": "cop",
        "copy_suppression": "sup",
    }
    for row in range(n_layers):
        for col in range(n_heads):
            cell = labels[row, col]
            text = "\n".join(abbrev.get(l, l[:3]) for l in cell) if cell else ""
            ax.text(col, row, text, ha="center", va="center", fontsize=6, color="white")

    # Legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c, label=k)
        for k, c in default_colors.items()
    ]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.15, 1),
              fontsize=8, title="Category")

    fig.tight_layout()
    return fig


def plot_attention_pattern(
    pattern: np.ndarray,
    tokens: list[str] | None = None,
    title: str = "Attention Pattern",
    figsize: tuple[int, int] = (8, 7),
) -> Figure:
    """Plot a single attention head's ``[seq, seq]`` pattern as a heatmap.

    Args:
        pattern:  Array of shape ``[seq_len, seq_len]``, rows = destination,
                  cols = source.
        tokens:   Optional list of token strings for axis tick labels.
        title:    Figure title.
        figsize:  ``(width, height)`` in inches.

    Returns:
        The ``matplotlib.figure.Figure`` object.
    """
    seq_len = pattern.shape[0]
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(pattern, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax)

    tick_labels = tokens if tokens is not None else [str(i) for i in range(seq_len)]
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(seq_len))
    ax.set_yticklabels(tick_labels, fontsize=8)
    ax.set_xlabel("Source (key)")
    ax.set_ylabel("Destination (query)")
    ax.set_title(title)

    fig.tight_layout()
    return fig
