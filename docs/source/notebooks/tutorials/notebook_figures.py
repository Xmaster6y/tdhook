"""Shared plotting style and figure exports for research notebooks."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

BLUE = "#2458A6"
ORANGE = "#C57920"
TEAL = "#247B74"
GRAY = "#8391A4"
INK = "#233247"
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.titlesize": 17,
    "axes.titleweight": "semibold",
    "axes.titlepad": 14,
    "axes.labelsize": 13,
    "axes.edgecolor": "#C8D0DA",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
}


def finish_figure(figure, name, output_dir=Path("figures")):
    """Save vector and high-resolution raster figures, then display inline."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "png"):
        figure.savefig(output_dir / f"{name}.{extension}", bbox_inches="tight", facecolor="white")
    plt.show()
    return figure


def draw_board(axis, board, title, highlight=None):
    """Draw encoded labels 0=empty, 1=white (simulator -1), 2=black (+1)."""
    board = np.asarray(board)
    if board.shape != (8, 8) or not np.isin(board, [0, 1, 2]).all():
        raise ValueError("Expected an 8 x 8 board with labels 0, 1, 2")
    axis.set_facecolor("#DDECE4")
    axis.set(xlim=(-0.5, 7.5), ylim=(7.5, -0.5), aspect="equal", title=title)
    axis.set_xticks(range(8), list("ABCDEFGH"))
    axis.set_yticks(range(8), range(1, 9))
    axis.set_xticks(np.arange(-0.5, 8, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, 8, 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.3)
    axis.tick_params(which="both", length=0)
    for row, column in np.argwhere(board != 0):
        color = "white" if board[row, column] == 1 else INK
        axis.add_patch(Circle((column, row), 0.37, facecolor=color, edgecolor=INK, linewidth=1))
    if highlight is not None:
        row, column = highlight
        axis.add_patch(Rectangle((column - 0.47, row - 0.47), 0.94, 0.94, fill=False, edgecolor=ORANGE, lw=3))
    for spine in axis.spines.values():
        spine.set_visible(False)


def plot_circuit_overview(view):
    """Plot serialized workflow artifacts without accessing the model or tokenizer."""
    scores = np.asarray(view["head_scores"], dtype=float)
    activations = np.asarray(view["activations"], dtype=float)
    labels = np.asarray(view["clusters"], dtype=int)
    count = len(activations)
    if scores.shape != (8, count) or len(labels) != count or len(view["contexts"]) != count or count == 0:
        raise ValueError("Expected matching contexts, clusters, activations, and eight-head scores")
    order = sorted(range(count), key=lambda index: (view["empty"][index], labels[index], index))
    active_heads = np.flatnonzero(np.any(scores != 0, axis=1))
    if not len(active_heads):
        active_heads = np.arange(8)
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(
            1, 3, figsize=(13, 6.5), sharey=True, width_ratios=(3, 1.8, 0.65), constrained_layout=True
        )
        limit = max(float(np.abs(scores).max()), 1e-6)
        matrix = scores[np.ix_(active_heads, order)].T
        image = axes[0].imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
        for row, column in np.argwhere(matrix != 0):
            value = matrix[row, column]
            axes[0].text(
                column,
                row,
                f"{value:+.2f}",
                ha="center",
                va="center",
                color="white" if abs(value) > limit * 0.65 else INK,
                fontsize=11,
            )
        contexts = [view["contexts"][index].removeprefix("<bos>") for index in order]
        contexts = [text if len(text) <= 48 else "…" + text[-47:] for text in contexts]
        axes[0].set(
            title="Selected contributions",
            xticks=range(len(active_heads)),
            xticklabels=[f"H{head}" for head in active_heads],
            yticks=range(count),
            yticklabels=[f"{index + 1}. {text}" for index, text in zip(order, contexts)],
            xlabel="Other heads have no selected contributions",
        )
        axes[0].tick_params(axis="y", labelsize=10)
        figure.colorbar(image, ax=axes[0], label="Signed score", shrink=0.65, pad=0.03)
        axes[1].barh(range(count), activations[order], color=BLUE, height=0.7)
        axes[1].set(title="Feature activation", xlabel="Layer 0 / feature 24")
        axes[1].tick_params(axis="y", left=False)
        palette = [BLUE, TEAL, ORANGE, "#79599B"]
        for row, index in enumerate(order):
            label = int(labels[index])
            empty = view["empty"][index]
            color = GRAY if empty or label < 0 else palette[label % len(palette)]
            axes[2].barh(row, 1, color=color, height=0.7)
            axes[2].text(
                0.5,
                row,
                "empty" if empty else ("noise" if label < 0 else str(label)),
                ha="center",
                va="center",
                color="white",
                fontsize=10,
            )
        axes[2].set(title="Cluster", xticks=[], xlim=(0, 1))
        axes[2].tick_params(axis="y", left=False)
        axes[2].spines[:].set_visible(False)
    return figure


def workflow_figure(stages, artifacts, name):
    """Draw an explanatory producer-consumer schematic, not a runtime plan."""
    if len(artifacts) != len(stages) - 1:
        raise ValueError("Each connection needs an artifact label")
    with plt.rc_context(STYLE):
        figure, axis = plt.subplots(figsize=(13, 2))
        axis.set(xlim=(0, len(stages)), ylim=(0.02, 0.9))
        axis.axis("off")
        for index, stage in enumerate(stages):
            axis.add_patch(
                FancyBboxPatch(
                    (index + 0.04, 0.38), 0.76, 0.38, boxstyle="round,pad=0.02", facecolor="#EEF3F9", edgecolor=BLUE
                )
            )
            axis.text(index + 0.42, 0.57, stage, ha="center", va="center", fontsize=13)
            if index < len(artifacts):
                axis.annotate(
                    "",
                    (index + 1.02, 0.57),
                    (index + 0.82, 0.57),
                    arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 1.8},
                )
                axis.text(index + 0.94, 0.19, artifacts[index], ha="center", va="center", fontsize=12)
        figure.tight_layout()
        return finish_figure(figure, name)
