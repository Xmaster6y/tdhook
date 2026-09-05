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
