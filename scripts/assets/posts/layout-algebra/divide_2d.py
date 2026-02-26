"""2D layout divide: B = (4,6):(1,4) divided by tiler (2,3).

Shows a 4x6 column-major matrix split into a 2x2 grid of 2x3 tiles,
with flat indices in each cell and tiles colored distinctly.

Result: ((2, 2), (3, 2)) : ((1, 2), (4, 12))
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from _common.paths import output_path
from _common.matplotlib import save_figure
from _common.colors import Colors

# Matrix dimensions and tiling
ROWS, COLS = 4, 6
TILE_ROWS, TILE_COLS = 2, 3
TILES_R = ROWS // TILE_ROWS  # 2
TILES_C = COLS // TILE_COLS  # 2

CELL = 1.0
TILE_GAP = 0.25

# Colors for the 4 tiles (row-major tile order)
TILE_FILL = [
    Colors.PRIMARY_LIGHT,
    Colors.SECONDARY_LIGHT,
    Colors.TERTIARY_LIGHT,
    Colors.SUCCESS,
]
TILE_EDGE = [
    Colors.PRIMARY,
    Colors.SECONDARY,
    Colors.TERTIARY,
    "#3d6b50",
]


def tile_of(row, col):
    """Which tile a cell belongs to."""
    tr = row // TILE_ROWS
    tc = col // TILE_COLS
    return tr * TILES_C + tc


def cell_xy(row, col):
    """Top-left corner of cell (row, col) with tile gaps."""
    tr = row // TILE_ROWS
    tc = col // TILE_COLS
    x = col * CELL + tc * TILE_GAP
    y = (ROWS - 1 - row) * CELL + (TILES_R - 1 - tr) * TILE_GAP  # flip y
    return x, y


def main():
    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw cells
    for r in range(ROWS):
        for c in range(COLS):
            t = tile_of(r, c)
            x, y = cell_xy(r, c)
            flat_idx = r + c * ROWS  # column-major

            rect = patches.FancyBboxPatch(
                (x + 0.04, y + 0.04),
                CELL - 0.08,
                CELL - 0.08,
                boxstyle="round,pad=0.03",
                facecolor=TILE_FILL[t],
                edgecolor=TILE_EDGE[t],
                linewidth=1.5,
                alpha=0.75,
            )
            ax.add_patch(rect)
            ax.text(
                x + CELL / 2,
                y + CELL / 2,
                str(flat_idx),
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color=Colors.GRAY_900,
            )

    # Row labels
    for r in range(ROWS):
        x, y = cell_xy(r, 0)
        ax.text(
            x - 0.25,
            y + CELL / 2,
            f"r{r}",
            ha="right",
            va="center",
            fontsize=10,
            color=Colors.GRAY_500,
        )

    # Column labels
    for c in range(COLS):
        x, y = cell_xy(0, c)
        ax.text(
            x + CELL / 2,
            y + CELL + 0.15,
            f"c{c}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=Colors.GRAY_500,
        )

    # Intra-tile annotations (right side)
    # Show intra-row within tile (0,0)
    x0, y0 = cell_xy(0, 0)
    x1, y1 = cell_xy(1, 0)
    ann_x = cell_xy(0, COLS - 1)[0] + CELL + 0.4
    ax.annotate(
        "",
        xy=(ann_x, y1 + CELL / 2),
        xytext=(ann_x, y0 + CELL / 2),
        arrowprops=dict(arrowstyle="<->", color=TILE_EDGE[0], lw=2),
    )
    ax.text(
        ann_x + 0.15,
        (y0 + y1 + CELL) / 2,
        r"$2:1$",
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=TILE_EDGE[0],
    )

    # Show inter-row (tile boundary)
    x2, y2 = cell_xy(2, 0)
    ann_x2 = ann_x + 1.2
    ax.annotate(
        "",
        xy=(ann_x2, y2 + CELL / 2),
        xytext=(ann_x2, y0 + CELL / 2),
        arrowprops=dict(arrowstyle="<->", color=TILE_EDGE[1], lw=2),
    )
    ax.text(
        ann_x2 + 0.15,
        (y0 + y2 + CELL) / 2,
        r"$2:2$",
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=TILE_EDGE[1],
    )

    # Intra-col annotation (below)
    _, y_bot = cell_xy(ROWS - 1, 0)
    ann_y = y_bot - 0.4
    x_c0, _ = cell_xy(0, 0)
    x_c2, _ = cell_xy(0, 2)
    ax.annotate(
        "",
        xy=(x_c2 + CELL / 2, ann_y),
        xytext=(x_c0 + CELL / 2, ann_y),
        arrowprops=dict(arrowstyle="<->", color=TILE_EDGE[0], lw=2),
    )
    ax.text(
        (x_c0 + x_c2 + CELL) / 2,
        ann_y - 0.15,
        r"$3:4$",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        color=TILE_EDGE[0],
    )

    # Inter-col annotation (below, wider)
    ann_y2 = ann_y - 0.7
    x_c3, _ = cell_xy(0, 3)
    ax.annotate(
        "",
        xy=(x_c3 + CELL / 2, ann_y2),
        xytext=(x_c0 + CELL / 2, ann_y2),
        arrowprops=dict(arrowstyle="<->", color=TILE_EDGE[2], lw=2),
    )
    ax.text(
        (x_c0 + x_c3 + CELL) / 2,
        ann_y2 - 0.15,
        r"$2:12$",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        color=TILE_EDGE[2],
    )

    # Legend
    _, y_top = cell_xy(0, 0)
    legend_y = y_bot - 1.6
    total_w = cell_xy(0, COLS - 1)[0] + CELL
    ax.text(
        total_w / 2,
        legend_y,
        r"$((2, 2), (3, 2)) : ((1, 2), (4, 12))$",
        ha="center",
        va="top",
        fontsize=13,
        color=Colors.GRAY_900,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor=Colors.GRAY_100,
            edgecolor=Colors.GRAY_300,
            linewidth=1,
        ),
    )

    ax.set_xlim(-0.6, total_w + 2.0)
    ax.set_ylim(legend_y - 0.5, y_top + CELL + 0.5)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    save_figure(fig, output_path(__file__, "svg"), format="svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
