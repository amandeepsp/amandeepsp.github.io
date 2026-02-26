"""1D layout divide: B = 12:1 divided by A = 4:1 -> (4, 3) : (1, 4)

Shows a flat 12-element array split into 3 tiles of 4 elements each,
with annotations for intra-tile and inter-tile modes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from _common.paths import output_path
from _common.matplotlib import save_figure
from _common.colors import Colors

TILE_COLORS = [Colors.PRIMARY_LIGHT, Colors.SECONDARY_LIGHT, Colors.TERTIARY_LIGHT]
TILE_BORDERS = [Colors.PRIMARY, Colors.SECONDARY, Colors.TERTIARY]

N = 12
TILE_SIZE = 4
N_TILES = N // TILE_SIZE
CELL_W = 1.0
CELL_H = 1.0
TILE_GAP = 0.4


def cell_x(i):
    """X position of cell i, accounting for tile gaps."""
    tile = i // TILE_SIZE
    local = i % TILE_SIZE
    return tile * (TILE_SIZE * CELL_W + TILE_GAP) + local * CELL_W


def main():
    fig, ax = plt.subplots(figsize=(10, 2.8))

    # Draw cells
    for i in range(N):
        t = i // TILE_SIZE
        x = cell_x(i)
        rect = patches.FancyBboxPatch(
            (x + 0.04, 0.04),
            CELL_W - 0.08,
            CELL_H - 0.08,
            boxstyle="round,pad=0.03",
            facecolor=TILE_COLORS[t],
            edgecolor=TILE_BORDERS[t],
            linewidth=1.5,
            alpha=0.75,
        )
        ax.add_patch(rect)
        ax.text(
            x + CELL_W / 2,
            CELL_H / 2,
            str(i),
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color=Colors.GRAY_900,
        )

    # Intra-tile brace above first tile
    brace_y = CELL_H + 0.12
    x0 = cell_x(0) + 0.15
    x1 = cell_x(TILE_SIZE - 1) + CELL_W - 0.15
    ax.annotate(
        "",
        xy=(x0, brace_y),
        xytext=(x1, brace_y),
        arrowprops=dict(arrowstyle="<->", color=TILE_BORDERS[0], lw=2),
    )
    ax.text(
        (x0 + x1) / 2,
        brace_y + 0.12,
        r"intra: $4 : 1$",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=TILE_BORDERS[0],
    )

    # Inter-tile arrows: from first element of each tile to the next
    arrow_y = -0.2
    for t in range(N_TILES - 1):
        ax.annotate(
            "",
            xy=(cell_x((t + 1) * TILE_SIZE) + CELL_W / 2, arrow_y),
            xytext=(cell_x(t * TILE_SIZE) + CELL_W / 2, arrow_y),
            arrowprops=dict(
                arrowstyle="->",
                color=Colors.GRAY_500,
                lw=1.5,
                connectionstyle="arc3,rad=0",
            ),
        )
        mid_x = (
            cell_x(t * TILE_SIZE) + cell_x((t + 1) * TILE_SIZE) + CELL_W
        ) / 2
        ax.text(
            mid_x,
            arrow_y - 0.05,
            r"$+4$",
            ha="center",
            va="top",
            fontsize=10,
            color=Colors.GRAY_500,
        )

    # Inter-tile label
    total_w = cell_x(N - 1) + CELL_W
    ax.text(
        total_w / 2,
        arrow_y - 0.35,
        r"inter: $3 : 4$",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        color=Colors.GRAY_700,
    )

    # Result formula
    ax.text(
        total_w / 2,
        -0.95,
        r"$B \oslash A = (4, 3) : (1, 4)$",
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

    ax.set_xlim(-0.4, total_w + 0.4)
    ax.set_ylim(-1.4, CELL_H + 0.7)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    save_figure(fig, output_path(__file__, "svg"), format="svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
