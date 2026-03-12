"""Volume of a unit ball in d dimensions, d = 1..50."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

from _common.paths import output_path
from _common.matplotlib import save_figure
from _common.colors import Colors

D_MAX = 50


def ball_volume(d):
    return np.pi ** (d / 2) / gamma(d / 2 + 1)


def main():
    dims = np.arange(1, D_MAX + 1)
    vols = np.array([ball_volume(d) for d in dims])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(dims, vols, color=Colors.PRIMARY, alpha=0.8, width=0.8)

    peak_d = dims[np.argmax(vols)]
    peak_v = vols[np.argmax(vols)]
    ax.annotate(
        f"d={peak_d}, V≈{peak_v:.2f}",
        xy=(peak_d, peak_v),
        xytext=(peak_d + 6, peak_v * 0.85),
        fontsize=10,
        color=Colors.SECONDARY,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=Colors.SECONDARY, lw=1.5),
    )

    ax.set_xlabel("Dimension (d)", fontsize=11)
    ax.set_ylabel("Volume of unit ball", fontsize=11)
    ax.set_xlim(0, D_MAX + 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    save_figure(fig, output_path(__file__, "svg"), format="svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
