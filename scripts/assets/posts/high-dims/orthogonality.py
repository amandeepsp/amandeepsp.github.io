"""Histogram of pairwise dot products for random unit vectors in varying dimensions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from _common.paths import output_path
from _common.matplotlib import save_figure
from _common.colors import Colors

N_VECTORS = 500
SEED = 42
DIMS = [3, 50, 500]
DIM_COLORS = [Colors.SECONDARY, Colors.PRIMARY, Colors.TERTIARY]


def random_unit_vectors(n, d, rng):
    v = rng.standard_normal((n, d))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms


def pairwise_dots(vectors):
    G = vectors @ vectors.T
    idx = np.triu_indices(len(vectors), k=1)
    return G[idx]


def main():
    rng = np.random.default_rng(SEED)

    fig, ax = plt.subplots(figsize=(8, 4))

    for d, color in zip(DIMS, DIM_COLORS):
        vecs = random_unit_vectors(N_VECTORS, d, rng)
        dots = pairwise_dots(vecs)
        ax.hist(
            dots,
            bins=60,
            density=True,
            alpha=0.55,
            color=color,
            label=f"d = {d}",
            edgecolor=color,
            linewidth=0.5,
        )

    ax.axvline(0, color=Colors.GRAY_400, linestyle="--", linewidth=1)
    ax.set_xlabel("Dot product (cosine similarity)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-1, 1)

    plt.tight_layout()
    save_figure(fig, output_path(__file__, "svg"), format="svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
