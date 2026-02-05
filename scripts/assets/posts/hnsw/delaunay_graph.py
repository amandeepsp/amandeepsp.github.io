"""Delaunay triangulation graph visualization for HNSW blog post."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, Voronoi

from _common.paths import output_path
from _common.matplotlib import save_figure
from voronoi_diagram import voronoi_finite_polygons_2d

SEED = 2026
N_POINTS = 8


def main():
    rng = np.random.default_rng(SEED)
    points = rng.uniform(0.1, 0.9, size=(N_POINTS, 2))

    tri = Delaunay(points)
    vor = Voronoi(points)

    fig, ax = plt.subplots(figsize=(8, 6))

    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            edge = tuple(sorted([simplex[i], simplex[(i + 1) % 3]]))
            edges.add(edge)

    for i, j in edges:
        ax.plot(
            [points[i, 0], points[j, 0]],
            [points[i, 1], points[j, 1]],
            "-",
            color="#6b7280",
            linewidth=2,
            zorder=1,
        )

    regions, vertices = voronoi_finite_polygons_2d(vor)
    for region in regions:
        polygon = vertices[region]
        ax.fill(*zip(*polygon), alpha=0.2)

    ax.plot(points[:, 0], points[:, 1], "ko")
    ax.set_xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
    ax.set_ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)
    ax.set_aspect("equal")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    save_figure(fig, output_path(__file__, "svg"), format="svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
