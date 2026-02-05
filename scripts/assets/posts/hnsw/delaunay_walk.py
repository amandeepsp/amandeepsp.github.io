"""Delaunay Walk visualization for HNSW blog post."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, Voronoi

# Import voronoi_diagram first - it sets up sys.path for _common imports
from voronoi_diagram import voronoi_finite_polygons_2d

from _common.paths import output_path
from _common.matplotlib import save_figure
from _common.colors import Colors

SEED = 2026
N_POINTS = 10


def build_adjacency(tri, n_points):
    """Build adjacency list from Delaunay triangulation."""
    adjacency = {i: set() for i in range(n_points)}
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(3):
                if i != j:
                    adjacency[simplex[i]].add(simplex[j])
    return adjacency


def delaunay_walk(points, adjacency, query, start_idx):
    """
    Perform greedy Delaunay walk from start_idx to find nearest neighbor of query.
    Returns the path taken as a list of indices.
    """
    path = [start_idx]
    current = start_idx
    current_dist = np.linalg.norm(points[current] - query)

    while True:
        # Find the neighbor closest to query
        best_neighbor = None
        best_dist = current_dist

        for neighbor in adjacency[current]:
            dist = np.linalg.norm(points[neighbor] - query)
            if dist < best_dist:
                best_dist = dist
                best_neighbor = neighbor

        if best_neighbor is None:
            # No closer neighbor found, we've arrived
            break

        current = best_neighbor
        current_dist = best_dist
        path.append(current)

    return path


def main():
    rng = np.random.default_rng(SEED)
    points = rng.uniform(0.1, 0.9, size=(N_POINTS, 2))

    tri = Delaunay(points)
    vor = Voronoi(points)
    adjacency = build_adjacency(tri, N_POINTS)

    # Query point and starting point for the walk
    query_point = np.array([0.65, 0.75])
    start_idx = 0  # Start from an arbitrary point

    # Perform the walk
    path = delaunay_walk(points, adjacency, query_point, start_idx)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Draw Voronoi regions (faded)
    regions, vertices = voronoi_finite_polygons_2d(vor)
    for region in regions:
        polygon = vertices[region]
        ax.fill(*zip(*polygon), alpha=Colors.REGION_FILL_ALPHA, color=Colors.REGION_FILL)

    # Draw all Delaunay edges (faded)
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
            color=Colors.EDGE,
            linewidth=1,
            zorder=1,
        )

    # Draw the walk path with arrows
    for k in range(len(path) - 1):
        i, j = path[k], path[k + 1]
        dx = points[j, 0] - points[i, 0]
        dy = points[j, 1] - points[i, 1]
        ax.annotate(
            "",
            xy=(points[j, 0] - dx * 0.1, points[j, 1] - dy * 0.1),
            xytext=(points[i, 0] + dx * 0.1, points[i, 1] + dy * 0.1),
            arrowprops=dict(
                arrowstyle="->",
                color=Colors.PATH,
                lw=2.5,
                mutation_scale=15,
            ),
            zorder=3,
        )

    # Draw all points
    ax.plot(points[:, 0], points[:, 1], "o", color=Colors.NODE, markersize=8, zorder=2)

    # Highlight start point
    ax.plot(
        points[start_idx, 0],
        points[start_idx, 1],
        "o",
        color=Colors.START_POINT,
        markersize=12,
        zorder=4,
    )
    ax.annotate(
        "start",
        points[start_idx],
        xytext=(-10, -15),
        textcoords="offset points",
        fontsize=11,
        color=Colors.START_POINT,
        fontweight="bold",
    )

    # Highlight end point (nearest neighbor)
    ax.plot(
        points[path[-1], 0],
        points[path[-1], 1],
        "o",
        color=Colors.END_POINT,
        markersize=12,
        zorder=4,
    )
    ax.annotate(
        "$p$",
        points[path[-1]],
        xytext=(-15, 8),
        textcoords="offset points",
        fontsize=14,
        fontweight="bold",
        color=Colors.END_POINT,
    )

    # Draw query point
    ax.plot(query_point[0], query_point[1], "*", color=Colors.QUERY_POINT, markersize=15, zorder=5)
    ax.annotate(
        "$q$",
        query_point,
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=14,
        fontweight="bold",
        color=Colors.QUERY_POINT,
    )

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
