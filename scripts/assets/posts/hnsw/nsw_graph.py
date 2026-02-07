"""Navigable Small World visualization for HNSW blog post.

Shows a graph with local Delaunay edges plus long-range links,
demonstrating how NSW enables faster greedy search.
"""

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
N_LONG_RANGE_LINKS = 6


def build_adjacency(tri, n_points):
    """Build adjacency list from Delaunay triangulation."""
    adjacency = {i: set() for i in range(n_points)}
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(3):
                if i != j:
                    adjacency[simplex[i]].add(simplex[j])
    return adjacency


def add_long_range_links(points, adjacency, rng, n_links, alpha, existing_edges):
    """
    Add long-range links with probability proportional to 1/d^alpha.
    Returns list of added long-range edges.
    """
    n = len(points)
    long_range_edges = []

    # Track all edges (Delaunay + new long-range)
    all_edges = set(existing_edges)

    # Compute distances between all pairs
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distances[i, j] = np.linalg.norm(points[i] - points[j])

    # For each node, potentially add a long-range link
    for _ in range(n_links):
        u = rng.integers(0, n)

        # Compute probabilities for all other nodes
        probs = np.zeros(n)
        for v in range(n):
            edge = tuple(sorted([u, v]))
            if v != u and edge not in all_edges:
                probs[v] = 1.0 / (distances[u, v] ** alpha + 1e-6)

        if probs.sum() > 0:
            probs /= probs.sum()
            v = rng.choice(n, p=probs)

            edge = tuple(sorted([u, v]))
            all_edges.add(edge)
            adjacency[u].add(v)
            adjacency[v].add(u)
            long_range_edges.append((u, v))

    return long_range_edges


def nsw_walk(points, adjacency, query, start_idx):
    """
    Perform greedy NSW walk from start_idx to find nearest neighbor of query.
    Returns the path taken as a list of indices.
    """
    path = [start_idx]
    current = start_idx
    current_dist = np.linalg.norm(points[current] - query)

    while True:
        best_neighbor = None
        best_dist = current_dist

        for neighbor in adjacency[current]:
            dist = np.linalg.norm(points[neighbor] - query)
            if dist < best_dist:
                best_dist = dist
                best_neighbor = neighbor

        if best_neighbor is None:
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

    # Collect Delaunay edges
    delaunay_edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            edge = tuple(sorted([simplex[i], simplex[(i + 1) % 3]]))
            delaunay_edges.add(edge)

    # Add long-range links with alpha = 2 (for 2D space)
    long_range_edges = add_long_range_links(
        points,
        adjacency,
        rng,
        N_LONG_RANGE_LINKS,
        alpha=2.0,
        existing_edges=delaunay_edges,
    )

    # Query point and starting point for the walk
    query_point = np.array([0.9, 0.8])
    start_idx = 0

    # Perform the walk
    path = nsw_walk(points, adjacency, query_point, start_idx)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Draw Voronoi regions (very faded)
    regions, vertices = voronoi_finite_polygons_2d(vor)
    for region in regions:
        polygon = vertices[region]
        ax.fill(*zip(*polygon), alpha=0.1, color=Colors.GRAY_400)

    # Draw Delaunay edges (local structure)
    for i, j in delaunay_edges:
        ax.plot(
            [points[i, 0], points[j, 0]],
            [points[i, 1], points[j, 1]],
            "-",
            color=Colors.GRAY_300,
            linewidth=1,
            zorder=1,
        )

    # Draw long-range links (curved for visual distinction)
    for i, j in long_range_edges:
        # Draw as dashed line with different color
        ax.plot(
            [points[i, 0], points[j, 0]],
            [points[i, 1], points[j, 1]],
            "--",
            color=Colors.TERTIARY,
            linewidth=1.5,
            alpha=0.7,
            zorder=1,
        )

    # Draw the walk path with arrows
    for k in range(len(path) - 1):
        i, j = path[k], path[k + 1]
        dx = points[j, 0] - points[i, 0]
        dy = points[j, 1] - points[i, 1]

        # Check if this edge is a long-range link
        edge = tuple(sorted([i, j]))
        is_long_range = edge in [(tuple(sorted(e))) for e in long_range_edges]
        arrow_color = Colors.TERTIARY_LIGHT if is_long_range else Colors.PRIMARY

        ax.annotate(
            "",
            xy=(points[j, 0] - dx * 0.1, points[j, 1] - dy * 0.1),
            xytext=(points[i, 0] + dx * 0.1, points[i, 1] + dy * 0.1),
            arrowprops=dict(
                arrowstyle="->",
                color=arrow_color,
                lw=2.5,
                mutation_scale=15,
            ),
            zorder=3,
        )

    # Draw all points
    ax.plot(
        points[:, 0], points[:, 1], "o", color=Colors.GRAY_700, markersize=8, zorder=2
    )

    # Highlight start point
    ax.plot(
        points[start_idx, 0],
        points[start_idx, 1],
        "o",
        color=Colors.WARNING,
        markersize=10,
        zorder=4,
    )
    ax.annotate(
        "start",
        points[start_idx],
        xytext=(-10, -15),
        textcoords="offset points",
        fontsize=11,
        color=Colors.WARNING,
        fontweight="bold",
    )

    # Highlight end point (nearest neighbor)
    ax.plot(
        points[path[-1], 0],
        points[path[-1], 1],
        "o",
        color=Colors.SUCCESS,
        markersize=10,
        zorder=4,
    )
    ax.annotate(
        "$p$",
        points[path[-1]],
        xytext=(-15, 8),
        textcoords="offset points",
        fontsize=14,
        fontweight="bold",
        color=Colors.SUCCESS,
    )

    # Draw query point
    ax.plot(
        query_point[0],
        query_point[1],
        "o",
        color=Colors.ACCENT,
        markersize=10,
        zorder=5,
    )
    ax.annotate(
        "$q$",
        query_point,
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=14,
        fontweight="bold",
        color=Colors.ACCENT,
    )

    # Add legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color=Colors.GRAY_300, linewidth=1, label="Local edges"),
        Line2D(
            [0],
            [0],
            color=Colors.TERTIARY,
            linewidth=1.5,
            linestyle="--",
            label="Long-range links",
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9)

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
