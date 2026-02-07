"""Voronoi diagram visualization for HNSW blog post."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi

from _common.paths import output_path
from _common.matplotlib import save_figure
from _common.colors import Colors

SEED = 2026
N_POINTS = 10


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points).max() * 2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def main():
    rng = np.random.default_rng(SEED)
    points = rng.uniform(0.1, 0.9, size=(N_POINTS, 2))

    vor = Voronoi(points)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Find which region the query point falls into
    query_point = np.array([0.65, 0.75])
    distances = np.linalg.norm(points - query_point, axis=1)
    nearest_idx = np.argmin(distances)

    regions, vertices = voronoi_finite_polygons_2d(vor)
    for i, region in enumerate(regions):
        polygon = vertices[region]
        # Highlight the region containing the query point
        if i == nearest_idx:
            ax.fill(
                *zip(*polygon),
                alpha=0.4,
                color=Colors.PRIMARY_LIGHT,
                edgecolor=Colors.PRIMARY,
                linewidth=2,
            )
        else:
            ax.fill(*zip(*polygon), alpha=0.2, color=Colors.GRAY_400)

    # Draw points
    ax.plot(points[:, 0], points[:, 1], "o", color=Colors.GRAY_700, markersize=8, zorder=3)

    # Draw and label the query point
    ax.plot(
        query_point[0],
        query_point[1],
        "o",
        color=Colors.ACCENT,
        markersize=8,
        zorder=4,
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

    # Label the nearest point
    ax.annotate(
        "$p$",
        points[nearest_idx],
        xytext=(-12, -12),
        textcoords="offset points",
        fontsize=14,
        fontweight="bold",
        color=Colors.SUCCESS,
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
