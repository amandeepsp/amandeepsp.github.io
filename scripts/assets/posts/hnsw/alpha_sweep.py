"""Alpha sweep visualization for HNSW blog post.

Shows theoretical expected number of steps vs alpha in a 2D NSW graph.
Demonstrates the optimal alpha = d (dimension) for navigable small worlds.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from _common.paths import output_path
from _common.matplotlib import save_figure
from _common.colors import Colors

D = 2  # dimension
N = 10000  # number of points for scaling
ALPHA_VALUES = np.linspace(0.1, 5.0, 100)


def expected_steps(alpha, n, d):
    """
    Theoretical expected steps for greedy search in NSW.
    - For alpha < d: n^((d - alpha) / 3)
    - For alpha > d: n^((alpha - d) / (alpha - 1))
    Both converge to n^0 = 1 at alpha = d (optimal)
    """
    if alpha <= d:
        return n ** ((d - alpha) / 3)
    else:
        return n ** ((alpha - d) / (alpha - 1))


def main():
    steps = []
    for alpha in ALPHA_VALUES:
        steps.append(expected_steps(alpha, N, D))

    fig, ax = plt.subplots(figsize=(8, 5))

    # Split into two segments for legend
    left_mask = ALPHA_VALUES <= D
    right_mask = ALPHA_VALUES >= D

    log_steps = np.log(steps)

    ax.plot(
        ALPHA_VALUES[left_mask], log_steps[left_mask], "-",
        color=Colors.PRIMARY, linewidth=2,
        label=r"$\alpha < d$: $n^{\frac{d - \alpha}{3}}$"
    )
    ax.plot(
        ALPHA_VALUES[right_mask], log_steps[right_mask], "-",
        color=Colors.SECONDARY, linewidth=2,
        label=r"$\alpha > d$: $n^{\frac{\alpha - d}{\alpha - 1}}$"
    )

    # Mark optimal alpha = 2 (dimension)
    optimal_idx = np.argmin(np.abs(ALPHA_VALUES - 2.0))
    ax.axvline(x=2.0, color=Colors.TERTIARY, linestyle="--", linewidth=1.5, alpha=0.7)
    ax.annotate(
        r"$\alpha = d = 2$",
        xy=(2.0, log_steps[optimal_idx]),
        xytext=(2.8, log_steps[optimal_idx] + 1),
        fontsize=11,
        color=Colors.TERTIARY,
        arrowprops=dict(arrowstyle="->", color=Colors.TERTIARY, lw=1.5),
    )

    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)

    ax.set_xlabel(r"$\alpha$", fontsize=12)
    ax.set_ylabel(r"$\log(\mathbb{E}[\text{steps}])$", fontsize=12)
    ax.set_xlim(0, 5.2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_path(__file__, "svg"), format="svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
