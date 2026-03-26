"""Complement visualization: A = (2,2):(1,6), complement under M=24.

Shows a flat strip of 24 indices colored by complement group.
A's offsets {0,1,6,7} share one color; each shifted copy gets its own,
making the repeating pattern visible.

The color mapping is built purely from layout algebra:
  compose(group_projection, left_inverse(combined_layout))
maps each offset to its complement group index.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tensor_layouts import Layout, complement, compose, left_inverse
from tensor_layouts.viz import draw_layout

from _common.paths import output_path


def main():
    A = Layout((2, 2), (1, 6))
    comp = complement(A, 24)
    combined = Layout(A, comp)  # ((2,2),(3,2)):((1,6),(2,12))

    # Build offset → group color mapping via layout algebra:
    # left_inverse maps offset → flat coordinate index,
    # group_proj extracts the complement coordinate (ignoring A's modes).
    li = left_inverse(combined)
    group_proj = Layout((2, 2, 3, 2), (0, 0, 1, 3))
    color = compose(group_proj, li)

    # Wrap as rank-1 (nested shape) so draw_layout accepts it
    color_r1 = Layout(
        ((color.shape),),
        ((color.stride),),
    )

    draw_layout(
        Layout(24, 1),
        str(output_path(__file__, "svg")),
        colorize=True,
        color_layout=color_r1,
        title=r"Complement (2,2):(1,6) under 24 is (3,2):(2,12)",
    )


if __name__ == "__main__":
    main()
