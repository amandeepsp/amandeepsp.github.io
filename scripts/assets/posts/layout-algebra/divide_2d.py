"""2D layout divide: B = (4,6):(1,4) divided by tiler (2,3).

Shows a 4x6 column-major matrix split into a 2x2 grid of 2x3 tiles.

Result: ((2, 2), (3, 2)) : ((1, 2), (4, 12))
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tensor_layouts import Layout, logical_divide
from tensor_layouts.viz import draw_layout

from _common.paths import output_path


def main():
    B = Layout((4, 6), (1, 4))
    divided = logical_divide(B, (2, 3))  # ((2,2),(3,2)):((1,2),(4,12))

    # Color by tile: map inter-tile modes to distinct values
    color = Layout(((1, 2), (1, 2)), ((0, 1), (0, 2)))

    draw_layout(
        divided,
        str(output_path(__file__, "svg")),
        colorize=True,
        color_layout=color,
        label_hierarchy_levels=True,
        title=r"$((2, 2), (3, 2)) : ((1, 2), (4, 12))$",
    )


if __name__ == "__main__":
    main()
