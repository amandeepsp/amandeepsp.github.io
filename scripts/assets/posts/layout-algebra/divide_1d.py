"""1D layout divide: B = 12:1 divided by A = 4:1 -> (4, 3) : (1, 4)

Shows the divided layout as a grid where columns are tiles and rows are
intra-tile positions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tensor_layouts import Layout, logical_divide
from tensor_layouts.viz import draw_layout

from _common.paths import output_path


def main():
    B = Layout(12, 1)
    divided = logical_divide(B, 4)  # (4, 3) : (1, 4)

    draw_layout(
        divided,
        str(output_path(__file__, "svg")),
        colorize=True,
        color_by="column",
        label_hierarchy_levels=True,
        title=r"$B \oslash A = (4, 3) : (1, 4)$",
    )


if __name__ == "__main__":
    main()
