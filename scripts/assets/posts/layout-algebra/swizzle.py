"""Swizzled (interleaved) layout visualization.

Layout: ((2,2),(2,2)) : ((1,4),(2,8))
This is a 4x4 layout with interleaved access pattern used to avoid
shared memory bank conflicts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tensor_layouts import Layout
from tensor_layouts.viz import draw_layout

from _common.paths import output_path


def main():
    swizzled = Layout(((2, 2), (2, 2)), ((1, 4), (2, 8)))

    draw_layout(
        swizzled,
        str(output_path(__file__, "svg")),
        colorize=True,
        label_hierarchy_levels=True,
        title=r"$((2,2),(2,2)) : ((1,4),(2,8))$",
    )


if __name__ == "__main__":
    main()
