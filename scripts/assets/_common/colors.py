"""Color palette for blog visualizations.

These colors are designed to work well with the blog's warm, neutral theme
and maintain good contrast on both light and dark backgrounds.
"""

# Base theme colors (from global.css)
TEXT_DARK = "#171717"
TEXT_LIGHT = "#F2F1EC"
BG_LIGHT = "#F2F1EC"
BG_DARK = "#171717"

# Semantic colors for visualizations
class Colors:
    """Color palette for matplotlib visualizations."""

    # Primary accent - warm teal/blue
    PRIMARY = "#2a7f8e"
    PRIMARY_LIGHT = "#4da6b5"

    # Secondary - warm coral/rust
    SECONDARY = "#c45a3b"
    SECONDARY_LIGHT = "#e07a5f"

    # Tertiary - muted gold
    TERTIARY = "#b5942a"
    TERTIARY_LIGHT = "#d4b545"

    # Success/destination - sage green
    SUCCESS = "#5a8a6e"
    SUCCESS_LIGHT = "#7eb090"

    # Warning/start - amber
    WARNING = "#c9913a"
    WARNING_LIGHT = "#e5b565"

    # Query/target - deep rose
    QUERY = "#a8495a"
    QUERY_LIGHT = "#c96b7c"

    # Neutral grays
    GRAY_900 = "#1a1a1a"
    GRAY_700 = "#404040"
    GRAY_500 = "#737373"
    GRAY_400 = "#a3a3a3"
    GRAY_300 = "#c4c4c4"
    GRAY_200 = "#e0e0e0"
    GRAY_100 = "#f0f0f0"

    # Points and nodes
    NODE = GRAY_700
    NODE_HIGHLIGHT = PRIMARY

    # Edges and paths
    EDGE = GRAY_300
    EDGE_HIGHLIGHT = PRIMARY
    PATH = PRIMARY

    # Regions and fills
    REGION_FILL = GRAY_400
    REGION_FILL_ALPHA = 0.2
    REGION_HIGHLIGHT = PRIMARY_LIGHT
    REGION_HIGHLIGHT_ALPHA = 0.4

    # Specific visualization elements
    START_POINT = WARNING
    END_POINT = SUCCESS
    QUERY_POINT = QUERY

    # Text annotations
    TEXT = GRAY_900
    TEXT_MUTED = GRAY_500
