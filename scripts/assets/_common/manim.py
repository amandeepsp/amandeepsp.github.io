"""Manim utilities and base configuration."""

from pathlib import Path

# Placeholder for manim helpers
# Add common scene configs, color palettes, etc. as needed


def get_output_path(script_file: str, ext: str = "mp4") -> Path:
    """Get manim output path - same logic as paths.output_path."""
    from _common.paths import output_path
    return output_path(script_file, ext)
