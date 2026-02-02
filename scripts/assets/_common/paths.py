"""Path utilities for asset scripts."""

from pathlib import Path

ASSETS_DIR = Path(__file__).parent.parent
OUTPUT_DIR = ASSETS_DIR.parent.parent / "public" / "blog"


def output_path(script_file: str, ext: str = "gif") -> Path:
    """
    Get output path for a script.
    
    Usage: output_path(__file__, "svg") → public/blog/<post-slug>/<script>.svg
    """
    script = Path(script_file)
    slug = script.parent.name
    out_dir = OUTPUT_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{script.stem}.{ext}"
