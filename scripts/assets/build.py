"""
Build script that discovers and runs all asset generators.

Usage: uv run python scripts/assets/build.py
"""

import subprocess
import sys
from pathlib import Path

ASSETS_DIR = Path(__file__).parent
POSTS_DIR = ASSETS_DIR / "posts"


def build_all():
    for post_dir in sorted(POSTS_DIR.iterdir()):
        if not post_dir.is_dir() or post_dir.name.startswith("_"):
            continue

        for script in sorted(post_dir.glob("*.py")):
            if script.name.startswith("_"):
                continue
            print(f"[{post_dir.name}] {script.name}")
            subprocess.run(
                [sys.executable, str(script)],
                cwd=ASSETS_DIR,
                check=True,
            )

    print("Done.")


if __name__ == "__main__":
    build_all()
