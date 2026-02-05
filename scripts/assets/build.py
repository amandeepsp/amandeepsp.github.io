"""
Build script that discovers and runs asset generators incrementally.

Usage: uv run python scripts/assets/build.py [--force]
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ASSETS_DIR = Path(__file__).parent
POSTS_DIR = ASSETS_DIR / "posts"
COMMON_DIR = ASSETS_DIR / "_common"
CACHE_FILE = ASSETS_DIR / ".asset-cache.json"


def hash_file(path: Path) -> str:
    """Return hash of file contents + mtime."""
    stat = path.stat()
    content_hash = hashlib.md5(path.read_bytes()).hexdigest()
    return f"{content_hash}:{stat.st_mtime_ns}"


def get_dependencies(script: Path) -> list[Path]:
    """Get all files a script depends on."""
    deps = [script]
    for common_file in sorted(COMMON_DIR.glob("*.py")):
        if not common_file.name.startswith("_"):
            deps.append(common_file)
    return deps


def compute_cache_key(script: Path) -> str:
    """Compute a cache key from all dependencies."""
    deps = get_dependencies(script)
    hashes = [f"{p.name}={hash_file(p)}" for p in deps if p.exists()]
    return hashlib.md5("|".join(hashes).encode()).hexdigest()


def load_cache() -> dict:
    """Load the build cache."""
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def save_cache(cache: dict):
    """Save the build cache."""
    CACHE_FILE.write_text(json.dumps(cache, indent=2))


def get_output_paths(script: Path) -> list[Path]:
    """Infer possible output paths for a script (matches _common/paths.py logic)."""
    from _common.paths import OUTPUT_DIR

    slug = script.parent.name
    paths = []
    for ext in ["svg", "gif", "png", "mp4"]:
        paths.append(OUTPUT_DIR / slug / f"{script.stem}.{ext}")
    return paths


def needs_rebuild(script: Path, cache: dict) -> bool:
    """Check if a script needs to be rebuilt."""
    cache_key = compute_cache_key(script)
    script_id = f"{script.parent.name}/{script.name}"

    cached_key = cache.get(script_id)
    if cached_key != cache_key:
        return True

    outputs = get_output_paths(script)
    if not any(p.exists() for p in outputs):
        return True

    return False


def build_all(force: bool = False):
    cache = load_cache()
    rebuilt = 0
    skipped = 0

    for post_dir in sorted(POSTS_DIR.iterdir()):
        if not post_dir.is_dir() or post_dir.name.startswith("_"):
            continue

        for script in sorted(post_dir.glob("*.py")):
            if script.name.startswith("_"):
                continue

            script_id = f"{post_dir.name}/{script.name}"

            if not force and not needs_rebuild(script, cache):
                print(f"[{post_dir.name}] {script.name} (cached)")
                skipped += 1
                continue

            print(f"[{post_dir.name}] {script.name}")
            subprocess.run(
                [sys.executable, str(script)],
                cwd=ASSETS_DIR,
                check=True,
            )

            cache[script_id] = compute_cache_key(script)
            rebuilt += 1

    save_cache(cache)
    print(f"Done. {rebuilt} rebuilt, {skipped} cached.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build asset generators")
    parser.add_argument("--force", "-f", action="store_true", help="Force rebuild all")
    args = parser.parse_args()

    build_all(force=args.force)
