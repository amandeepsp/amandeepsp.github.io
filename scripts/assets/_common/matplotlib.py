"""Matplotlib utilities for figures and animations."""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path


def save_figure(fig: plt.Figure, path: Path, **kwargs):
    """Save a matplotlib figure."""
    fig.savefig(path, **kwargs)
    print(f"Saved figure to {path}")
    plt.close(fig)


def save_animation(
    ani: animation.Animation,
    path: Path,
    fps: int = 24,
):
    """Save a matplotlib animation to gif or mp4."""
    ext = path.suffix.lower()
    if ext == ".mp4":
        writer = animation.FFMpegWriter(fps=fps)
        ani.save(path, writer=writer)
    elif ext == ".gif":
        ani.save(path, writer="pillow", fps=fps)
    else:
        raise ValueError(f"Unsupported format: {ext}")
    print(f"Saved animation to {path}")
    plt.close(ani._fig)
