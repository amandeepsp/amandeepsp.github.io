#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "matplotlib"
# ]
# ///

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def simulate_step(loads, mode, n, arrivals, rng):
    """Simulate one time step, mutate loads in place."""
    if arrivals <= 0:
        return
    if mode == "one":
        idx = rng.integers(0, n, size=arrivals)
        np.add.at(loads, idx, 1)
    elif mode == "two":
        i1 = rng.integers(0, n, size=arrivals)
        i2 = rng.integers(0, n, size=arrivals)
        loads1 = loads[i1]
        loads2 = loads[i2]
        choose_first = loads1 < loads2
        choose_second = loads2 < loads1
        tie = ~(choose_first | choose_second)

        if choose_first.any():
            np.add.at(loads, i1[choose_first], 1)
        if choose_second.any():
            np.add.at(loads, i2[choose_second], 1)
        if tie.any():
            coin = rng.integers(0, 2, size=tie.sum())
            tie_idx = np.where(coin == 0, i1[tie], i2[tie])
            np.add.at(loads, tie_idx, 1)
    else:
        raise ValueError("mode must be one|two")


def main():
    p = argparse.ArgumentParser(
        description="Compare random vs power-2 load balancing (strip heatmap)"
    )
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--arrivals", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--interval", type=int, default=1, help="ms per frame")
    p.add_argument("--save", type=str, default=None, help="save animation to mp4/gif")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    loads_one = np.zeros(args.n, dtype=int)
    loads_two = np.zeros(args.n, dtype=int)

    # initial frames
    strip_one = np.tile(loads_one, (2, 1))  # shape (2,n)
    strip_two = np.tile(loads_two, (2, 1))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(min(14, args.n / 8), 4), sharex=True)
    im1 = ax1.imshow(
        strip_one, aspect="auto", origin="lower", vmin=0, vmax=args.n // 2 + 5
    )
    im2 = ax2.imshow(
        strip_two, aspect="auto", origin="lower", vmin=0, vmax=args.n // 2 + 5
    )

    ax1.set_title("Random choice (1 server)")
    ax2.set_title("Power of 2 choices")
    ax2.set_xlabel("Server index")
    for ax in (ax1, ax2):
        ax.set_yticks([])

    cbar = fig.colorbar(im2, ax=[ax1, ax2], orientation="vertical")
    cbar.set_label("Load (tasks)")

    def update(frame):
        arrivals = rng.poisson(args.arrivals)
        simulate_step(loads_one, "one", args.n, arrivals, rng)
        simulate_step(loads_two, "two", args.n, arrivals, rng)

        strip_one = np.tile(loads_one, (2, 1))
        strip_two = np.tile(loads_two, (2, 1))
        im1.set_data(strip_one)
        im2.set_data(strip_two)

        ax1.set_title(f"Random choice — step {frame}")
        ax2.set_title(f"Power of 2 — step {frame}")
        return im1, im2

    ani = animation.FuncAnimation(
        fig, update, frames=args.steps, interval=args.interval, blit=False
    )

    if args.save:
        if args.save.endswith(".mp4"):
            Writer = animation.FFMpegWriter
            writer = Writer(fps=24)
            ani.save(args.save, writer=writer)
        elif args.save.endswith(".gif"):
            ani.save(args.save, writer="imagemagick", fps=24)
        print("Saved to", args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
