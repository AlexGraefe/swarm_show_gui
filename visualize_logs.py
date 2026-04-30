#!/usr/bin/env python3
"""Plot per-drone x/y/z timeseries from a show_results CSV."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

COORD_COLORS = {"x": "tab:red", "y": "tab:green", "z": "tab:blue"}


def drone_indices(df: pd.DataFrame) -> list[int]:
    indices = set()
    for col in df.columns:
        m = re.match(r"actual_[xyz]_(\d+)$", col)
        if m:
            indices.add(int(m.group(1)))
    return sorted(indices)


def plot_csv(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    time = range(len(df))
    drones = drone_indices(df)

    if not drones:
        print("No drone columns found in CSV.", file=sys.stderr)
        sys.exit(1)

    n = len(drones)
    ncols = 7
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True)
    fig.suptitle(csv_path.name)

    # Flatten to a 1-D list; hide any surplus axes.
    ax_flat = [axes[r][c] for r in range(nrows) for c in range(ncols)] if nrows > 1 else list(axes)
    for extra in ax_flat[n:]:
        extra.set_visible(False)

    for ax, drone_idx in zip(ax_flat[:n], drones):
        for coord, color in COORD_COLORS.items():
            actual_col = f"actual_{coord}_{drone_idx}"
            target_col = f"target_{coord}_{drone_idx}"

            if actual_col in df.columns:
                ax.plot(time, df[actual_col], color=color, linewidth=1.2,
                        label=f"{coord} actual")

            if target_col in df.columns:
                ax.plot(time, df[target_col], color=color, linewidth=1.0,
                        linestyle="--", alpha=0.7, label=f"{coord} target")

        ax.set_ylabel("Position (m)")
        ax.set_title(f"Drone {drone_idx}")
        ax.set_ylim(-3.8, 3.8)
        ax.legend(ncol=6, fontsize=7, loc="upper right")
        ax.grid(True, linewidth=0.4)

    for c in range(ncols):
        ax_flat[(nrows - 1) * ncols + c].set_xlabel("Sample")
    fig.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize show_results CSV logs.")
    parser.add_argument("csv", type=Path, help="Path to the show_results CSV file.")
    args = parser.parse_args()

    if not args.csv.is_file():
        print(f"File not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    plot_csv(args.csv)


if __name__ == "__main__":
    main()
