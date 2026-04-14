#!/usr/bin/env python3
"""
Reinforced random-walk simulation plots.

Produces two outputs saved to --output-dir:
  1. random-walk-learning-vs-not            – simple 2-panel overview
  2. random-walk-learning-vs-not-with-hist  – publication figure with histograms
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial"]
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["lines.linewidth"] = 1
matplotlib.rcParams["xtick.major.width"] = 2
matplotlib.rcParams["ytick.major.width"] = 2
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.linewidth"] = 2
matplotlib.rcParams["font.size"] = 20

DEFAULT_OUTPUT_DIR = Path("/mnt/upramdya_data/MD/Affordance_Figures/Figure1") / Path(__file__).stem


def sim_reinforced_walk(num_steps, momentum=0.5, initial_p=0.5, delta=0.1, min_position=0, max_ball=np.inf):
    """Simulate fly-ball interaction with reinforcement learning dynamics.

    Args:
        num_steps: Simulation duration
        momentum: Probability of repeating last movement direction
        initial_p: Initial push probability at boundary
        delta: Learning rate for probability adjustments
        min_position: Minimum position (typically 0)
        max_ball: Absolute maximum possible ball position

    Returns:
        (fly_path, ball_path): Tuple of position arrays
    """
    fly = [0]
    ball = [0]
    current_ball = 0
    p = initial_p
    last_dir = random.choice([-1, 1])

    for _ in range(num_steps - 1):
        # Update fly position with momentum
        direction = last_dir if random.random() < momentum else -last_dir
        new_fly = fly[-1] + direction

        # Keep fly within boundaries (0 to current ball position)
        new_fly = np.clip(new_fly, min_position, current_ball)
        fly.append(new_fly)
        last_dir = direction if new_fly != fly[-2] else last_dir  # Only update direction if moved

        # Attempt push when at boundary
        if new_fly == current_ball and current_ball < max_ball:
            if random.random() < p:  # Successful push
                current_ball += 1
                p = min(p + delta, 1.0)  # Positive reinforcement
            else:  # Failed push
                p = max(p - delta, 0.0)  # Negative reinforcement

        ball.append(current_ball)

    return np.array(fly), np.array(ball)


def main():
    parser = argparse.ArgumentParser(description="Reinforced random-walk simulation plots")
    parser.add_argument("--num-simulations", type=int, default=53)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    random.seed(args.seed)
    num_simulations = args.num_simulations
    steps = args.steps
    output_dir = args.output_dir

    ball_positions_learning = np.array(
        [sim_reinforced_walk(steps, momentum=0.5, initial_p=0.5, delta=0.08)[1] for _ in range(num_simulations)]
    )

    ball_positions_no_learning = np.array(
        [sim_reinforced_walk(steps, momentum=0.5, initial_p=0.5, delta=0.0)[1] for _ in range(num_simulations)]
    )

    # ---- Plot 2: publication figure with histograms ----------------------
    final_no_learning = ball_positions_no_learning[:, -1]
    final_learning = ball_positions_learning[:, -1]

    LABEL_FS = 7
    TICK_FS = 6
    TITLE_FS = 7

    TOTAL_W_MM = 46.073
    TOTAL_H_MM = 33.865
    HIST_W_MM = TOTAL_W_MM * (15.0 / 52.5)  # proportional to experimental layout
    MAIN_W_MM = TOTAL_W_MM - HIST_W_MM
    TOTAL_W_IN = TOTAL_W_MM / 25.4
    TOTAL_H_IN = TOTAL_H_MM / 25.4

    fig, axs = plt.subplots(
        2,
        2,
        sharex="col",
        sharey="row",
        figsize=(TOTAL_W_IN, TOTAL_H_IN),
        gridspec_kw={
            "width_ratios": [MAIN_W_MM, HIST_W_MM],
            "wspace": 0.18,
            "hspace": 0.08,
        },
    )

    ax_traj_0, ax_hist_0 = axs[0, 0], axs[0, 1]
    ax_traj_1, ax_hist_1 = axs[1, 0], axs[1, 1]

    ax_traj_0.plot(ball_positions_no_learning.T, color="grey", linewidth=0.4, alpha=0.5)
    ax_traj_0.text(0.02, 0.96, "no learning", fontsize=TITLE_FS, transform=ax_traj_0.transAxes, va="top", ha="left")

    ax_traj_1.plot(ball_positions_learning.T, color="grey", linewidth=0.4, alpha=0.5)
    ax_traj_1.text(0.02, 0.96, "with learning", fontsize=TITLE_FS, transform=ax_traj_1.transAxes, va="top", ha="left")

    xticks = np.arange(0, steps + 1, 250)
    ax_traj_1.set_xticks(xticks, [tick if tick % 1000 == 0 else "" for tick in xticks])

    yticks = np.arange(0, 150 + 1, 25)
    ax_traj_0.set_yticks(yticks, [tick if tick % 75 == 0 else "" for tick in yticks])
    ax_traj_1.set_yticks(yticks, [tick if tick % 75 == 0 else "" for tick in yticks])

    # Single y-axis label centred across both rows using the figure
    fig.text(0.04, 0.5, "Ball position (a.u.)", va="center", rotation="vertical", fontsize=LABEL_FS)
    ax_traj_1.set_xlabel("Time steps (a.u.)", fontsize=LABEL_FS, labelpad=2)

    for ax_traj in (ax_traj_0, ax_traj_1):
        ax_traj.tick_params(labelsize=TICK_FS, width=0.4, length=1.35)
        for spine in ax_traj.spines.values():
            spine.set_linewidth(0.4)

    # Tight histogram bins (no space between bars)
    shared_ymin = min(ax_traj_0.get_ylim()[0], ax_traj_1.get_ylim()[0])
    shared_ymax = max(ax_traj_0.get_ylim()[1], ax_traj_1.get_ylim()[1])
    bin_width = 15
    bins = np.arange(shared_ymin, shared_ymax + bin_width, bin_width)

    ax_hist_0.hist(
        final_no_learning,
        bins=bins,
        orientation="horizontal",
        color="grey",
        edgecolor="none",
        alpha=0.9,
    )
    ax_hist_1.hist(
        final_learning,
        bins=bins,
        orientation="horizontal",
        color="grey",
        edgecolor="none",
        alpha=0.9,
    )

    for ax_hist in (ax_hist_0, ax_hist_1):
        ax_hist.yaxis.set_visible(False)
        ax_hist.set_xlim(0, 10)
        ax_hist.set_xticks([0, 5, 10])
        ax_hist.tick_params(labelsize=TICK_FS, width=0.4, length=1.35)
        for spine in ax_hist.spines.values():
            spine.set_linewidth(0.4)

    ax_hist_1.set_xlabel("count", fontsize=LABEL_FS, labelpad=2)

    ax_hist_0.set_ylim(ax_traj_0.get_ylim())
    ax_hist_1.set_ylim(ax_traj_1.get_ylim())

    fig.subplots_adjust(left=0.22, right=0.955, bottom=0.22, top=0.985)

    output_dir.mkdir(parents=True, exist_ok=True)
    out2 = output_dir / "Fig1_f_simulatedTrajectories"
    fig.savefig(out2.with_suffix(".pdf"))
    # fig.savefig(out2.with_suffix(".svg"))
    plt.close(fig)
    print(f"Saved: {out2.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
