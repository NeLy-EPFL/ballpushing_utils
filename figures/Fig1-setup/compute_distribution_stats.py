#!/usr/bin/env python3
"""
Normality statistics for final ball positions:
  - Simulation: random walk without learning
  - Simulation: random walk with learning
  - Wildtype flies (from coordinates feather file)

Tests run per distribution:
  - Shapiro-Wilk            (scipy.stats.shapiro)
  - D'Agostino-Pearson K²   (scipy.stats.normaltest)
  - Kolmogorov-Smirnov vs fitted normal  (scipy.stats.kstest)

Descriptive stats included: n, mean, std, skewness, kurtosis (excess).

Output: final_position_normality_stats.csv in --output-dir.

Usage:
  python compute_distribution_stats.py
  python compute_distribution_stats.py --feather path/to/file.feather
  python compute_distribution_stats.py --coordinates-dir path/to/coords/dir/
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import diptest
import numpy as np
import pandas as pd
import scipy.stats as stats

from ballpushing_utils import dataset, figure_output_dir

# ---- defaults matching the two sibling scripts ---------------------------
# All paths route through ``dataset()`` / ``figure_output_dir()`` so the
# script picks up ``BALLPUSHING_DATA_ROOT`` / ``BALLPUSHING_FIGURES_ROOT``
# env vars on machines where the lab share isn't mounted at the default
# location.
DEFAULT_FEATHER = dataset(
    "Ballpushing_Exploration/Datasets/260220_10_summary_control_folders_Data"
    "/coordinates/230704_FeedingState_1_AM_Videos_Tracked_coordinates.feather"
)
DEFAULT_COORDINATES_DIR = dataset(
    "Ballpushing_Exploration/Datasets/260220_10_summary_control_folders_Data/coordinates"
)
DEFAULT_OUTPUT_DIR = figure_output_dir("Figure1", __file__, create=False)

PX_PER_MM = 500 / 30


# ---- simulation ----------------------------------------------------------


def sim_reinforced_walk(num_steps, momentum=0.5, initial_p=0.5, delta=0.1, min_position=0, max_ball=np.inf):
    fly = [0]
    ball = [0]
    current_ball = 0
    p = initial_p
    last_dir = random.choice([-1, 1])

    for _ in range(num_steps - 1):
        direction = last_dir if random.random() < momentum else -last_dir
        new_fly = fly[-1] + direction
        new_fly = np.clip(new_fly, min_position, current_ball)
        fly.append(new_fly)
        last_dir = direction if new_fly != fly[-2] else last_dir

        if new_fly == current_ball and current_ball < max_ball:
            if random.random() < p:
                current_ball += 1
                p = min(p + delta, 1.0)
            else:
                p = max(p - delta, 0.0)

        ball.append(current_ball)

    return np.array(fly), np.array(ball)


def simulate_final_positions(num_simulations: int, steps: int, seed: int, delta: float) -> np.ndarray:
    random.seed(seed)
    return np.array(
        [sim_reinforced_walk(steps, momentum=0.5, initial_p=0.5, delta=delta)[1][-1] for _ in range(num_simulations)]
    )


# ---- wildtype data -------------------------------------------------------


def load_wildtype_final_positions(feather_path: Path | None, coordinates_dir: Path | None) -> np.ndarray:
    """Return array of per-fly final aligned ball positions (mm)."""
    if feather_path is not None:
        frames = [pd.read_feather(feather_path)]
    else:
        feather_files = sorted(coordinates_dir.glob("*_coordinates.feather"))
        if not feather_files:
            raise FileNotFoundError(f"No feather files in {coordinates_dir}")
        frames = [pd.read_feather(fp) for fp in feather_files]

    combined = pd.concat(frames, ignore_index=True)

    combined["distance_ball_0_mm"] = combined.groupby("fly")["y_ball_0"].transform(
        lambda y: -(y - y.iloc[0]) / PX_PER_MM
    )

    final_positions = combined.sort_values("time").groupby("fly")["distance_ball_0_mm"].last().dropna().values
    return final_positions


# ---- statistics ----------------------------------------------------------


def run_normality_tests(label: str, values: np.ndarray) -> dict:
    n = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    skew = float(stats.skew(values))
    kurt = float(stats.kurtosis(values))  # excess kurtosis

    row: dict = {
        "distribution": label,
        "n": n,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "skewness": round(skew, 4),
        "excess_kurtosis": round(kurt, 4),
    }

    # Shapiro-Wilk
    if n <= 5000:
        sw_stat, sw_p = stats.shapiro(values)
        row["shapiro_wilk_W"] = round(sw_stat, 5)
        row["shapiro_wilk_p"] = round(sw_p, 5)
    else:
        row["shapiro_wilk_W"] = None
        row["shapiro_wilk_p"] = None  # not reliable for n > 5000

    # D'Agostino-Pearson K²
    dp_stat, dp_p = stats.normaltest(values)
    row["dagostino_pearson_K2"] = round(dp_stat, 5)
    row["dagostino_pearson_p"] = round(dp_p, 5)

    # KS test against fitted normal (parameters estimated from data)
    loc, scale = stats.norm.fit(values)
    ks_stat, ks_p = stats.kstest(values, "norm", args=(loc, scale))
    row["ks_stat"] = round(ks_stat, 5)
    row["ks_p"] = round(ks_p, 5)

    # Hartigan's dip test for unimodality
    dip_stat, dip_p = diptest.diptest(values)
    row["hartigan_dip"] = round(dip_stat, 5)
    row["hartigan_dip_p"] = round(dip_p, 5)

    return row


# ---- main ----------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Normality tests for final ball positions")
    parser.add_argument("--feather", type=Path, default=None, help="Single coordinates feather file for wildtype data")
    parser.add_argument(
        "--coordinates-dir",
        type=Path,
        default=None,
        help="Directory of coordinates feather files (all loaded and pooled)",
    )
    parser.add_argument("--num-simulations", type=int, default=53)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    rows = []

    # Simulation: no learning
    print("Simulating random walk (no learning)...")
    no_learning = simulate_final_positions(args.num_simulations, args.steps, args.seed, delta=0.0)
    rows.append(run_normality_tests("simulation_no_learning", no_learning))
    print(f"  n={len(no_learning)}, mean={no_learning.mean():.1f}")

    # Simulation: with learning
    print("Simulating random walk (with learning)...")
    with_learning = simulate_final_positions(args.num_simulations, args.steps, args.seed, delta=0.08)
    rows.append(run_normality_tests("simulation_with_learning", with_learning))
    print(f"  n={len(with_learning)}, mean={with_learning.mean():.1f}")

    # Wildtype
    feather_path = args.feather
    coordinates_dir = args.coordinates_dir

    if feather_path is None and coordinates_dir is None:
        # fall back to defaults
        if DEFAULT_FEATHER.exists():
            feather_path = DEFAULT_FEATHER
        elif DEFAULT_COORDINATES_DIR.exists():
            coordinates_dir = DEFAULT_COORDINATES_DIR
        else:
            print("Warning: no wildtype data found; skipping wildtype statistics.")

    if feather_path is not None or coordinates_dir is not None:
        print("Loading wildtype data...")
        wt = load_wildtype_final_positions(feather_path, coordinates_dir)
        rows.append(run_normality_tests("wildtype", wt))
        print(f"  n={len(wt)}, mean={wt.mean():.1f}")

    df = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "final_position_normality_stats.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
