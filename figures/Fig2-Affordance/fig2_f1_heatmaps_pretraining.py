#!/usr/bin/env python3
"""
Figure 2 — F1 fly-position heatmaps grouped by Pretraining (naive vs pretrained).

Linear-clipped scale, with 200 px proximity circle around the ball.
Outputs: PDF + PNG.

Usage:
    python fig2_f1_heatmaps_pretraining.py [--test]
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter

from ballpushing_utils import dataset, figure_output_dir, read_feather
from ballpushing_utils.plotting import set_illustrator_style

set_illustrator_style()

# --- Paths ---
TEMPLATE_PATH = dataset("F1_Tracks/F1_New_Template.png")
DATASET_PATH = dataset(
    "F1_Tracks/Datasets/251121_17_summary_F1_New_Data/fly_positions/pooled_fly_positions.feather"
)
VIDEOS_BASE = dataset("F1_Tracks/Videos/251008_F1_New_Videos_Checked")

# --- Parameters ---
BINS = 100
BLUR_SIGMA = 2.0
CLIP_PERCENTILE = 95
PROXIMITY_RADIUS_PX = 200
CONDITION_LABELS = {"n": "Naive", "y": "Pretrained"}


# ---------------------------------------------------------------------------
# Image / arena helpers
# ---------------------------------------------------------------------------


def binarize_template(template):
    gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    return binary


def binarize_video_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def add_padding(image, padding_percent=0.4):
    h, w = image.shape[:2]
    pad_h = int(h * padding_percent)
    pad_w = int(w * padding_percent)
    if len(image.shape) == 2:
        padded = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=[0])
    else:
        mean_color = np.mean(image, axis=(0, 1))
        padded = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=mean_color.tolist())
    return padded, (pad_h, pad_w)


def match_binary_multiscale(frame_binary, template_binary, scale_range=(0.3, 0.8), scale_steps=60):
    frame_h, frame_w = frame_binary.shape
    template_h, template_w = template_binary.shape

    best_score = -np.inf
    best_loc = best_scale = best_template = None

    for scale in np.linspace(scale_range[0], scale_range[1], scale_steps):
        new_w, new_h = int(template_w * scale), int(template_h * scale)
        if new_w > frame_w or new_h > frame_h:
            continue
        resized = cv2.resize(template_binary, (new_w, new_h), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(frame_binary, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val > best_score:
            best_score, best_loc, best_scale, best_template = max_val, max_loc, scale, resized

    if best_loc is None:
        return None
    return {
        "location": best_loc,
        "score": best_score,
        "scale": best_scale,
        "size": (best_template.shape[1], best_template.shape[0]),
    }


def detect_arena_in_video(video_path, template_binary, padding_percent=0.4):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames - 1))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None

    frame_padded, (pad_h, pad_w) = add_padding(frame, padding_percent)
    frame_binary = binarize_video_frame(frame_padded)
    match = match_binary_multiscale(frame_binary, template_binary)

    if match is None or match["score"] < 0.7:
        return None

    padded_loc = match["location"]
    return {
        "score": match["score"],
        "scale": match["scale"],
        "arena_x": padded_loc[0] - pad_w,
        "arena_y": padded_loc[1] - pad_h,
        "arena_width": match["size"][0],
        "arena_height": match["size"][1],
    }


def transform_positions(fly_data, arena_params, template_shape, x_col, y_col):
    df = fly_data.copy()
    scale = arena_params["scale"]
    df["x_template"] = (df[x_col].values - arena_params["arena_x"]) / scale
    df["y_template"] = (df[y_col].values - arena_params["arena_y"]) / scale
    th, tw = template_shape[:2]
    mask = (df["x_template"] >= 0) & (df["x_template"] < tw) & (df["y_template"] >= 0) & (df["y_template"] < th)
    return df[mask].copy()


def filter_from_first_movement(df, x_col, threshold=100, n_avg=100):
    parts = []
    for fly_id, fly_data in df.groupby("fly"):
        fly_data = fly_data.sort_values("frame") if "frame" in fly_data.columns else fly_data.copy()
        n_use = min(n_avg, len(fly_data))
        if n_use == 0:
            continue
        x_start = fly_data[x_col].iloc[:n_use].mean()
        dist = (fly_data[x_col] - x_start).abs()
        first_idx = dist[dist > threshold].index
        if len(first_idx):
            parts.append(fly_data.loc[first_idx[0] :])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def get_video_path(fly_id, videos_base):
    parts = str(fly_id).split("_")
    arena = side = None
    for i, p in enumerate(parts):
        if "arena" in p.lower() and arena is None:
            try:
                arena = p
                side = parts[i + 1]
            except IndexError:
                pass
    if arena is None or side is None:
        return None
    path = videos_base / arena / side / f"{side}.mp4"
    return path if path.exists() else None


def build_heatmap(data, template_shape, bins=BINS, blur_sigma=BLUR_SIGMA):
    th, tw = template_shape[:2]
    x_edges = np.linspace(0, tw, bins + 1)
    y_edges = np.linspace(0, th, bins + 1)

    fly_heatmaps = []
    for _, fly_data in data.groupby("fly"):
        x = fly_data["x_template"].values
        y = fly_data["y_template"].values
        if len(x) == 0:
            continue
        h, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
        fly_heatmaps.append(h / len(x))

    if not fly_heatmaps:
        return None

    avg = np.median(fly_heatmaps, axis=0)
    if blur_sigma > 0:
        avg = gaussian_filter(avg, sigma=blur_sigma)

    # Upscale and transpose to (height, width)
    upscaled = cv2.resize(avg.T, (tw, th), interpolation=cv2.INTER_LINEAR)
    return upscaled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(test_mode=False):
    output_dir = figure_output_dir("Figure2", __file__)

    # Load template
    print("Loading template...")
    template = cv2.imread(str(TEMPLATE_PATH))
    if template is None:
        raise FileNotFoundError(f"Template not found: {TEMPLATE_PATH}")
    template_binary = binarize_template(template)
    template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)

    # Load dataset
    print("Loading dataset...")
    df = read_feather(DATASET_PATH)
    print(f"  {len(df):,} rows, {df['fly'].nunique()} flies")

    # Find position columns
    x_col = next((c for c in ["x_thorax_fly_0", "x_thorax", "x_thorax_skeleton"] if c in df.columns), None)
    y_col = next((c for c in ["y_thorax_fly_0", "y_thorax", "y_thorax_skeleton"] if c in df.columns), None)
    if x_col is None or y_col is None:
        raise ValueError("Could not find thorax position columns")
    print(f"  Position columns: {x_col}, {y_col}")

    if "Pretraining" not in df.columns:
        raise ValueError("'Pretraining' column not found in dataset")

    # Keep only data from when the fly is in the second corridor (positive adjusted time)
    if "adjusted_time_fly_0" in df.columns:
        before = len(df)
        df = df[df["adjusted_time_fly_0"] > 0].copy()
        print(
            f"  Filtered to adjusted_time_fly_0 > 0: {len(df):,} rows ({len(df)/before*100:.1f}%), {df['fly'].nunique()} flies"
        )
    else:
        print("  WARNING: 'adjusted_time_fly_0' column not found; skipping corridor filter")

    # Filter from first movement
    print("Filtering from first movement >100 px...")
    df = filter_from_first_movement(df, x_col)
    print(f"  {len(df):,} rows, {df['fly'].nunique()} flies remaining")

    conditions = ["n", "y"]

    if test_mode:
        test_ids = []
        for c in conditions:
            test_ids.extend(df[df["Pretraining"] == c]["fly"].unique()[:2])
        df = df[df["fly"].isin(test_ids)].copy()
        print(f"Test mode: {df['fly'].nunique()} flies")

    # Align all flies to template
    print("Aligning flies to template...")
    aligned = {c: [] for c in conditions}
    n_ok = n_fail = 0

    fly_ids = df["fly"].unique()
    for i, fly_id in enumerate(fly_ids, 1):
        if i % 20 == 0:
            print(f"  {i}/{len(fly_ids)}...")
        fly_data = df[df["fly"] == fly_id].copy()
        cond = fly_data["Pretraining"].iloc[0]
        if cond not in conditions or pd.isna(cond):
            continue
        vp = get_video_path(fly_id, VIDEOS_BASE)
        if vp is None:
            n_fail += 1
            continue
        arena = detect_arena_in_video(vp, template_binary)
        if arena is None:
            n_fail += 1
            continue
        transformed = transform_positions(fly_data, arena, template.shape, x_col, y_col)
        if len(transformed) == 0:
            n_fail += 1
            continue
        aligned[cond].append(transformed)
        n_ok += 1

    print(f"  Processed {n_ok} flies, {n_fail} failed")

    condition_data = {}
    for c in conditions:
        if aligned[c]:
            condition_data[c] = pd.concat(aligned[c], ignore_index=True)
            print(f"  {CONDITION_LABELS[c]}: {condition_data[c]['fly'].nunique()} flies")

    if not condition_data:
        raise RuntimeError("No aligned data to plot")

    # Compute common ball position (ball_1)
    print("Computing ball position...")
    ball_pos = None
    all_data = pd.concat(condition_data.values(), ignore_index=True)
    if "x_centre_ball_1" in all_data.columns and "y_centre_ball_1" in all_data.columns:
        sample_fly = all_data["fly"].iloc[0]
        sample_vp = get_video_path(sample_fly, VIDEOS_BASE)
        if sample_vp:
            arena = detect_arena_in_video(sample_vp, template_binary)
            if arena:
                bx = (all_data["x_centre_ball_1"].median() - arena["arena_x"]) / arena["scale"]
                by = (all_data["y_centre_ball_1"].median() - arena["arena_y"]) / arena["scale"]
                th, tw = template.shape[:2]
                if 0 <= bx < tw and 0 <= by < th:
                    ball_pos = (bx, by)
                    print(f"  Ball position: ({bx:.1f}, {by:.1f})")

    # Build heatmaps
    print("Building heatmaps...")
    heatmaps = {}
    for c, data in condition_data.items():
        heatmaps[c] = build_heatmap(data, template.shape)

    # ---- Collect positive heatmap values for scale computations ------------
    all_vals = np.concatenate([h[h > 0] for h in heatmaps.values() if h is not None])
    print(f"  heatmap value range (raw): {all_vals.min():.2e} – {all_vals.max():.2e}")

    # ---- Display assets: rotate 90° CW, replace outer PNG white with black -
    th, tw = template.shape[:2]
    template_mask = binarize_template(template) > 0

    outer_bg = (template_rgb[..., 0] > 252) & (template_rgb[..., 1] > 252) & (template_rgb[..., 2] > 252)
    template_rgb_fixed = template_rgb.copy()
    template_rgb_fixed[outer_bg] = [0, 0, 0]

    template_rgb_rot = np.rot90(template_rgb_fixed, k=3)  # 90° CW → (tw, th, 3)
    template_mask_rot = np.rot90(template_mask, k=3)  # (tw, th)
    rot_th, rot_tw = template_rgb_rot.shape[:2]  # rot_th=tw, rot_tw=th

    # Ball position after 90° CW: old (x, y) → new (th − y,  x)
    ball_pos_rot = (float(th) - ball_pos[1], float(ball_pos[0])) if ball_pos is not None else None

    # ---- Layout constants -------------------------------------------------
    FIG_W_IN = 2.8256 / 2.54
    FIG_H_IN = 2.8479 / 2.54
    FS = 5  # all text 5 pt, no bold

    COND_CONFIG = [
        ("n", "No training ball", "#cb7c29"),
        ("y", "With training ball", "#4C72B0"),
    ]
    SAMPLE_SIZES = {"n": 30, "y": 39}

    SCALE = 1e4
    _p95 = float(np.percentile(all_vals, 95))
    _p99 = float(np.percentile(all_vals, 99))
    _max = float(all_vals.max())
    log_vals = np.log10(all_vals)

    def _linear_cfg(vmax_raw, label_suffix):
        vm = vmax_raw * SCALE
        ticks = [t for t in range(0, int(np.ceil(vm)) + 2, 2) if t <= vm + 0.5]
        return dict(
            transform=lambda h, S=SCALE: np.where(h > 0, h * S, np.nan),
            vmin=0,
            vmax=vm,
            ticks=ticks,
            tick_labels=[str(t) for t in ticks],
            cbar_label=f"Avg. norm. density (fraction/bin)\n({label_suffix})",
            multiplier=r"$\times 10^{-4}$",
        )

    STRATEGIES = [
        ("raw", _linear_cfg(_max, "raw median")),
        ("p95", _linear_cfg(_p95, "clipped at 95th %ile")),
        ("p99", _linear_cfg(_p99, "clipped at 99th %ile")),
        (
            "sqrt",
            dict(
                transform=lambda h: np.where(h > 0, np.sqrt(h), np.nan),
                vmin=0,
                vmax=float(np.sqrt(_max)),
                ticks=None,
                tick_labels=None,
                cbar_label="√(avg. norm. density)",
                multiplier="",
            ),
        ),
        (
            "log",
            dict(
                transform=lambda h: np.where(h > 0, np.log10(h), np.nan),
                vmin=float(log_vals.min()),
                vmax=float(log_vals.max()),
                ticks=None,
                tick_labels=None,
                cbar_label="log₁₀(avg. norm. density)",
                multiplier="",
            ),
        ),
    ]

    # ---- Plot loop: one PDF per scale strategy ----------------------------
    print("Plotting...")
    for strategy_name, scfg in STRATEGIES:
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(alpha=0)

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(FIG_W_IN, FIG_H_IN),
            gridspec_kw={"hspace": 0.18},
        )

        last_im = None
        for i, (ax, (cond, label, label_color)) in enumerate(zip(axes, COND_CONFIG)):
            heatmap = heatmaps.get(cond)

            ax.set_facecolor("black")
            ax.imshow(template_rgb_rot, aspect="equal", alpha=1.0, rasterized=True)

            if heatmap is not None:
                hm = scfg["transform"](np.rot90(heatmap.copy().astype(float), k=3))
                # Inside-mask bins with no data → vmin color (deep blue); outside mask → transparent
                hm[template_mask_rot & np.isnan(hm)] = scfg["vmin"]
                hm[~template_mask_rot] = np.nan

                last_im = ax.imshow(
                    hm,
                    extent=(0, rot_tw, rot_th, 0),
                    origin="upper",
                    cmap=cmap,
                    alpha=0.85,
                    interpolation="bilinear",
                    vmin=scfg["vmin"],
                    vmax=scfg["vmax"],
                    rasterized=True,
                )

            # White dashed proximity circle — 0.5 px, no center marker
            if ball_pos_rot is not None:
                ax.add_patch(
                    Circle(
                        ball_pos_rot,
                        PROXIMITY_RADIUS_PX,
                        color="white",
                        fill=False,
                        linestyle="--",
                        linewidth=0.5,
                        alpha=0.9,
                    )
                )

            # "12 mm\nradius" annotation — first subplot only, top-right of circle in dark wall area
            if i == 0 and ball_pos_rot is not None:
                ax.text(
                    0.55,
                    0.60,
                    "12 mm\nradius",
                    transform=ax.transAxes,
                    color="white",
                    fontsize=FS,
                    ha="center",
                    va="center",
                    linespacing=0.9,
                )

            ax.set_xlim(0, rot_tw)
            ax.set_ylim(rot_th, 0)
            ax.set_aspect("equal")
            ax.axis("off")

            # Condition label and sample size ABOVE axes (clip_on=False)
            n_display = SAMPLE_SIZES.get(cond, "?")
            ax.text(
                0.0,
                1.02,
                label,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=FS,
                color=label_color,
                clip_on=False,
            )
            ax.text(
                1.0,
                1.02,
                f"n={n_display}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=FS,
                color="#888888",
                clip_on=False,
            )

        # Shared colorbar spanning both axes
        cbar_kw = dict(ax=axes.tolist(), location="right", pad=0.02, fraction=0.08, aspect=25)
        if scfg["ticks"] is not None:
            cbar_kw["ticks"] = scfg["ticks"]
        cbar = fig.colorbar(last_im, **cbar_kw)
        cbar.outline.set_linewidth(0.3)
        if scfg["tick_labels"] is not None:
            cbar.ax.set_yticklabels(scfg["tick_labels"], fontsize=FS)
        else:
            cbar.ax.tick_params(labelsize=FS)
        # Label constrained to colorbar bounds
        cbar.set_label(scfg["cbar_label"], size=FS, labelpad=3)
        # Multiplier "10^-4" placed inside the colorbar near the top tick (not above it)
        if scfg["multiplier"]:
            cbar.ax.text(
                1.02,
                1.02,
                r"$10^{-4}$",
                transform=cbar.ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=FS,
                clip_on=False,
            )

        out_path = output_dir / f"fig2_heatmaps_pretraining_{strategy_name}.pdf"
        fig.savefig(out_path, dpi=300, bbox_inches="tight", format="pdf")
        plt.close(fig)
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fig. 2 F1 fly-position heatmaps by pretraining")
    parser.add_argument("--test", action="store_true", help="Run on 2 flies per condition")
    args = parser.parse_args()
    main(test_mode=args.test)
