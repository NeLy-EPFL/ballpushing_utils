#!/usr/bin/env python
"""Compare the new affine-fit ball preprocessing against the legacy geometry-based math.

Background
----------
:class:`ballpushing_utils.SkeletonMetrics` brings raw ball coordinates
(in the per-corridor video's pixel space, where the fly tracker also
lives) into the skeleton's *template* coordinate space, so contact
detection between fly skeleton keypoints and the ball can run on a
shared coordinate system.

Two implementations exist:

1. **Affine fit** (new, default): fit
   :math:`x_\\text{tmpl} = s_x \\cdot x_\\text{raw} + o_x` and same for
   y from matched-frame thorax positions in the fly tracker (raw video
   coords, lowercase ``x_thorax``) and the skeleton tracker (template
   coords, capitalised ``x_Thorax``). No video file needed.
2. **Geometry-based** (legacy fallback): rescale + crop + pad raw ball
   coords using ``original_size`` (read from the ``.mp4`` via OpenCV)
   and the static ``Config.template_*`` / ``padding`` / ``y_crop``
   constants.

The new method assumes a strict translation + uniform-scale
relationship. If the actual transform between raw and template
coordinates includes any non-uniform scaling — e.g. because the
preprocessing pipeline cropped each corridor to a slightly different
height before resizing to the canonical 96×516 — the fit will be
biased. This script measures whether that bias affects downstream
contact detection enough to matter.

What it does
------------
For each fly directory provided:

1. Build a ``Fly`` and run the **new** preprocessing path (affine fit).
2. Build a fresh ``Fly`` and run the **legacy** preprocessing path
   (forced via a subclass that overrides
   ``_estimate_raw_to_template_transform`` to ``None``).
3. Compare:

   - **Pixel-space ball drift**: per-frame Euclidean distance between
     the two preprocessed ball trajectories. Mean / max / 95th pct.
   - **Contact-mask IoU**: build a binary "in contact" mask for each
     method (1 if any contact event covers that frame, else 0). Report
     intersection-over-union — 1.0 means perfect agreement, 0.0 means
     disjoint.
   - **Contact-count delta**: number of contact events found by each
     method. Big mismatches indicate one method is dropping or
     fragmenting contacts the other isn't.

Usage
-----
::

    # Pass fly directories explicitly (recommended for a quick smoke):
    python tools/compare_ball_preprocess.py \\
        /path/to/fly_dir1 /path/to/fly_dir2 ...

    # Or via a YAML of flies (same format as dataset_builder --yaml):
    python tools/compare_ball_preprocess.py --yaml flies.yaml --max-flies 10

    # Append a CSV summary for later inspection:
    python tools/compare_ball_preprocess.py --yaml flies.yaml \\
        --out compare_report.csv

Each fly must have its three SLEAP HDF5 tracks (ball, fly, full_body)
**and** a ``.mp4`` next to them — the legacy method needs the video to
read ``original_size``. Use this for validation against the lab share;
re-running on a Dataverse-only download would force-skip every fly
through the legacy fallback and defeat the comparison.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

from ballpushing_utils import Fly
from ballpushing_utils.skeleton_metrics import SkeletonMetrics


# ---------------------------------------------------------------------------
# Forced-legacy SkeletonMetrics
# ---------------------------------------------------------------------------


class LegacySkeletonMetrics(SkeletonMetrics):
    """SkeletonMetrics that always uses the geometry-based ball preprocessing.

    Returning ``None`` from the affine-fit estimator routes
    ``preprocess_ball`` into its fallback branch, which calls
    ``resize_and_transform_coordinate`` with
    ``self.fly.metadata.original_size``. That requires a real ``.mp4``
    next to the SLEAP tracks (or an explicit
    ``Config.default_video_size`` override) — both of which exist on
    the lab share.
    """

    def _estimate_raw_to_template_transform(self):  # type: ignore[override]
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_fly_paths(args) -> list[Path]:
    """Resolve fly directories from --yaml or positional args."""
    if args.yaml:
        with open(args.yaml) as f:
            data = yaml.safe_load(f) or {}
        return [Path(d) for d in data.get("directories", [])]
    return [Path(p) for p in args.dirs]


def _contact_mask(contacts: Iterable, n_frames: int) -> np.ndarray:
    """Build a per-frame binary mask from a list of [start, end, length] runs."""
    mask = np.zeros(n_frames, dtype=bool)
    for c in contacts or []:
        if hasattr(c, "__len__") and len(c) >= 2:
            s, e = int(c[0]), int(c[1])
            if s <= e:
                mask[s : e + 1] = True
    return mask


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union for two binary arrays, defined as
    ``|a∩b| / |a∪b|``. Returns 1.0 when both masks are empty (vacuously
    perfect agreement) and 0.0 when they're disjoint.
    """
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 1.0
    return inter / union


def _percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.nanpercentile(values, q))


# ---------------------------------------------------------------------------
# Per-fly comparison
# ---------------------------------------------------------------------------


def compare_one_fly(fly_dir: Path) -> dict:
    """Run both preprocessing methods on ``fly_dir`` and return a row dict."""
    # Fresh Fly for the affine-fit (new) path.
    fly_new = Fly(fly_dir, as_individual=True)
    if fly_new.tracking_data is None or fly_new.tracking_data.skeletontrack is None:
        return {"fly": fly_dir.name, "error": "no_skeleton_track"}
    sm_new = SkeletonMetrics(fly_new)
    ball_new = sm_new.ball.objects[0].dataset

    # Fresh Fly for the legacy (geometry-based) path. Note: building a
    # second Fly re-loads the SLEAP tracks from disk — tolerable for a
    # one-off compare-N-flies smoke test, and avoids any state leakage
    # between runs.
    fly_legacy = Fly(fly_dir, as_individual=True)
    sm_legacy = LegacySkeletonMetrics(fly_legacy)
    ball_legacy = sm_legacy.ball.objects[0].dataset

    # Per-frame pixel drift between the two preprocessed ball trajectories.
    n = min(len(ball_new), len(ball_legacy))
    dx = (ball_new["x_centre_preprocessed"].to_numpy(dtype=float)[:n]
          - ball_legacy["x_centre_preprocessed"].to_numpy(dtype=float)[:n])
    dy = (ball_new["y_centre_preprocessed"].to_numpy(dtype=float)[:n]
          - ball_legacy["y_centre_preprocessed"].to_numpy(dtype=float)[:n])
    pix_dist = np.sqrt(dx * dx + dy * dy)
    pix_dist = pix_dist[~np.isnan(pix_dist)]

    # Contact masks at the union of frame counts.
    n_frames = max(
        int(ball_new.index.max()) + 1 if len(ball_new) else 0,
        int(ball_legacy.index.max()) + 1 if len(ball_legacy) else 0,
    )
    mask_new = _contact_mask(sm_new.contacts, n_frames)
    mask_legacy = _contact_mask(sm_legacy.contacts, n_frames)

    # Recovered fit coefficients (None for the legacy run).
    fit = sm_new._raw_to_template_cache
    sx, ox, sy, oy = (fit[0], fit[1], fit[2], fit[3]) if fit else (np.nan,) * 4

    return {
        "fly": fly_new.metadata.name,
        "n_frames": n,
        "n_contacts_new": len(sm_new.contacts or []),
        "n_contacts_legacy": len(sm_legacy.contacts or []),
        "contact_iou": round(_iou(mask_new, mask_legacy), 4),
        "contact_frames_new": int(mask_new.sum()),
        "contact_frames_legacy": int(mask_legacy.sum()),
        "ball_pix_diff_mean": round(float(np.nanmean(pix_dist)) if pix_dist.size else float("nan"), 3),
        "ball_pix_diff_p95": round(_percentile(pix_dist, 95), 3),
        "ball_pix_diff_max": round(float(np.nanmax(pix_dist)) if pix_dist.size else float("nan"), 3),
        "fit_sx": round(sx, 4),
        "fit_ox": round(ox, 2),
        "fit_sy": round(sy, 4),
        "fit_oy": round(oy, 2),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--yaml", help="YAML file with a 'directories:' key listing fly dirs.")
    p.add_argument(
        "dirs",
        nargs="*",
        help="Fly directories. Mutually exclusive with --yaml.",
    )
    p.add_argument(
        "--max-flies",
        type=int,
        default=10,
        help="Cap the comparison at the first N flies (default: 10).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional CSV path to write the per-fly comparison table.",
    )
    args = p.parse_args(argv)

    paths = _load_fly_paths(args)
    if not paths:
        print("No fly directories provided. Pass --yaml or positional dirs.", file=sys.stderr)
        return 1
    paths = paths[: args.max_flies]
    print(f"Comparing {len(paths)} flies\n")

    rows = []
    for path in paths:
        print(f"--- {path}")
        try:
            row = compare_one_fly(path)
        except Exception:
            print(f"  ERROR: {traceback.format_exc()}")
            continue
        rows.append(row)
        if "error" in row:
            print(f"  skipped: {row['error']}")
            continue
        print(
            f"  contacts: new={row['n_contacts_new']}, legacy={row['n_contacts_legacy']},"
            f" IoU={row['contact_iou']:.3f}"
        )
        print(
            f"  ball pix diff: mean={row['ball_pix_diff_mean']:.3f},"
            f" p95={row['ball_pix_diff_p95']:.3f},"
            f" max={row['ball_pix_diff_max']:.3f}"
        )
        print(
            f"  fit: x = {row['fit_sx']:.4f}·raw + {row['fit_ox']:.2f},"
            f" y = {row['fit_sy']:.4f}·raw + {row['fit_oy']:.2f}"
        )

    df = pd.DataFrame([r for r in rows if "error" not in r])
    if df.empty:
        print("\nNo flies produced a comparable result.", file=sys.stderr)
        return 1

    print()
    print(df.to_string(index=False))
    print()
    print("Aggregate (across flies):")
    print(f"  median contact IoU:        {df['contact_iou'].median():.4f}")
    print(f"  worst-case contact IoU:    {df['contact_iou'].min():.4f}")
    print(f"  mean ball pixel drift:     {df['ball_pix_diff_mean'].mean():.3f}")
    print(f"  worst-case max drift:      {df['ball_pix_diff_max'].max():.3f}")
    print(f"  contacts new vs legacy:    {df['n_contacts_new'].sum()} vs {df['n_contacts_legacy'].sum()}")

    # Quick-pass summary the user can eyeball: contact IoU >= 0.95
    # everywhere is a strong signal that the affine fit is faithful.
    bad = df[df["contact_iou"] < 0.95]
    if len(bad):
        print(
            f"\n⚠ {len(bad)}/{len(df)} flies have contact IoU < 0.95 — "
            "the affine fit may be missing a non-uniform scaling component."
        )
        print(bad[["fly", "contact_iou", "n_contacts_new", "n_contacts_legacy",
                   "ball_pix_diff_mean", "ball_pix_diff_max"]].to_string(index=False))
    else:
        print(f"\n✓ All {len(df)} flies have contact IoU ≥ 0.95.")

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nWrote {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
