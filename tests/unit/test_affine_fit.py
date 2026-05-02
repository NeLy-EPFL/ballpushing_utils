"""Unit tests for :meth:`SkeletonMetrics._estimate_raw_to_template_transform`.

The thorax-anchored affine fit is the core of how
:meth:`SkeletonMetrics.preprocess_ball` brings raw ball coordinates into
skeleton-template space without needing the original video dimensions.
This file fakes the minimum-surface inputs (matched-frame thorax tracks
in the fly tracker + skeleton tracker) and asserts:

- The recovered ``(sx, ox, sy, oy)`` match a known synthetic transform
  to sub-pixel precision.
- The fit returns ``None`` when one of the tracks is missing, when the
  thorax keypoint isn't present, or when fewer than 50 matched
  non-NaN frames are available — so :meth:`preprocess_ball` falls
  through to the geometry-based legacy path cleanly.
- The result is cached on the instance.

No real fly, no SLEAP, no h5py — just stubs.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ballpushing_utils.skeleton_metrics import SkeletonMetrics


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


def _make_stub(raw_thorax, tmpl_thorax, debugging=False):
    """Build the minimum SimpleNamespace surface ``_estimate_raw_to_template_transform`` reads."""
    raw_df = pd.DataFrame({
        "x_thorax": raw_thorax[:, 0],
        "y_thorax": raw_thorax[:, 1],
    })
    skel_df = pd.DataFrame({
        "x_Thorax": tmpl_thorax[:, 0],
        "y_Thorax": tmpl_thorax[:, 1],
    })

    raw_obj = SimpleNamespace(dataset=raw_df)
    skel_obj = SimpleNamespace(dataset=skel_df)
    return SimpleNamespace(
        fly=SimpleNamespace(
            config=SimpleNamespace(debugging=debugging),
            tracking_data=SimpleNamespace(
                raw_flytrack=SimpleNamespace(objects=[raw_obj]),
                skeletontrack=SimpleNamespace(objects=[skel_obj]),
            ),
        )
    )


def _apply_known_affine(raw_xy, sx, ox, sy, oy):
    """Apply the same transform :meth:`preprocess_ball` will recover."""
    out = np.empty_like(raw_xy, dtype=float)
    out[:, 0] = sx * raw_xy[:, 0] + ox
    out[:, 1] = sy * raw_xy[:, 1] + oy
    return out


# ---------------------------------------------------------------------------
# Happy path: recover a known affine from matched thorax tracks.
# ---------------------------------------------------------------------------


def test_affine_fit_recovers_known_coefficients_subpixel():
    """A perfectly matched pair of tracks recovers the input affine."""
    rng = np.random.default_rng(42)
    raw = rng.uniform(0, 600, size=(500, 2))

    sx, ox, sy, oy = 0.74, 19.5, 0.86, -74.0
    tmpl = _apply_known_affine(raw, sx, ox, sy, oy)

    stub = _make_stub(raw, tmpl)
    result = SkeletonMetrics._estimate_raw_to_template_transform(stub)
    assert result is not None
    fit_sx, fit_ox, fit_sy, fit_oy, n_used = result
    assert n_used == 500
    # The fit is least-squares on perfectly clean inputs → should recover
    # the coefficients exactly modulo float rounding.
    assert fit_sx == pytest.approx(sx, abs=1e-9)
    assert fit_sy == pytest.approx(sy, abs=1e-9)
    assert fit_ox == pytest.approx(ox, abs=1e-6)
    assert fit_oy == pytest.approx(oy, abs=1e-6)


def test_affine_fit_robust_to_nan_frames():
    """The fit drops frames with NaN in either side and still recovers
    the transform. Asserts the n_used count reflects the remaining
    overlap, not the input length.
    """
    rng = np.random.default_rng(0)
    raw = rng.uniform(0, 500, size=(300, 2))
    tmpl = _apply_known_affine(raw, 0.6, 12.0, 0.9, -30.0)

    # Inject 50 NaNs on one side, 30 on the other (90 unique = 50 + 30 - 0 overlap)
    raw[:50, 0] = np.nan
    tmpl[260:290, 1] = np.nan
    expected_valid = 300 - 50 - 30  # disjoint NaN sets

    stub = _make_stub(raw, tmpl)
    result = SkeletonMetrics._estimate_raw_to_template_transform(stub)
    assert result is not None
    sx_hat, ox_hat, sy_hat, oy_hat, n_used = result
    assert n_used == expected_valid
    assert sx_hat == pytest.approx(0.6, abs=1e-9)
    assert sy_hat == pytest.approx(0.9, abs=1e-9)


def test_affine_fit_handles_unequal_track_lengths():
    """Tracks of different lengths are intersected at the shorter length
    (this happens when one SLEAP model was applied to a manually-trimmed
    video).
    """
    rng = np.random.default_rng(1)
    raw = rng.uniform(0, 400, size=(120, 2))
    tmpl = _apply_known_affine(raw, 0.5, 10.0, 0.8, -20.0)
    # Skeleton tracker stopped 10 frames early.
    tmpl_short = tmpl[:110]

    stub = _make_stub(raw, tmpl_short)
    result = SkeletonMetrics._estimate_raw_to_template_transform(stub)
    assert result is not None
    _, _, _, _, n_used = result
    assert n_used == 110


def test_affine_fit_caches_on_instance():
    """Repeated calls return the cached tuple (or None) without re-running
    np.polyfit.
    """
    raw = np.linspace([0, 0], [600, 600], 200)
    tmpl = _apply_known_affine(raw, 0.7, 15.0, 0.85, -50.0)
    stub = _make_stub(raw, tmpl)

    first = SkeletonMetrics._estimate_raw_to_template_transform(stub)
    assert hasattr(stub, "_raw_to_template_cache")
    second = SkeletonMetrics._estimate_raw_to_template_transform(stub)
    assert first is second  # same tuple object → cache hit


# ---------------------------------------------------------------------------
# Failure modes — must return None so preprocess_ball falls through to the
# legacy geometry-based path.
# ---------------------------------------------------------------------------


def test_affine_fit_none_when_skeleton_track_missing():
    raw = np.linspace([0, 0], [600, 600], 200)
    tmpl = _apply_known_affine(raw, 0.7, 15.0, 0.85, -50.0)
    stub = _make_stub(raw, tmpl)
    stub.fly.tracking_data.skeletontrack = None

    assert SkeletonMetrics._estimate_raw_to_template_transform(stub) is None


def test_affine_fit_none_when_raw_flytrack_missing():
    raw = np.linspace([0, 0], [600, 600], 200)
    tmpl = _apply_known_affine(raw, 0.7, 15.0, 0.85, -50.0)
    stub = _make_stub(raw, tmpl)
    stub.fly.tracking_data.raw_flytrack = None

    assert SkeletonMetrics._estimate_raw_to_template_transform(stub) is None


def test_affine_fit_none_when_thorax_column_missing():
    """If the SLEAP model doesn't expose a thorax keypoint, fit isn't
    feasible → None.
    """
    raw_df = pd.DataFrame({"x_head": [1.0], "y_head": [2.0]})  # no thorax!
    skel_df = pd.DataFrame({"x_Thorax": [10.0], "y_Thorax": [20.0]})
    stub = SimpleNamespace(
        fly=SimpleNamespace(
            config=SimpleNamespace(debugging=False),
            tracking_data=SimpleNamespace(
                raw_flytrack=SimpleNamespace(objects=[SimpleNamespace(dataset=raw_df)]),
                skeletontrack=SimpleNamespace(objects=[SimpleNamespace(dataset=skel_df)]),
            ),
        )
    )
    assert SkeletonMetrics._estimate_raw_to_template_transform(stub) is None


def test_affine_fit_none_when_too_few_overlapping_frames():
    """Fewer than 50 matched non-NaN frames → return None (the threshold is
    deliberately generous; even 10 would suffice for a well-conditioned
    linear fit, but we want headroom against outliers / SLEAP misses).
    """
    rng = np.random.default_rng(99)
    raw = rng.uniform(0, 500, size=(100, 2))
    tmpl = _apply_known_affine(raw, 0.5, 10.0, 0.8, -20.0)
    # Wipe out 60 frames on the raw side → 40 valid frames remain (< 50).
    raw[:60] = np.nan

    stub = _make_stub(raw, tmpl)
    assert SkeletonMetrics._estimate_raw_to_template_transform(stub) is None
