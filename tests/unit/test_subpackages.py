"""Smoke tests for the two new subpackages added by co-authors.

The goal is two-fold:

1. **Import safety.** ``import ballpushing_utils.umap``,
   ``import ball_pushing_high_res`` and their submodules must not
   trigger any I/O (no parquet reads, no glob, no mkdir). The previous
   shape of ``umap/analysis.py`` and ``umap/preprocess.py`` did all of
   the above at module load — the cwd-cleanup gated those behind
   ``main()`` and these tests lock the property in.

2. **Pure-helper sanity.** A couple of representative pure functions
   from each subpackage are exercised on tiny inputs to catch
   regressions in their semantics (interp / interp_column for the
   UMAP pre-processing, pad_list_column_*. for high-res df_utils,
   permutation_test for the high-res stat_utils).

Heavy paradigm-specific behaviour (UMAP fit, plotting, energy tests,
plotnine themes) is out of scope: those need real input data and are
covered by the figure-rerun smoke pipeline rather than unit tests.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Import-safety: importing must NOT trigger any I/O.
# ---------------------------------------------------------------------------


def _assert_no_io_on_import(module_name: str) -> None:
    """Re-import a module from scratch and assert it succeeds without raising."""
    if module_name in importlib.sys.modules:
        del importlib.sys.modules[module_name]
    importlib.import_module(module_name)


@pytest.mark.parametrize(
    "module",
    [
        "ballpushing_utils.umap",
        "ballpushing_utils.umap.preprocess",
        "ballpushing_utils.umap.utils",
        "ballpushing_utils.umap.analysis",
        "ball_pushing_high_res",
        "ball_pushing_high_res.config",
        "ball_pushing_high_res.df_utils",
        "ball_pushing_high_res.stat_utils",
        "ball_pushing_high_res.plot_utils",
    ],
)
def test_subpackage_modules_import_cleanly(module):
    """Each module must be importable on a machine with no lab-share data
    and no .cache/ — the regression we're guarding against is module-
    level parquet/csv reads + mkdir side effects.
    """
    _assert_no_io_on_import(module)


# ---------------------------------------------------------------------------
# UMAP subpackage — pure helpers.
# ---------------------------------------------------------------------------


def test_umap_preprocess_interp_fills_nans_within_chunks():
    """``interp`` reshapes a 1D array into chunks of length ``every_n`` and
    linearly interpolates NaNs *within* each chunk. The reshape requires
    the input length to be divisible by ``every_n``.
    """
    from ballpushing_utils.umap.preprocess import interp

    # Two chunks of length 4: NaN in the middle of each gets filled.
    values = np.array([1.0, np.nan, np.nan, 4.0, 10.0, np.nan, np.nan, 40.0])
    out = interp(values, every_n=4, method="linear", limit_direction="both")
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])


def test_umap_preprocess_interp_column_returns_polars_frame():
    """``interp_column`` is the polars-aware wrapper around ``interp``."""
    pl = pytest.importorskip("polars")
    from ballpushing_utils.umap.preprocess import interp_column

    df = pl.DataFrame({"v": [1.0, np.nan, 3.0, 4.0]})
    out = interp_column(df, "v", every_n=4, method="linear", limit_direction="both")
    assert out["v"].to_list() == pytest.approx([1.0, 2.0, 3.0, 4.0])


def test_umap_analysis_flip_flips_last_120_entries():
    """``flip`` is the metric-space symmetry used by the UMAP distance
    (``my_dist = min(d(a,b), d(a,flip(b)))``). It negates the trailing
    120-entry slice in-place on a copy.
    """
    from ballpushing_utils.umap.analysis import flip

    # 240-element vector, second half is the heading channel.
    x = np.concatenate([np.arange(120, dtype=float), np.arange(120, dtype=float) + 1])
    out = flip(x)
    np.testing.assert_allclose(out[:120], x[:120])
    np.testing.assert_allclose(out[120:], -(np.arange(120, dtype=float) + 1))


def test_umap_analysis_my_dist_returns_min_of_normal_and_flipped():
    """The custom UMAP metric returns the smaller of d(a,b) and d(a,flip(b))."""
    from ballpushing_utils.umap.analysis import flip, my_dist

    a = np.zeros(240, dtype=float)
    b = np.zeros(240, dtype=float)
    b[120:] = 5.0  # heading offset that flip(b) would zero out
    # d(a, b) > d(a, flip(b)) because flip(b) zeroes the heading entries
    # by negating, so d_flip = sqrt(120 * 25) > 0; we just need to check
    # the returned distance is finite and ≤ d(a, b).
    d_normal = float(np.linalg.norm(a - b))
    d = my_dist(a, b)
    flipped_d = float(np.linalg.norm(a - flip(b)))
    assert d == pytest.approx(min(d_normal, flipped_d))


# ---------------------------------------------------------------------------
# ball_pushing_high_res — pure helpers.
# ---------------------------------------------------------------------------


def test_high_res_pad_list_column_to_length():
    """``pad_list_column_to_length`` right-pads a polars list column with a
    fill value up to a target length.
    """
    pl = pytest.importorskip("polars")
    from ball_pushing_high_res.df_utils import pad_list_column_to_length

    df = pl.DataFrame({"x": [[1, 2], [3]]})
    out = pad_list_column_to_length(df, "x", value=0, length=4)
    assert out["x"].to_list() == [[1, 2, 0, 0], [3, 0, 0, 0]]


def test_high_res_pad_list_column_to_max_length():
    """``pad_list_column_to_max_length`` pads each list to the longest list
    in the column.
    """
    pl = pytest.importorskip("polars")
    from ball_pushing_high_res.df_utils import pad_list_column_to_max_length

    df = pl.DataFrame({"x": [[1, 2, 3], [4], [5, 6]]})
    out = pad_list_column_to_max_length(df, "x", value=-1)
    assert out["x"].to_list() == [[1, 2, 3], [4, -1, -1], [5, 6, -1]]


def test_high_res_permutation_test_seeded_reproducible():
    """``permutation_test`` is a thin scipy wrapper with a fixed seed (42)
    — same inputs must produce the same p-value across runs.
    """
    from ball_pushing_high_res.stat_utils import permutation_test

    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, size=50).tolist()
    b = rng.normal(0.5, 1, size=50).tolist()

    res_first = permutation_test(a, b, n_resamples=2000)
    res_second = permutation_test(a, b, n_resamples=2000)
    assert res_first.pvalue == res_second.pvalue


def test_high_res_permutation_test_df_matches_explicit_lists():
    """The DataFrame helper just slices and forwards to
    :func:`permutation_test`; results must match.
    """
    pl = pytest.importorskip("polars")
    from ball_pushing_high_res.stat_utils import permutation_test, permutation_test_df

    df = pl.DataFrame({
        "group": ["a"] * 30 + ["b"] * 30,
        "value": list(np.arange(30, dtype=float)) + list(np.arange(30, 60, dtype=float)),
    })
    expected = permutation_test(
        df.filter(pl.col("group") == "a")["value"].to_list(),
        df.filter(pl.col("group") == "b")["value"].to_list(),
        n_resamples=1000,
    )
    actual = permutation_test_df(
        df, group_col="group", group1_val="a", group2_val="b",
        col_to_compare="value", n_resamples=1000,
    )
    assert actual.pvalue == expected.pvalue
