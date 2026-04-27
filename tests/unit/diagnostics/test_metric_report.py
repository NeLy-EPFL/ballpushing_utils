"""Unit tests for :mod:`ballpushing_utils.diagnostics.metric_report`.

These tests exercise the builder against a stub Fly so they run without
SLEAP / video data. They lock down: (a) the long-table schema, (b) that
verdicts ('ok'/'low'/'high'/'nan') are computed from
:data:`DEFAULT_METRIC_RANGES`, (c) the summary aggregation, and (d) the
behaviour when no metrics are present.
"""

from __future__ import annotations

import math

import pytest

from ballpushing_utils.diagnostics import (
    DEFAULT_METRIC_RANGES,
    RangeCheck,
    build_metric_report,
    write_report,
)


class TestRangeCheck:
    def test_in_range_is_ok(self) -> None:
        rc = RangeCheck("x", lo=0.0, hi=1.0)
        assert rc.verdict(0.5) == "ok"
        assert rc.verdict(0.0) == "ok"
        assert rc.verdict(1.0) == "ok"

    def test_below_lo_is_low(self) -> None:
        assert RangeCheck("x", lo=0.0).verdict(-0.1) == "low"

    def test_above_hi_is_high(self) -> None:
        assert RangeCheck("x", hi=1.0).verdict(1.5) == "high"

    def test_nan_is_nan(self) -> None:
        assert RangeCheck("x", lo=0.0, hi=1.0).verdict(float("nan")) == "nan"

    def test_none_bounds_disable_check(self) -> None:
        # No bounds set → always ok unless NaN.
        assert RangeCheck("x").verdict(-1e9) == "ok"
        assert RangeCheck("x").verdict(1e9) == "ok"


class TestBuildMetricReport:
    def test_sections_present(self, stub_metrics_fly) -> None:
        report = build_metric_report(stub_metrics_fly)
        names = [s.name for s in report.sections]
        assert names == ["summary", "flagged", "all_metrics"]

    def test_long_table_schema(self, stub_metrics_fly) -> None:
        report = build_metric_report(stub_metrics_fly)
        long_table = report.sections[2].table
        assert long_table is not None
        expected = {"fly_idx", "ball_idx", "key", "metric", "value", "dtype", "verdict", "note"}
        assert expected.issubset(long_table.columns)

    def test_long_table_excludes_index_columns(self, stub_metrics_fly) -> None:
        report = build_metric_report(stub_metrics_fly)
        long_table = report.sections[2].table
        # ``fly_idx`` and ``ball_idx`` are index keys, not metrics — they
        # should never appear in the ``metric`` column.
        assert not long_table["metric"].isin(["fly_idx", "ball_idx", "ball_identity"]).any()

    def test_verdicts_match_default_ranges(self, stub_metrics_fly) -> None:
        report = build_metric_report(stub_metrics_fly)
        long_table = report.sections[2].table

        def _verdict(metric: str, key: str) -> str:
            row = long_table.loc[(long_table["metric"] == metric) & (long_table["key"] == key)]
            assert not row.empty, f"missing row for {metric}/{key}"
            return row.iloc[0]["verdict"]

        assert _verdict("nb_events", "fly_0_ball_0") == "ok"
        assert _verdict("chamber_ratio", "fly_0_ball_0") == "high"
        assert _verdict("median_pause_duration", "fly_0_ball_0") == "nan"
        assert _verdict("nb_events", "fly_0_ball_1") == "low"

    def test_undeclared_metric_is_listed_as_ok(self, stub_metrics_fly) -> None:
        # A metric without a RangeCheck must still appear with verdict "ok",
        # never silently dropped.
        report = build_metric_report(stub_metrics_fly)
        long_table = report.sections[2].table
        row = long_table.loc[long_table["metric"] == "made_up_metric"]
        assert len(row) == 1
        assert row.iloc[0]["verdict"] == "ok"
        assert "made_up_metric" not in DEFAULT_METRIC_RANGES

    def test_flagged_section_only_contains_non_ok(self, stub_metrics_fly) -> None:
        report = build_metric_report(stub_metrics_fly)
        flagged = report.sections[1].table
        assert flagged is not None
        assert not flagged.empty
        assert set(flagged["verdict"]).issubset({"low", "high", "nan"})

    def test_summary_counts_match_long_table(self, stub_metrics_fly) -> None:
        report = build_metric_report(stub_metrics_fly)
        summary = report.sections[0].table
        long_table = report.sections[2].table
        # Per-metric totals from the summary should match the long table
        # row counts (sanity check on the aggregation).
        for metric in summary["metric"]:
            n_observed = int(summary.loc[summary["metric"] == metric, "n_observed"].iloc[0])
            assert n_observed == int((long_table["metric"] == metric).sum())

    def test_metadata_counts(self, stub_metrics_fly) -> None:
        report = build_metric_report(stub_metrics_fly)
        long_table = report.sections[2].table
        flagged = report.sections[1].table
        assert report.metadata["n_metric_rows"] == len(long_table)
        assert report.metadata["n_flagged"] == len(flagged)

    def test_empty_summaries_degrades_gracefully(self, stub_empty_metrics_fly) -> None:
        report = build_metric_report(stub_empty_metrics_fly)
        # The flagged section should be empty and carry the "nothing flagged"
        # note, never raise.
        flagged_section = report.sections[1]
        assert flagged_section.table is not None and flagged_section.table.empty
        assert any("Nothing flagged" in n for n in flagged_section.notes)


class TestWriteReportIntegration:
    def test_writes_csvs_and_summary(self, stub_metrics_fly, tmp_path) -> None:
        report = build_metric_report(stub_metrics_fly)
        summary_path = write_report(report, tmp_path)
        assert summary_path.name == "summary.md"
        assert summary_path.exists()
        for stem in ("summary", "flagged", "all_metrics"):
            assert (tmp_path / f"{stem}.csv").exists(), f"missing {stem}.csv"
        body = summary_path.read_text()
        # The Markdown body should reference each section.
        assert "summary" in body
        assert "flagged" in body
        assert "all_metrics" in body
