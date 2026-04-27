"""Unit tests for :mod:`ballpushing_utils.diagnostics.event_timeline`.

These tests exercise the builder against a stub Fly so they run without
SLEAP / video data. The goal is to lock down the table schema, the
mm:ss formatting, and the significance bucketing — those are the bits a
human reader of the diagnostic will inspect first.
"""

from __future__ import annotations

import math

import pytest

from ballpushing_utils.diagnostics import build_event_timeline, write_report
from ballpushing_utils.diagnostics.event_timeline import _format_mm_ss


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class TestFormatMmSs:
    def test_basic(self) -> None:
        assert _format_mm_ss(0.0) == "00:00.00"
        assert _format_mm_ss(59.5) == "00:59.50"
        assert _format_mm_ss(60.0) == "01:00.00"
        assert _format_mm_ss(125.42) == "02:05.42"

    def test_nan_returns_empty_string(self) -> None:
        assert _format_mm_ss(float("nan")) == ""

    def test_none_returns_empty_string(self) -> None:
        assert _format_mm_ss(None) == ""


# ---------------------------------------------------------------------------
# build_event_timeline
# ---------------------------------------------------------------------------


class TestBuildEventTimeline:
    def test_table_schema(self, stub_events_fly) -> None:
        report = build_event_timeline(stub_events_fly)
        events = report.sections[0]
        assert events.name == "events"
        assert events.table is not None

        expected_cols = {
            "fly_idx", "ball_idx", "event_idx",
            "start_frame", "end_frame",
            "start_time_s", "end_time_s",
            "start_mm_ss", "end_mm_ss",
            "duration_s", "ball_displacement_px",
            "classification", "is_significant", "is_major",
        }
        assert expected_cols.issubset(events.table.columns)

    def test_one_row_per_event(self, stub_events_fly) -> None:
        report = build_event_timeline(stub_events_fly)
        # Stub has 3 events on (0, 0) + 1 on (0, 1) = 4 events.
        assert len(report.sections[0].table) == 4
        assert report.metadata["n_events"] == 4

    def test_classification_buckets(self, stub_events_fly) -> None:
        # Defaults: signif_threshold=5.0 (strict >), major_threshold=20.0 (>=)
        report = build_event_timeline(stub_events_fly)
        table = report.sections[0].table
        # Match (start_frame, expected_classification) pairs.
        expected = {
            30: "minor",
            300: "significant",
            900: "major",
            600: "significant",
        }
        for start_frame, cls in expected.items():
            row = table.loc[table["start_frame"] == start_frame].iloc[0]
            assert row["classification"] == cls
            assert row["is_significant"] == (cls in ("significant", "major"))
            assert row["is_major"] == (cls == "major")

    def test_threshold_overrides_change_classification(self, stub_events_fly) -> None:
        # With major_threshold=10.0, the 12.0px event should now be major.
        report = build_event_timeline(
            stub_events_fly, signif_threshold=1.0, major_threshold=10.0,
        )
        table = report.sections[0].table
        twelve_px = table.loc[table["start_frame"] == 300].iloc[0]
        assert twelve_px["classification"] == "major"
        assert twelve_px["is_major"] is True or twelve_px["is_major"] == True  # noqa: E712

    def test_mm_ss_matches_frame_and_fps(self, stub_events_fly) -> None:
        # FPS is 30 in the stub. Frame 300 → 10.0s → "00:10.00".
        report = build_event_timeline(stub_events_fly)
        table = report.sections[0].table
        row = table.loc[table["start_frame"] == 300].iloc[0]
        assert row["start_time_s"] == pytest.approx(10.0, abs=1e-6)
        assert row["start_mm_ss"] == "00:10.00"

    def test_metadata_populated(self, stub_events_fly) -> None:
        report = build_event_timeline(stub_events_fly)
        assert report.metadata["fps"] == 30.0
        assert report.metadata["signif_threshold_px"] == 5.0
        assert report.metadata["major_threshold_px"] == 20.0
        assert report.subject == "stub_fly"

    def test_empty_events_yields_note(self, stub_empty_events_fly) -> None:
        report = build_event_timeline(stub_empty_events_fly)
        assert report.metadata["n_events"] == 0
        events_section = report.sections[0]
        assert events_section.table is not None and events_section.table.empty
        assert any("No interaction events" in n for n in events_section.notes)

    def test_timeline_figure_is_returned(self, stub_events_fly) -> None:
        report = build_event_timeline(stub_events_fly)
        timeline = report.sections[1]
        assert timeline.name == "timeline"
        assert timeline.figure is not None


# ---------------------------------------------------------------------------
# integration with write_report
# ---------------------------------------------------------------------------


class TestWriteReportIntegration:
    def test_writes_summary_csv_and_plot(self, stub_events_fly, tmp_path) -> None:
        report = build_event_timeline(stub_events_fly)
        summary_path = write_report(report, tmp_path)
        assert summary_path.name == "summary.md"
        assert summary_path.exists()
        assert (tmp_path / "events.csv").exists()
        assert (tmp_path / "plots" / "timeline.png").exists()
        # summary.md should embed the events table preview and the plot.
        body = summary_path.read_text()
        assert "events" in body
        assert "plots/timeline.png" in body
