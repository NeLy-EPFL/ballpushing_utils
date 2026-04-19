"""Unit tests for :mod:`ballpushing_utils.diagnostics.report`.

Tests :func:`write_report` round-trips DiagnosticReport / DiagnosticSection
to disk correctly, and that the writer never raises on the empty-section
edge cases that diagnostics rely on.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless for CI

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from ballpushing_utils.diagnostics import (
    DiagnosticReport,
    DiagnosticSection,
    write_report,
)


def _toy_report() -> DiagnosticReport:
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 4])

    table = pd.DataFrame({"metric": ["a", "b"], "value": [1, 2]})

    report = DiagnosticReport(
        title="Toy",
        subject="toy_subject",
        metadata={"k": 1},
    )
    report.add(DiagnosticSection(name="numbers", description="A table.", table=table))
    report.add(DiagnosticSection(name="picture", description="A plot.", figure=fig))
    return report


def test_write_report_creates_expected_files(tmp_path) -> None:
    report = _toy_report()
    summary_path = write_report(report, tmp_path)
    assert summary_path == tmp_path / "summary.md"
    assert summary_path.exists()
    assert (tmp_path / "numbers.csv").exists()
    assert (tmp_path / "plots" / "picture.png").exists()


def test_write_report_summary_inlines_table_preview(tmp_path) -> None:
    report = _toy_report()
    summary = write_report(report, tmp_path).read_text()
    assert "# Toy" in summary
    assert "toy_subject" in summary
    assert "numbers" in summary
    # Table values should appear in the rendered Markdown.
    assert "metric" in summary
    assert "value" in summary
    # And the figure should be referenced via a relative link.
    assert "plots/picture.png" in summary


def test_write_report_handles_empty_report(tmp_path) -> None:
    report = DiagnosticReport(title="Empty", subject="nothing")
    summary_path = write_report(report, tmp_path)
    body = summary_path.read_text()
    assert "# Empty" in body
    assert "nothing" in body
    # No CSVs, no plots/ folder needed.
    assert not (tmp_path / "plots").exists()


def test_write_report_handles_empty_table_section(tmp_path) -> None:
    report = DiagnosticReport(title="Edge", subject="edge_subject")
    report.add(DiagnosticSection(
        name="empty",
        description="No rows.",
        table=pd.DataFrame(columns=["a", "b"]),
        notes=["Nothing to see here."],
    ))
    summary_path = write_report(report, tmp_path)
    body = summary_path.read_text()
    # Empty CSV is still written (downstream tooling expects it to exist).
    assert (tmp_path / "empty.csv").exists()
    # The note from the section must be rendered.
    assert "Nothing to see here." in body


def test_subject_with_unsafe_chars_is_sanitised(tmp_path) -> None:
    """Section file names should never contain path separators."""
    report = DiagnosticReport(title="x", subject="y")
    report.add(DiagnosticSection(
        name="bad/name with spaces",
        table=pd.DataFrame({"v": [1]}),
    ))
    write_report(report, tmp_path)
    # Slash and space are replaced with underscores in the file stem.
    assert (tmp_path / "bad_name_with_spaces.csv").exists()
