"""Data structures and writer for diagnostic reports.

Each builder in this subpackage returns a :class:`DiagnosticReport`,
containing one or more :class:`DiagnosticSection` entries plus any number
of tabular and figure attachments. :func:`write_report` turns that data
structure into a per-run folder on disk::

    <output_dir>/
        summary.md       # human-readable overview
        <section>.csv    # one CSV per tabular section
        plots/<name>.png # one PNG per figure section

The writer never mutates the report in place, and never raises if the
section lists are empty — diagnostics should degrade gracefully when
nothing interesting is detected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.figure
import pandas as pd

__all__ = [
    "DiagnosticReport",
    "DiagnosticSection",
    "RangeCheck",
    "write_report",
]


@dataclass(frozen=True)
class RangeCheck:
    """Declared plausible range for a metric value.

    Used by :mod:`ballpushing_utils.diagnostics.metric_report` to flag
    metrics that fell outside what we expect. A *None* bound disables that
    side of the check (useful for non-negative metrics with no upper cap).
    """

    metric: str
    lo: float | None = None
    hi: float | None = None
    note: str = ""

    def verdict(self, value: float) -> str:
        """Return ``"ok"``, ``"low"``, ``"high"``, or ``"nan"``."""
        if value is None or (isinstance(value, float) and value != value):  # NaN
            return "nan"
        if self.lo is not None and value < self.lo:
            return "low"
        if self.hi is not None and value > self.hi:
            return "high"
        return "ok"


@dataclass
class DiagnosticSection:
    """One section of a :class:`DiagnosticReport`.

    A section is either **tabular** (``table`` populated, a CSV is written)
    or **figurative** (``figure`` populated, a PNG is written) or both.
    ``description`` is rendered as prose in the Markdown summary.
    """

    name: str
    description: str = ""
    table: pd.DataFrame | None = None
    figure: matplotlib.figure.Figure | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class DiagnosticReport:
    """Bundle of sections + metadata for one diagnostic run."""

    title: str
    subject: str  # e.g. the fly or experiment name
    sections: list[DiagnosticSection] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, object] = field(default_factory=dict)

    def add(self, section: DiagnosticSection) -> None:
        self.sections.append(section)

    def extend(self, sections: Iterable[DiagnosticSection]) -> None:
        self.sections.extend(sections)


def _sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def _render_summary(report: DiagnosticReport, out_dir: Path) -> str:
    """Return the Markdown body of the summary report."""
    lines: list[str] = []
    lines.append(f"# {report.title}")
    lines.append("")
    lines.append(f"- **Subject:** {report.subject}")
    lines.append(f"- **Generated:** {report.generated_at.isoformat(timespec='seconds')}")
    for key, value in report.metadata.items():
        lines.append(f"- **{key}:** {value}")
    lines.append("")

    for section in report.sections:
        lines.append(f"## {section.name}")
        lines.append("")
        if section.description:
            lines.append(section.description)
            lines.append("")
        if section.figure is not None:
            lines.append(f"![{section.name}](plots/{_sanitize(section.name)}.png)")
            lines.append("")
        if section.table is not None and not section.table.empty:
            preview = section.table.head(20)
            lines.append(preview.to_markdown(index=False))
            if len(section.table) > len(preview):
                lines.append(f"\n_…{len(section.table) - len(preview)} more rows in "
                             f"`{_sanitize(section.name)}.csv`._")
            lines.append("")
        for note in section.notes:
            lines.append(f"> {note}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_report(report: DiagnosticReport, output_dir: str | Path) -> Path:
    """Write *report* to *output_dir* and return the path of ``summary.md``.

    Creates *output_dir* (and a ``plots/`` subfolder if any section has a
    figure). Tabular sections are written as ``<name>.csv``, figure
    sections as ``plots/<name>.png``. The Markdown summary is written as
    ``summary.md`` and inlines a 20-row preview of every table + any
    figures via relative links.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for section in report.sections:
        stem = _sanitize(section.name)
        if section.table is not None:
            section.table.to_csv(out / f"{stem}.csv", index=False)
        if section.figure is not None:
            (out / "plots").mkdir(exist_ok=True)
            section.figure.savefig(out / "plots" / f"{stem}.png", dpi=150, bbox_inches="tight")

    summary_path = out / "summary.md"
    summary_path.write_text(_render_summary(report, out))
    return summary_path
