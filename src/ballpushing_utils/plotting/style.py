"""Matplotlib style helpers shared by every paper figure.

All figure scripts embed vector text as editable glyphs so that Illustrator
can re-flow labels after export. :func:`set_illustrator_style` collects the
rcParams required to produce PDFs that round-trip through Illustrator
cleanly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib

__all__ = ["set_illustrator_style"]

#: Font fallback stack used when Arial is unavailable (e.g. on Linux CI
#: machines without the Microsoft core fonts). DejaVu Sans is metric-close
#: enough for rough inspection; Illustrator handles the final typesetting.
_DEFAULT_SANS_STACK: tuple[str, ...] = ("Arial", "Helvetica", "DejaVu Sans")


def set_illustrator_style(
    *,
    sans_fonts: Sequence[str] = _DEFAULT_SANS_STACK,
    font_size: float | None = None,
    line_width: float | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Apply the paper-standard matplotlib rcParams.

    Parameters
    ----------
    sans_fonts:
        Font fallback stack for ``font.sans-serif``. Defaults to Arial →
        Helvetica → DejaVu Sans.
    font_size:
        If given, overrides ``font.size`` and ``axes.labelsize``. Most paper
        figures use 5–7 pt; leave as ``None`` to honour the matplotlib
        defaults and let the caller set sizes per-artist.
    line_width:
        If given, overrides ``axes.linewidth`` (axis spine width). The
        paper figures use 0.6 pt.
    extra:
        Additional rcParams merged last, so callers can tweak without
        touching this helper.

    Notes
    -----
    Sets Type 42 (TrueType) fonts for both PDF and PS backends so that text
    remains editable in Illustrator. Also forces ``font.family`` to
    ``sans-serif`` and installs *sans_fonts* as the fallback stack.
    """
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = list(sans_fonts)
    # Vector-export friendly defaults.
    matplotlib.rcParams["svg.fonttype"] = "none"

    if font_size is not None:
        matplotlib.rcParams["font.size"] = font_size
        matplotlib.rcParams["axes.labelsize"] = font_size
    if line_width is not None:
        matplotlib.rcParams["axes.linewidth"] = line_width
    if extra:
        matplotlib.rcParams.update(extra)
