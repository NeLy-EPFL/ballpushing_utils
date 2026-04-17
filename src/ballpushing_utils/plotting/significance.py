"""Helpers for rendering statistical significance on plots.

Every figure script in the paper uses the same three-tier p-value → star
mapping (``***`` / ``**`` / ``*`` / ``ns``). These helpers centralise that
logic and the drawing of the short horizontal bar annotating a pairwise
comparison.
"""

from __future__ import annotations

from typing import Literal

import matplotlib.axes
import matplotlib.pyplot as plt

__all__ = [
    "significance_symbol",
    "format_significance",
    "draw_significance_bar",
]


def significance_symbol(
    p_value: float,
    *,
    ns_label: str = "ns",
    thresholds: tuple[float, float, float] = (0.001, 0.01, 0.05),
) -> str:
    """Return the Asterisks-or-``ns`` symbol for *p_value*.

    The default thresholds produce the conventional

    ``p < 0.001`` → ``***``
    ``p < 0.01``  → ``**``
    ``p < 0.05``  → ``*``
    otherwise     → ``ns``

    mapping. Pass ``ns_label=""`` to omit the ``ns`` annotation entirely.
    """
    t_high, t_mid, t_low = thresholds
    if p_value < t_high:
        return "***"
    if p_value < t_mid:
        return "**"
    if p_value < t_low:
        return "*"
    return ns_label


def format_significance(p_value: float, *, digits: int = 4) -> str:
    """Format *p_value* for human-readable logs (``p = 0.0023 | **``)."""
    return f"p = {p_value:.{digits}f} | {significance_symbol(p_value)}"


def draw_significance_bar(
    ax: matplotlib.axes.Axes,
    x_left: float,
    x_right: float,
    y: float | None = None,
    *,
    p_value: float | None = None,
    symbol: str | None = None,
    bar_offset_frac: float = 0.03,
    text_offset_frac: float = 0.01,
    line_width: float = 0.6,
    font_size: float = 8.0,
    font_name: str = "Arial",
    color: str = "black",
    ha: Literal["left", "center", "right"] = "center",
) -> None:
    """Draw a pairwise-comparison significance bar + label above an axes.

    Parameters
    ----------
    ax:
        Matplotlib axes to draw into. Drawing is ``clip_on=False`` so the
        annotation is visible even if it lives outside the data limits.
    x_left, x_right:
        X positions of the two groups in data coordinates.
    y:
        Bar y position (data coordinates). If ``None``, placed
        ``bar_offset_frac`` of the current y-axis range above ``ax.ylim[1]``.
    p_value:
        P-value to map to a symbol via :func:`significance_symbol`. Ignored
        if *symbol* is given explicitly.
    symbol:
        Symbol to draw above the bar. Overrides *p_value* if both supplied.
    bar_offset_frac, text_offset_frac:
        Vertical offsets (as fraction of current y-range) for the bar and
        the label above it.
    line_width, font_size, font_name, color:
        Cosmetic knobs matching the paper style.
    ha:
        Horizontal alignment of the text label.
    """
    if symbol is None:
        if p_value is None:
            raise ValueError("Either p_value or symbol must be provided.")
        symbol = significance_symbol(p_value)

    y_lo, y_hi = ax.get_ylim()
    y_range = y_hi - y_lo
    if y is None:
        y = y_hi + bar_offset_frac * y_range

    ax.plot([x_left, x_right], [y, y], color=color, linewidth=line_width, clip_on=False)
    ax.text(
        (x_left + x_right) / 2.0,
        y + text_offset_frac * y_range,
        symbol,
        ha=ha,
        va="bottom",
        fontsize=font_size,
        fontname=font_name,
        color=color,
        clip_on=False,
    )
