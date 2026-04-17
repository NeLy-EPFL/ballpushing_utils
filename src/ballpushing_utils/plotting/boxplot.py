"""Styled boxplot + jittered scatter overlay used throughout the paper.

The Figure 2 panels (and many supplementary plots) all draw the same
two-group summary: unfilled box with median, jittered individual
observations colour-coded per group, and a significance bar above. This
module collects that recipe into :func:`paired_boxplot` and the
significance-aware convenience wrapper
:func:`paired_boxplot_with_significance`.

Axis annotation (ticks, labels, spines) is deliberately **not** handled
here — scripts typically want fine control over y-axis limits and label
text. The helpers return the created artists so callers can style on top.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from .palette import DEFAULT_TWO_GROUP_COLORS
from .significance import draw_significance_bar

__all__ = [
    "BoxplotArtists",
    "paired_boxplot",
    "paired_boxplot_with_significance",
]


class BoxplotArtists(NamedTuple):
    """Return bundle for :func:`paired_boxplot`."""

    boxplot: dict
    scatter: list


def paired_boxplot(
    ax: matplotlib.axes.Axes,
    data: Sequence[ArrayLike],
    *,
    positions: Sequence[float] | None = None,
    colors: Sequence[str] = DEFAULT_TWO_GROUP_COLORS,
    box_width: float = 0.45,
    scatter_size: float = 3.0,
    scatter_jitter: float = 0.06,
    scatter_seed: int | None = 99,
    line_width: float = 0.6,
    median_line_width: float = 0.8,
) -> BoxplotArtists:
    """Draw the paper-standard unfilled boxplot + jittered scatter.

    Parameters
    ----------
    ax:
        Axes to draw into.
    data:
        Sequence of 1-D arrays, one per group.
    positions:
        X positions for each group. Defaults to ``[0, 1, 2, ...]``.
    colors:
        Scatter colour per group. Cycled if fewer colours than groups.
    box_width:
        Width of each boxplot (in data units).
    scatter_size, scatter_jitter, scatter_seed:
        Size of each scatter point, standard deviation of the jitter,
        and RNG seed (``None`` for nondeterministic).
    line_width, median_line_width:
        Line widths for the box outline/whiskers and the median stroke.
    """
    n_groups = len(data)
    if positions is None:
        positions = list(range(n_groups))
    assert len(positions) == n_groups, "positions must match the number of groups"

    box_kw = dict(linewidth=line_width, color="black")
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        boxprops=dict(linewidth=line_width, edgecolor="black"),
        whiskerprops=box_kw,
        capprops=box_kw,
        medianprops=dict(linewidth=median_line_width, color="black"),
        showfliers=False,
    )
    for box in bp["boxes"]:
        box.set_facecolor("none")
        box.set_edgecolor("black")

    rng = np.random.default_rng(scatter_seed)
    scatter_artists = []
    for i, (pos, y_data) in enumerate(zip(positions, data)):
        y_arr = np.asarray(y_data)
        x_pos = rng.normal(pos, scatter_jitter, size=y_arr.size)
        color = colors[i % len(colors)]
        sc = ax.scatter(
            x_pos,
            y_arr,
            s=scatter_size,
            c=color,
            alpha=1.0,
            edgecolors="none",
            zorder=3,
            clip_on=False,
        )
        scatter_artists.append(sc)

    return BoxplotArtists(boxplot=bp, scatter=scatter_artists)


def paired_boxplot_with_significance(
    ax: matplotlib.axes.Axes,
    data: Sequence[ArrayLike],
    *,
    p_value: float,
    positions: Sequence[float] | None = None,
    colors: Sequence[str] = DEFAULT_TWO_GROUP_COLORS,
    box_width: float = 0.45,
    sig_bar_pair: tuple[int, int] = (0, 1),
    **kwargs,
) -> BoxplotArtists:
    """Convenience wrapper: :func:`paired_boxplot` + a significance bar.

    ``sig_bar_pair`` selects which two ``positions`` to span. The other
    ``kwargs`` are forwarded to :func:`paired_boxplot`.
    """
    artists = paired_boxplot(
        ax,
        data,
        positions=positions,
        colors=colors,
        box_width=box_width,
        **kwargs,
    )
    if positions is None:
        positions = list(range(len(data)))
    left = positions[sig_bar_pair[0]]
    right = positions[sig_bar_pair[1]]
    draw_significance_bar(ax, left, right, p_value=p_value)
    return artists
