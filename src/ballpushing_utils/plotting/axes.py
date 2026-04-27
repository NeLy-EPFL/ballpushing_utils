"""Helpers for sizing axes in physical units (centimetres).

The paper figures specify exact axes dimensions in centimetres so that
panels assemble predictably in Illustrator. This module implements the
well-tested recipe that was copy-pasted through every figure script: call
:func:`resize_axes_cm` **after** ``tight_layout`` + all padding adjustments
so that the layout engine does not overwrite the forced dimensions.
"""

from __future__ import annotations

import matplotlib.axes
import matplotlib.figure

__all__ = ["resize_axes_cm", "CM_PER_INCH"]

#: Conversion factor from centimetres to inches used by matplotlib.
CM_PER_INCH: float = 2.54


def resize_axes_cm(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    width_cm: float,
    height_cm: float,
) -> None:
    """Resize *fig* so that *ax* is exactly *width_cm* × *height_cm*.

    Parameters
    ----------
    fig:
        Figure containing *ax*.
    ax:
        Target axes.
    width_cm, height_cm:
        Desired axes dimensions, in centimetres.

    Notes
    -----
    This relies on ``ax.get_position()`` returning the layout-finalised
    fractional position of the axes within the figure. Call this **after**
    ``plt.tight_layout()`` (or equivalent) so the fractions reflect the
    final layout. After resizing, the figure size is adjusted but the
    fractional axes position is preserved.
    """
    ax_pos = ax.get_position()
    new_fig_w_in = (width_cm / CM_PER_INCH) / ax_pos.width
    new_fig_h_in = (height_cm / CM_PER_INCH) / ax_pos.height
    fig.set_size_inches(new_fig_w_in, new_fig_h_in)
