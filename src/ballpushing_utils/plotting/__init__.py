"""Shared plotting utilities for figure and supplementary scripts.

The modules in this subpackage factor out the boilerplate that was previously
duplicated across ~45 figure scripts:

* :mod:`ballpushing_utils.plotting.style` — Illustrator-compatible rcParams
* :mod:`ballpushing_utils.plotting.significance` — p-value → symbol mapping
  and significance-bar drawing.
* :mod:`ballpushing_utils.plotting.axes` — centimetre-based axes sizing.
* :mod:`ballpushing_utils.plotting.boxplot` — styled boxplot with jittered
  scatter overlay.
* :mod:`ballpushing_utils.plotting.palette` — colours shared by conditions
  and brain regions.

Typical figure script::

    from ballpushing_utils import figure_output_dir
    from ballpushing_utils.plotting import (
        set_illustrator_style,
        paired_boxplot_with_significance,
    )
    from ballpushing_utils.stats import permutation_test

    set_illustrator_style()
    res = permutation_test(control, test, statistic="median", n_permutations=10_000)
    paired_boxplot_with_significance(..., p_value=res.p_value, ...)
"""

from __future__ import annotations

from .axes import resize_axes_cm
from .boxplot import paired_boxplot, paired_boxplot_with_significance
from .palette import CONDITION_COLORS, DEFAULT_TWO_GROUP_COLORS
from .significance import (
    draw_significance_bar,
    format_significance,
    significance_symbol,
)
from .style import set_illustrator_style

__all__ = [
    "CONDITION_COLORS",
    "DEFAULT_TWO_GROUP_COLORS",
    "draw_significance_bar",
    "format_significance",
    "paired_boxplot",
    "paired_boxplot_with_significance",
    "resize_axes_cm",
    "set_illustrator_style",
    "significance_symbol",
]
