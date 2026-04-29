"""Colour palettes used across paper figures.

Colours are defined centrally so that (i) the same experimental condition
always renders with the same hex code and (ii) any palette change
propagates to every figure without an error-prone find-and-replace.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_TWO_GROUP_COLORS",
    "DEFAULT_TWO_GROUP_LABEL_COLORS",
    "CONDITION_COLORS",
    "BRAIN_REGION_COLORS",
]

#: Marker colours for the classic control-vs-experimental pair used in
#: MagnetBlock, F1, and related Fig. 2 panels. ``orange, blue``.
DEFAULT_TWO_GROUP_COLORS: tuple[str, str] = ("#FAA41A", "#3953A4")

#: Slightly-darker variants used for axis tick label colouring so that text
#: on a white background stays legible.
DEFAULT_TWO_GROUP_LABEL_COLORS: tuple[str, str] = ("#cb7c29", "#3953A4")

#: Hex colours keyed by the short condition name used in YAML configs and
#: summary feathers. Extend cautiously — prefer adding an alias to changing
#: an existing hex.
CONDITION_COLORS: dict[str, str] = {
    "control": "#7f7f7f",
    "experimental": "#3953A4",
    "no_ball": "#FAA41A",
    "immobile_ball": "#3953A4",
    "dark": "#1f1f1f",
    "light": "#FAA41A",
    "fed": "#2ca02c",
    "starved": "#d62728",
}

#: Hex colours for neural silencing screen brain regions. Mirrors the
#: values hand-authored in ``src/Plotting/Config.py`` but centralises them
#: so downstream scripts import from one place.
BRAIN_REGION_COLORS: dict[str, str] = {
    "MB": "#1f77b4",
    "Vision": "#ff7f0e",
    "LH": "#2ca02c",
    "Neuropeptide": "#d62728",
    "Olfaction": "#9467bd",
    "MB extrinsic neurons": "#8c564b",
    "CX": "#e377c2",
    "Control": "#7f7f7f",
    "None": "#bcbd22",
    "fchON": "#17becf",
    "JON": "#ffbb78",
    "DN": "#c5b0d5",
}

FLY_COLORS = {
    "lf": "#0f7399",
    "lm": "#188bad",
    "lh": "#76bcc9",
    "rf": "#b82032",
    "rm": "#c95750",
    "rh": "#d38279",
    "head": "green",
    "abdomen": "purple",
}

def get_cluster_palette(n_clusters: int):
    import numpy as np
    import colorcet as cc
    from matplotlib.colors import ListedColormap

    return ListedColormap(cc.rainbow)(np.linspace(0, 1, n_clusters))
