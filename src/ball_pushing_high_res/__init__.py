"""High-resolution ball-pushing analysis package.

Original author: Dominic Dall'Osto (@Dominic-DallOsto,
dominic.dallosto@epfl.ch). Polars + plotnine pipeline that classifies
early contact events from the high-resolution recordings and produces
the ED Fig. 1 panel of Durrieu et al. (2026).

Sibling top-level package to ``ballpushing_utils``; ships in the same
wheel via
``[tool.setuptools.packages.find] include = [..., "ball_pushing_high_res*"]``.

Public submodules:

- ``config``:      paradigm constants (``CORRIDOR_END``,
                   ``MAJOR_EVENT_DISTANCE``, ``SIGNIFICANT_EVENT_DISTANCE``).
- ``df_utils``:    polars helpers (``pad_list_column_to_*``,
                   ``timeseries_confidence_intervals``) plus the
                   contact-event selectors used by the analysis.
- ``plot_utils``:  plotnine theme, image cropping / pose overlay,
                   boxplot-with-jitter, annotated split heatmap.
- ``stat_utils``:  thin scipy-permutation-test wrappers
                   (``permutation_test`` + ``permutation_test_df``).

The driver lives in
``figures/EDFigure1-HighRes/edfigure1_early_contact_classification.ipynb``.
"""
