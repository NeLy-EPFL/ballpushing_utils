"""UMAP analysis subpackage.

Original author: Tommy Lam (@tkclam). UMAP-based analysis of
contact-event kinematics for the silencing screen — feature-matrix
construction, embedding fit + clustering, KDE overlays, and the
energy-test pipeline that drives Fig. 3 and ED Fig. 5.

Originally lived at the repo root as ``umap/`` (sibling of ``src/``);
relocated here so the helpers + bundled ``data/genotype_names.csv``
ship with the wheel and are importable as
``from ballpushing_utils.umap import …``.

Public submodules:

- ``preprocess``: feature-matrix prep (polars-based) feeding into UMAP.
- ``analysis``:   embedding fitting + post-processing helpers used by
  the figure scripts under ``figures/Fig3-Screen``,
  ``figures/EDFigure5-UMAP``, ``figures/SuppInfo``, etc.
- ``utils``:      shared utility functions (KDE, rotation, clustering
  labels, cluster palette, region overlays, …).

Note: ``analysis.py`` imports the third-party ``umap`` package
(``umap-learn`` on PyPI) for the ``UMAP`` class. That import resolves
to the top-level ``umap`` package, NOT this subpackage — Python finds
top-level distributions before sub-packages, so there's no shadowing.
"""

# Re-export nothing by default; submodules are imported explicitly.
__all__: list[str] = []
