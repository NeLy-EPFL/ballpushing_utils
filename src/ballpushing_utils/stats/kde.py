"""FFT-based kernel density estimation for UMAP embedding maps.

Original author: Tommy Lam (@tkclam). Used by
``ballpushing_utils.umap.analysis`` to render per-genotype density
overlays on the cluster map.
"""

import numpy as np


def get_kde(points, n_bins=512, bound=None, bw=0.1, border_rel=0.1):
    """Estimate pdf using FFT KDE.

    Parameters
    ----------
    points : array_like
        Datapoints to estimate from, with shape (# of samples, 2).
    n_bins : int
        Number of bins for each dimension.
    bound : float, optional
        Upper bound of the absolute values of the data.
        Will be calculated as max(abs(points)) * (1 + border_rel) if not provided.
    bw : float
        The bandwidth.
    border_rel : float, optional
        See description for bound.

    Returns
    -------
    ndarray
        Estimated 2D pdf.
    """
    from KDEpy.FFTKDE import FFTKDE

    if bound is None:
        bound = np.abs(points).max()
        if border_rel > 0:
            border = bound * border_rel
        else:
            border = 1e-7
        bound += border

    points = points[np.abs(points).max(1) < bound]
    grid = np.mgrid[-bound : bound : n_bins * 1j, -bound : bound : n_bins * 1j]
    grid = grid.reshape((2, -1)).T
    pdf = FFTKDE(bw=bw).fit(points).evaluate(grid)
    return pdf.reshape((n_bins, n_bins), order="F"), bound
