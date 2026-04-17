"""Statistics helpers shared across paper figure scripts.

Currently exposes:

* :func:`~ballpushing_utils.stats.permutation.permutation_test` — two-group
  permutation test with configurable statistic.
* :func:`~ballpushing_utils.stats.permutation.bootstrap_ci_difference` —
  bootstrap confidence interval for the difference of a statistic.
* :func:`~ballpushing_utils.stats.permutation.cohens_d` — Cohen's *d*
  effect size.
"""

from __future__ import annotations

from .permutation import (
    PermutationResult,
    bootstrap_ci_difference,
    cohens_d,
    permutation_test,
)

__all__ = [
    "PermutationResult",
    "bootstrap_ci_difference",
    "cohens_d",
    "permutation_test",
]
