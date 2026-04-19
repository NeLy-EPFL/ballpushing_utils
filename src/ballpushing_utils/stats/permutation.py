"""Permutation tests, bootstrap CIs, and effect-size helpers.

These helpers were previously inlined in every figure/plot script. The
:func:`permutation_test` implementation matches the existing median-diff
test (two-sided, with the same ``np.random.seed`` convention) so that
refactored scripts reproduce the published p-values bit-for-bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "PermutationResult",
    "permutation_test",
    "bootstrap_ci_difference",
    "cohens_d",
]

StatisticName = Literal["median", "mean"]
StatisticFn = Callable[[NDArray[np.floating]], float]
PCorrection = Literal["proportion", "plus_one"]


@dataclass(frozen=True)
class PermutationResult:
    """Outcome of a two-group permutation test.

    Attributes
    ----------
    observed_diff:
        ``statistic(group_b) - statistic(group_a)`` for the observed data.
    p_value:
        Two-sided p-value. With ``p_correction="proportion"`` (the default)
        this is ``count / n_permutations`` where ``count`` is the number of
        permutations with ``abs(perm_diff) >= abs(observed_diff)``. With
        ``p_correction="plus_one"`` it is ``(count + 1) / (n_permutations + 1)``.
    n_a, n_b:
        Sample sizes of the two groups.
    n_permutations:
        Number of permutations actually drawn.
    """

    observed_diff: float
    p_value: float
    n_a: int
    n_b: int
    n_permutations: int


def _resolve_statistic(statistic: StatisticName | StatisticFn) -> StatisticFn:
    if callable(statistic):
        return statistic
    if statistic == "median":
        return lambda arr: float(np.median(arr))
    if statistic == "mean":
        return lambda arr: float(np.mean(arr))
    raise ValueError(f"Unknown statistic: {statistic!r}")


def permutation_test(
    group_a: ArrayLike,
    group_b: ArrayLike,
    *,
    statistic: StatisticName | StatisticFn = "median",
    n_permutations: int = 10_000,
    seed: int | None = 42,
    rng: np.random.RandomState | np.random.Generator | None = None,
    p_correction: PCorrection = "proportion",
) -> PermutationResult:
    """Two-sided permutation test comparing *group_b* to *group_a*.

    The null hypothesis is that the two samples are exchangeable. The test
    statistic is ``statistic(group_b) - statistic(group_a)``; ``statistic``
    may be ``"median"``, ``"mean"``, or any callable that accepts a 1-D
    array and returns a float.

    Parameters
    ----------
    rng:
        Optional pre-constructed RNG. Accepts either ``np.random.RandomState``
        or ``np.random.Generator`` (i.e. the return value of
        ``np.random.default_rng``). If supplied, ``seed`` is ignored and the
        RNG is advanced in-place, which means the caller can share a single
        RNG across a loop of calls for reproducible joint sequences. If
        ``None``, a fresh ``RandomState(seed)`` is used, preserving the
        legacy bit-for-bit behaviour.
    p_correction:
        ``"proportion"`` (default) returns ``count / n_permutations``.
        ``"plus_one"`` returns ``(count + 1) / (n_permutations + 1)``, which
        is the Laplace-style correction used by some of the screen panels
        and guarantees a strictly positive p-value.

    Reproducibility
    ---------------
    With the default arguments uses the legacy Mersenne Twister RNG
    (``np.random.RandomState``) with ``.permutation`` so that calling
    ``permutation_test(a, b, seed=42)`` produces **identical** p-values to
    the pre-refactor figure scripts, which wrote ``np.random.seed(42)``
    followed by ``np.random.permutation``. Unlike ``np.random.seed``,
    ``RandomState`` does not mutate the global RNG, so calling this
    function has no observable side effects on other code that relies on
    the global state.

    Passing ``rng=np.random.default_rng(42)`` with ``statistic="mean"`` and
    ``p_correction="plus_one"`` reproduces the screen-panel convention used
    in ``figures/Fig3-Screen/fig3_f1_tnt.py`` bit-for-bit, including when
    a single ``Generator`` is threaded across several calls.
    """
    if p_correction not in ("proportion", "plus_one"):
        raise ValueError(
            f"p_correction must be 'proportion' or 'plus_one', got {p_correction!r}"
        )

    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    stat_fn = _resolve_statistic(statistic)

    observed = stat_fn(b) - stat_fn(a)
    combined = np.concatenate([a, b])
    n_a = a.size

    if rng is None:
        rng = np.random.RandomState(seed)

    perm_diffs = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        permuted = rng.permutation(combined)
        perm_diffs[i] = stat_fn(permuted[n_a:]) - stat_fn(permuted[:n_a])

    count = int(np.sum(np.abs(perm_diffs) >= np.abs(observed)))
    if p_correction == "plus_one":
        p_value = (count + 1) / (n_permutations + 1)
    else:
        p_value = count / n_permutations
    return PermutationResult(
        observed_diff=float(observed),
        p_value=float(p_value),
        n_a=n_a,
        n_b=b.size,
        n_permutations=n_permutations,
    )


def bootstrap_ci_difference(
    group_a: ArrayLike,
    group_b: ArrayLike,
    *,
    statistic: StatisticName | StatisticFn = "median",
    n_bootstraps: int = 10_000,
    ci: float = 0.95,
    seed: int | None = 42,
) -> tuple[float, float]:
    """Bootstrap CI for ``statistic(group_b) - statistic(group_a)``.

    Resamples each group with replacement ``n_bootstraps`` times and
    returns the ``ci``-coverage percentile CI. Defaults match the legacy
    helper embedded in the plot scripts.
    """
    if not 0 < ci < 1:
        raise ValueError("ci must be in (0, 1)")
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    stat_fn = _resolve_statistic(statistic)

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_bootstraps, dtype=float)
    n_a, n_b = a.size, b.size
    for i in range(n_bootstraps):
        sa = a[rng.integers(0, n_a, size=n_a)]
        sb = b[rng.integers(0, n_b, size=n_b)]
        diffs[i] = stat_fn(sb) - stat_fn(sa)

    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(diffs, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def cohens_d(group_a: ArrayLike, group_b: ArrayLike) -> float:
    """Cohen's *d* with pooled standard deviation (Hedges-style *n-1*).

    Returns ``(mean_b - mean_a) / pooled_sd``. ``np.nan`` is returned if
    either group has fewer than 2 samples or pooled variance is zero.
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    n_a, n_b = a.size, b.size
    if n_a < 2 or n_b < 2:
        return float("nan")
    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)
    pooled = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled == 0 or not np.isfinite(pooled):
        return float("nan")
    return float((b.mean() - a.mean()) / pooled)
