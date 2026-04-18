"""Unit tests for :mod:`ballpushing_utils.stats.permutation`.

Hermetic: no figure data, no disk I/O. The tests lock down the two
reproducibility contracts the paper figures rely on:

1. The **legacy path** (``RandomState(seed)`` + ``rng.permutation(array)``
   + ``statistic="median"`` + ``p_correction="proportion"``) matches the
   original inline test once embedded in every plot script.

2. The **screen path** (pre-seeded ``np.random.default_rng(42)`` shared
   across calls + ``statistic="mean"`` + ``p_correction="plus_one"``)
   matches the inline helper in ``figures/Fig3-Screen/fig3_f1_tnt.py``
   bit-for-bit, including when a single ``Generator`` is threaded
   through several tests so the joint RNG sequence lines up.

If either drifts, a published p-value will silently change. These
tests are the tripwire.
"""

from __future__ import annotations

import numpy as np
import pytest

from ballpushing_utils.stats import PermutationResult, permutation_test


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _legacy_inline(a: np.ndarray, b: np.ndarray, n_perm: int, seed: int = 42) -> float:
    """Reproduces the pre-refactor inline median-diff permutation test.

    Matches the code that used to live in every figure script:
    ``np.random.seed(42)`` followed by ``np.random.permutation(combined)``
    plus a ``fraction``-style p-value.
    """
    rng = np.random.RandomState(seed)
    obs = float(np.median(b)) - float(np.median(a))
    combined = np.concatenate([a, b])
    n_a = a.size
    count = 0
    for _ in range(n_perm):
        permuted = rng.permutation(combined)
        diff = float(np.median(permuted[n_a:])) - float(np.median(permuted[:n_a]))
        if abs(diff) >= abs(obs):
            count += 1
    return count / n_perm


def _screen_inline(a: np.ndarray, b: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    """Reproduces the inline helper in ``figures/Fig3-Screen/fig3_f1_tnt.py``."""
    obs = abs(a.mean() - b.mean())
    combined = np.concatenate([a, b])
    n = len(a)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(combined))
        shuffled = combined[perm]
        if abs(shuffled[:n].mean() - shuffled[n:].mean()) >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)


@pytest.fixture
def two_samples() -> tuple[np.ndarray, np.ndarray]:
    """Small, reproducible pair of samples independent of the test RNG."""
    src = np.random.RandomState(0)
    a = src.normal(0.0, 1.0, 40)
    b = src.normal(0.5, 1.0, 40)
    return a, b


# ---------------------------------------------------------------------------
# basic surface
# ---------------------------------------------------------------------------


class TestResultShape:
    def test_returns_permutation_result(self, two_samples: tuple[np.ndarray, np.ndarray]) -> None:
        a, b = two_samples
        r = permutation_test(a, b, n_permutations=500)
        assert isinstance(r, PermutationResult)
        assert r.n_a == a.size and r.n_b == b.size
        assert r.n_permutations == 500
        assert 0.0 <= r.p_value <= 1.0

    def test_observed_diff_orientation(self, two_samples: tuple[np.ndarray, np.ndarray]) -> None:
        """Observed diff is ``statistic(b) - statistic(a)``, not the reverse."""
        a, b = two_samples
        r_mean = permutation_test(a, b, statistic="mean", n_permutations=100)
        assert r_mean.observed_diff == pytest.approx(float(b.mean() - a.mean()))

    def test_rejects_unknown_statistic(self, two_samples: tuple[np.ndarray, np.ndarray]) -> None:
        a, b = two_samples
        with pytest.raises(ValueError):
            permutation_test(a, b, statistic="variance", n_permutations=10)  # type: ignore[arg-type]

    def test_rejects_unknown_p_correction(self, two_samples: tuple[np.ndarray, np.ndarray]) -> None:
        a, b = two_samples
        with pytest.raises(ValueError):
            permutation_test(a, b, n_permutations=10, p_correction="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# reproducibility: legacy RandomState / median / proportion path
# ---------------------------------------------------------------------------


class TestLegacyPathBitForBit:
    def test_matches_inline_median_permutation(
        self, two_samples: tuple[np.ndarray, np.ndarray]
    ) -> None:
        a, b = two_samples
        inline = _legacy_inline(a, b, n_perm=2000, seed=42)
        helper = permutation_test(a, b, n_permutations=2000, seed=42).p_value
        # Plain ==, not approx: the contract is byte-for-byte.
        assert helper == inline

    def test_seed_is_deterministic(self, two_samples: tuple[np.ndarray, np.ndarray]) -> None:
        a, b = two_samples
        r1 = permutation_test(a, b, n_permutations=500, seed=42)
        r2 = permutation_test(a, b, n_permutations=500, seed=42)
        assert r1.p_value == r2.p_value
        assert r1.observed_diff == r2.observed_diff

    def test_observed_diff_independent_of_seed(
        self, two_samples: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """``observed_diff`` is a function of the data only — the seed
        controls the null ensemble, not the observation."""
        a, b = two_samples
        r1 = permutation_test(a, b, n_permutations=500, seed=42)
        r2 = permutation_test(a, b, n_permutations=500, seed=7)
        assert r1.observed_diff == r2.observed_diff

    def test_no_global_rng_side_effects(
        self, two_samples: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """The helper must not mutate the global numpy RNG.

        The legacy inline code used ``np.random.seed(42)`` which *did*
        leak into global state. The refactored helper uses a local
        ``RandomState``, so the sequence of draws from the global RNG
        must be identical whether or not the helper is called between
        them.
        """
        a, b = two_samples

        # Baseline: two draws from seed 123, with nothing in between.
        np.random.seed(123)
        expected_first = np.random.rand()
        expected_second = np.random.rand()

        # Interleaved: permutation_test called between the two draws.
        np.random.seed(123)
        observed_first = np.random.rand()
        permutation_test(a, b, n_permutations=100, seed=42)
        observed_second = np.random.rand()

        assert observed_first == expected_first
        assert observed_second == expected_second


# ---------------------------------------------------------------------------
# reproducibility: screen-panel path (Generator / mean / plus_one)
# ---------------------------------------------------------------------------


class TestScreenPathBitForBit:
    def test_matches_inline_generator_mean_plus_one(
        self, two_samples: tuple[np.ndarray, np.ndarray]
    ) -> None:
        a, b = two_samples

        rng_inline = np.random.default_rng(42)
        rng_helper = np.random.default_rng(42)

        p_inline = _screen_inline(a, b, n_perm=2000, rng=rng_inline)
        p_helper = permutation_test(
            a,
            b,
            statistic="mean",
            n_permutations=2000,
            rng=rng_helper,
            p_correction="plus_one",
        ).p_value
        assert p_helper == p_inline

    def test_shared_generator_across_multiple_calls(
        self, two_samples: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """A single Generator threaded across calls must match the inline loop.

        This is exactly the shape of ``run_stats`` in the screen panel:
        one ``default_rng(42)`` created up front and then passed into
        ``permutation_test`` for each genotype in sequence.
        """
        a, b = two_samples
        rng_inline = np.random.default_rng(42)
        rng_helper = np.random.default_rng(42)

        inline_ps = [_screen_inline(a + k, b + k, n_perm=500, rng=rng_inline) for k in range(4)]
        helper_ps = [
            permutation_test(
                a + k,
                b + k,
                statistic="mean",
                n_permutations=500,
                rng=rng_helper,
                p_correction="plus_one",
            ).p_value
            for k in range(4)
        ]
        assert helper_ps == inline_ps

    def test_rng_overrides_seed(self, two_samples: tuple[np.ndarray, np.ndarray]) -> None:
        """When ``rng`` is provided, ``seed`` is ignored — the contract is
        that the same pre-constructed RNG produces the same p-value
        regardless of what ``seed`` is set to alongside it.
        """
        a, b = two_samples
        with_seed_a = permutation_test(
            a, b, statistic="mean", n_permutations=500,
            rng=np.random.default_rng(7), seed=42,
        ).p_value
        with_seed_b = permutation_test(
            a, b, statistic="mean", n_permutations=500,
            rng=np.random.default_rng(7), seed=999,
        ).p_value
        assert with_seed_a == with_seed_b

    def test_random_state_via_rng_matches_default_seed(
        self, two_samples: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Passing ``rng=RandomState(42)`` reproduces the ``seed=42`` default."""
        a, b = two_samples
        r_default = permutation_test(a, b, statistic="mean", n_permutations=500)
        r_via_rs = permutation_test(
            a,
            b,
            statistic="mean",
            n_permutations=500,
            rng=np.random.RandomState(42),
        )
        assert r_default.p_value == r_via_rs.p_value


# ---------------------------------------------------------------------------
# p_correction arithmetic
# ---------------------------------------------------------------------------


class TestPCorrection:
    def test_plus_one_never_zero(self) -> None:
        """A perfectly-separated sample pair must still give p > 0 under plus_one."""
        a = np.zeros(30)
        b = np.ones(30) * 10.0
        r = permutation_test(
            a,
            b,
            statistic="mean",
            n_permutations=500,
            p_correction="plus_one",
        )
        assert r.p_value > 0.0
        # The proportion path is expected to hit 0 on this pathological case
        r_prop = permutation_test(
            a, b, statistic="mean", n_permutations=500, p_correction="proportion"
        )
        assert r_prop.p_value == 0.0

    def test_plus_one_relationship_to_proportion(
        self, two_samples: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """``plus_one`` is ``(count + 1) / (n_perm + 1)`` by construction."""
        a, b = two_samples
        n = 500
        p_prop = permutation_test(
            a, b, statistic="mean", n_permutations=n, p_correction="proportion"
        ).p_value
        p_plus = permutation_test(
            a, b, statistic="mean", n_permutations=n, p_correction="plus_one"
        ).p_value
        count = round(p_prop * n)
        assert p_plus == (count + 1) / (n + 1)
