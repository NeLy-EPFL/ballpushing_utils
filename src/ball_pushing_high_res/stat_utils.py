from collections.abc import Callable

import numpy as np
import polars as pl
import scipy.stats


def permutation_test_df(
    data: pl.DataFrame,
    group_col: str,
    group1_val,
    group2_val,
    col_to_compare: str,
    *,
    statistic: Callable = np.nanmean,
    n_resamples: int = 10000,
):
    return permutation_test(
        data.filter(pl.col(group_col) == group1_val)
        .get_column(col_to_compare)
        .to_list(),
        data.filter(pl.col(group_col) == group2_val)
        .get_column(col_to_compare)
        .to_list(),
        statistic=statistic,
        n_resamples=n_resamples,
    )


def permutation_test(
    group1: list, group2: list, *, statistic: Callable = np.nanmean, n_resamples=10000
):
    return scipy.stats.permutation_test(
        [group1, group2],
        lambda x, y: statistic(x) - statistic(y),
        n_resamples=n_resamples,
        rng=42,
    )
