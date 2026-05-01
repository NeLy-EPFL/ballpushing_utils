import typing
from collections.abc import Callable

import numpy as np
import polars as pl
import scipy.stats

from .config import CORRIDOR_END, MAJOR_EVENT_DISTANCE, SIGNIFICANT_EVENT_DISTANCE


def pad_list_column_to_length(
    data: pl.DataFrame, col: str, value: typing.Any, length: int
):
    return data.with_columns(
        pl.concat_list(
            col, pl.lit(value).repeat_by(length - pl.col(col).list.len().cast(pl.Int32))
        )
    )


def pad_list_column_to_max_length(data: pl.DataFrame, col: str, value: typing.Any):
    max_length = data.select(pl.col(col).list.len()).max().item()
    return data.with_columns(
        pl.concat_list(
            col,
            pl.lit(value).repeat_by(max_length - pl.col(col).list.len().cast(pl.Int32)),
        )
    )


def timeseries_confidence_intervals(
    data: np.ndarray,
    *,
    statistic: Callable = np.nanmean,
):
    """Get the confidence interval for the given set of timeseries data

    Args:
        data (np.ndarray): data with shape [NUM_SAMPLES, NUM_TIMESTEMPS]
        statistic (Callable, optional): What statistic to take the confidence interval of the data for. Defaults to np.nanmean.

    Returns:
        tuple[np.ndarray,np.ndarray]: low and high confidence intervals of length `NUM_TIMESTEPS`
    """
    bootstrap_samples = scipy.stats.bootstrap(
        (data,), statistic=statistic, n_resamples=1000, rng=42
    )
    return (
        bootstrap_samples.confidence_interval.low,
        bootstrap_samples.confidence_interval.high,
    )


# dataframe event selectors
until_ball_at_end = (
    pl.col("start_index")
    <= pl.col("start_index").filter(pl.col("ball_pos_end") >= CORRIDOR_END).first()
)

# major pushes
until_major_push = (
    pl.col("start_index")
    <= pl.col("start_index")
    .filter(pl.col("net_ball_dist") >= MAJOR_EVENT_DISTANCE)
    .first()
)
before_major_push = (
    pl.col("start_index")
    < pl.col("start_index")
    .filter(pl.col("net_ball_dist") >= MAJOR_EVENT_DISTANCE)
    .first()
)
first_major_push = (
    pl.col("start_index")
    == pl.col("start_index")
    .filter(pl.col("net_ball_dist") >= MAJOR_EVENT_DISTANCE)
    .first()
)
after_major_push = (
    pl.col("start_index")
    > pl.col("start_index")
    .filter(pl.col("net_ball_dist") >= MAJOR_EVENT_DISTANCE)
    .first()
)
major_pushes = pl.col("net_ball_dist") >= MAJOR_EVENT_DISTANCE

# major pulls
until_major_pull = (
    pl.col("start_index")
    <= pl.col("start_index")
    .filter(pl.col("net_ball_dist") <= -MAJOR_EVENT_DISTANCE)
    .first()
)
before_major_pull = (
    pl.col("start_index")
    < pl.col("start_index")
    .filter(pl.col("net_ball_dist") <= -MAJOR_EVENT_DISTANCE)
    .first()
)
first_major_pull = (
    pl.col("start_index")
    == pl.col("start_index")
    .filter(pl.col("net_ball_dist") <= -MAJOR_EVENT_DISTANCE)
    .first()
)
major_pulls = pl.col("net_ball_dist") <= -MAJOR_EVENT_DISTANCE

# major dist
until_major_dist = (
    pl.col("start_index")
    <= pl.col("start_index")
    .filter(
        pl.col("ball_pos_end") - pl.col("ball_pos_start").first()
        >= MAJOR_EVENT_DISTANCE
    )
    .first()
)

# significant pushes
until_significant_push = (
    pl.col("start_index")
    <= pl.col("start_index")
    .filter(pl.col("net_ball_dist") >= SIGNIFICANT_EVENT_DISTANCE)
    .first()
)
before_significant_push = (
    pl.col("start_index")
    < pl.col("start_index")
    .filter(pl.col("net_ball_dist") >= SIGNIFICANT_EVENT_DISTANCE)
    .first()
)
first_significant_push = (
    pl.col("start_index")
    == pl.col("start_index")
    .filter(pl.col("net_ball_dist") >= SIGNIFICANT_EVENT_DISTANCE)
    .first()
)
after_significant_push = (
    pl.col("start_index")
    > pl.col("start_index")
    .filter(pl.col("net_ball_dist") >= SIGNIFICANT_EVENT_DISTANCE)
    .first()
)
significant_pushes = pl.col("net_ball_dist") >= SIGNIFICANT_EVENT_DISTANCE

# significant pulls
until_significant_pull = (
    pl.col("start_index")
    <= pl.col("start_index")
    .filter(pl.col("net_ball_dist") <= -SIGNIFICANT_EVENT_DISTANCE)
    .first()
)
before_significant_pull = (
    pl.col("start_index")
    < pl.col("start_index")
    .filter(pl.col("net_ball_dist") <= -SIGNIFICANT_EVENT_DISTANCE)
    .first()
)
first_significant_pull = (
    pl.col("start_index")
    == pl.col("start_index")
    .filter(pl.col("net_ball_dist") <= -SIGNIFICANT_EVENT_DISTANCE)
    .first()
)
after_significant_pull = (
    pl.col("start_index")
    > pl.col("start_index")
    .filter(pl.col("net_ball_dist") <= -SIGNIFICANT_EVENT_DISTANCE)
    .first()
)
significant_pulls = pl.col("net_ball_dist") <= -SIGNIFICANT_EVENT_DISTANCE
