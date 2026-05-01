"""Pre-aggregation pipeline feeding the UMAP feature matrix.

Reads per-fly ``standardized_contacts`` feathers (one per experiment),
keeps only contact events of the canonical length, normalises the
keypoint column names, fills NaNs by interpolation + opposite-leg
fallback, and writes pooled ``contacts.parquet`` / ``flies.parquet``
into the project cache directory (``ballpushing_utils.paths.get_cache_dir()``).

The reusable pieces (``interp``, ``interp_column``, ``preprocess_data``)
are importable; the driver code lives behind ``if __name__ == "__main__"``
so importing this module never touches the lab share or writes to disk.

TODO(durrieu, coauthor): the input dataset is not yet on Dataverse —
see ``preprocess_screen_data.py`` for the same TODO marker.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

# Column-name conventions for the on-server standardized_contacts feathers.
# ``keypoints`` maps the SLEAP node-name suffix used in those feathers
# (``x_centre_preprocessed``, ``x_Thorax``, …) to the short snake-case
# names the UMAP analysis uses (``ball``, ``thorax``, …).
INDEX_COLS = ["fly", "event_id", "frame"]
KEYPOINTS = {
    "centre_preprocessed": "ball",
    "Thorax": "thorax",
    "Head": "head",
    "Abdomen": "abdomen",
    "Lfront": "lf",
    "Lmid": "lm",
    "Lhind": "lh",
    "Rfront": "rf",
    "Rmid": "rm",
    "Rhind": "rh",
}
DATA_COLS = [f"{coord}_{name}" for name in KEYPOINTS for coord in "xy"]
NEW_DATA_COLS = [f"{name}_{coord}" for name in KEYPOINTS.values() for coord in "xy"]
INFO_COLS = [
    "flypath",
    "Genotype",
    "Nickname",
    "Simplified Nickname",
    "Brain region",
    "Split",
    "event_type",
]
NEW_INFO_COLS = [
    "path",
    "genotype",
    "nickname",
    "simplified_nickname",
    "brain_region",
    "split",
    "event_type",
]


# Interpolate per-event chunks of length every_n and flatten back to a 1D array.
def interp(values, every_n: int, **kwargs) -> np.ndarray:
    reshaped = np.reshape(values, (-1, every_n))
    return pd.DataFrame(reshaped).interpolate(axis=1, **kwargs).to_numpy().ravel()


def interp_column(df: pl.DataFrame, col: str, every_n: int, **kwargs) -> pl.DataFrame:
    return df.with_columns(pl.Series(col, interp(df[col], every_n, **kwargs)))


def preprocess_data(
    data_path: Path,
    frames_per_event: int = 120,
    ball_y_threshold: float = 8,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    df = (
        pl.read_ipc(
            data_path,
            memory_map=False,
            columns=INDEX_COLS + INFO_COLS + DATA_COLS,
        )
        .rename(dict(zip(INFO_COLS + DATA_COLS, NEW_INFO_COLS + NEW_DATA_COLS)))
        .filter(
            pl.col("event_type").eq("contact")
            & pl.len().over(["fly", "event_id"]).eq(frames_per_event)
        )
        .sort(INDEX_COLS)
    )

    df_fly = df.select(
        pl.col(col).cast(pl.String) for col in ["fly", *NEW_INFO_COLS[:-1]]
    ).unique()

    df = df.select(
        *INDEX_COLS, *(pl.col(col).cast(pl.Float64) for col in NEW_DATA_COLS)
    )

    strict_interp_kwargs = {
        "method": "linear",
        "limit_direction": "both",
        "limit": 30,
        "limit_area": "inside",
    }

    relaxed_interp_kwargs = {
        "method": "linear",
        "limit": 10,
    }

    y_ball = interp(df["ball_y"], frames_per_event, **strict_interp_kwargs)
    y_ball = interp(y_ball, frames_per_event, **relaxed_interp_kwargs)
    y_ball_mid = y_ball[frames_per_event // 2 :: frames_per_event]
    y_ball_end = y_ball[frames_per_event - 1 :: frames_per_event]
    diff_y_ball = np.abs(y_ball_end - y_ball_mid)
    df = df.filter(np.repeat(diff_y_ball > ball_y_threshold, frames_per_event))

    for col in NEW_DATA_COLS:
        df = interp_column(df, col, frames_per_event, **strict_interp_kwargs)

    legs = ["lf", "lm", "lh", "rf", "rm", "rh"]
    opposite_leg = {leg: {"l": "r", "r": "l"}[leg[0]] + leg[1] for leg in legs}
    df = df.with_columns(
        pl.col(f"{leg}_{coord}").fill_nan(pl.col(f"{opposite_leg[leg]}_{coord}"))
        for leg in legs
        for coord in ["x", "y"]
    )

    relaxed_interp_kwargs = {
        "method": "linear",
        "limit": 10,
    }
    for col in NEW_DATA_COLS:
        df = interp_column(df, col, frames_per_event, **relaxed_interp_kwargs)

    df = df.drop_nans().filter(pl.len().over(["fly", "event_id"]).eq(frames_per_event))

    return df, df_fly


def main() -> None:
    """Build the pooled ``contacts.parquet`` + ``flies.parquet`` cache."""
    from ballpushing_utils.paths import get_cache_dir, require_path

    # TODO(durrieu, coauthor): switch to ballpushing_utils.dataset(...) once
    # the input is on Dataverse — see preprocess_screen_data.py.
    data_dir = require_path(
        "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Datasets/"
        "250809_02_standardized_contacts_TNT_screen_Data/standardized_contacts",
        description="screen standardized_contacts feathers",
        env_var="BALLPUSHING_SCREEN_CONTACTS_DIR",
    )
    data_paths = sorted(data_dir.glob("2*.feather"))

    if not data_paths:
        raise FileNotFoundError(
            f"No standardized_contacts feathers found under {data_dir}. "
            f"Expected files matching '2*.feather' (per-experiment standardized "
            f"contacts feathers built by dataset_builder.py). "
            f"Build them first via "
            f"`python src/dataset_builder.py --datasets standardized_contacts ...`."
        )

    save_dir = get_cache_dir()  # <repo>/.cache/

    df, df_fly = zip(*(preprocess_data(path) for path in tqdm(data_paths)))
    pl.concat(df).write_parquet(save_dir / "contacts.parquet")
    pl.concat(df_fly).write_parquet(save_dir / "flies.parquet")


if __name__ == "__main__":
    main()
