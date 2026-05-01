from tqdm import tqdm
import numpy as np
from pathlib import Path
import polars as pl
import pandas as pd

root_dir = Path("/mnt/upramdya/data/MD/Ballpushing_TNTScreen/Datasets/")
dataset_name = "250809_02_standardized_contacts_TNT_screen_Data"
data_dir = root_dir / dataset_name / "standardized_contacts"
data_paths = sorted(data_dir.glob("2*.feather"))

index_cols = ["fly", "event_id", "frame"]
keypoints = {
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
data_cols = [f"{coord}_{name}" for name in keypoints for coord in "xy"]
new_data_cols = [f"{name}_{coord}" for name in keypoints.values() for coord in "xy"]
info_cols = [
    "flypath",
    "Genotype",
    "Brain region",
    "Split",
    "event_type",
]
new_info_cols = [
    "path",
    "genotype",
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
            columns=index_cols + info_cols + data_cols,
        )
        .rename(dict(zip(info_cols + data_cols, new_info_cols + new_data_cols)))
        .filter(pl.col("event_type").eq("contact") & pl.len().over(["fly", "event_id"]).eq(frames_per_event))
        .sort(index_cols)
    )

    df_fly = df.select(pl.col(col).cast(pl.String) for col in ["fly", *new_info_cols[:-1]]).unique()

    df = df.select(*index_cols, *(pl.col(col).cast(pl.Float64) for col in new_data_cols))

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

    for col in new_data_cols:
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
    for col in new_data_cols:
        df = interp_column(df, col, frames_per_event, **relaxed_interp_kwargs)

    df = df.drop_nans().filter(pl.len().over(["fly", "event_id"]).eq(frames_per_event))

    return df, df_fly


def get_preprocessed_data(
    cache_dir: Path,
    genotype_name_csv: str | Path | None = None,
    excluded_genotypes: list[str] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if (cache_dir / "contacts.parquet").exists() and (cache_dir / "flies.parquet").exists():
        df = pl.read_parquet(cache_dir / "contacts.parquet")
        df_fly = pl.read_parquet(cache_dir / "flies.parquet")
    else:
        df, df_fly = map(pl.concat, zip(*(preprocess_data(path) for path in tqdm(data_paths))))
        df.write_parquet(cache_dir / "contacts.parquet")
        df_fly.write_parquet(cache_dir / "flies.parquet")

    if genotype_name_csv is not None:
        genotype_names = dict(zip(*pl.read_csv(genotype_name_csv).to_dict().values()))
        df_fly = df_fly.with_columns(pl.col("genotype").replace(genotype_names))

    if excluded_genotypes:
        df_fly = df_fly.filter(~pl.col("genotype").is_in(excluded_genotypes))
        df = df.filter(pl.col("fly").is_in(df_fly["fly"].implode()))

    return df, df_fly


def get_heading(body_coords: np.ndarray) -> np.ndarray:
    """Calculate the heading vector from the coordinates of the nodes by
    fitting a line through the neck, thorax, and abdomen nodes.

    Parameters
    ----------
    body_coords : np.ndarray
        Array of shape (..., 3, 2) containing the coordinates of the anterior,
        medial, and posterior nodes.

    Returns
    -------
    np.ndarray
        Array of shape (..., 2) containing the heading vectors.
    """
    cov = np.einsum(
        "...pi,...pj->...ij",
        *(body_coords - body_coords.mean(axis=-2, keepdims=True),) * 2,
    )
    heading = np.linalg.eigh(cov)[1][..., -1]
    dot_prod = np.einsum("...i,...i->...", body_coords[..., 0, :] - body_coords[..., -1, :], heading)
    invert = dot_prod < 0
    heading[invert] = -heading[invert]
    return heading


def get_features(df: pl.DataFrame, frames_per_event=120) -> pl.DataFrame:
    heading = get_heading(
        df.select(f"{i}_{c}" for i in ["head", "thorax", "abdomen"] for c in "xy").to_numpy().reshape((-1, 3, 2))
    )
    front_leg_rel_pos = df.select((pl.col(f"lf_{c}") + pl.col(f"rf_{c}")) / 2 - pl.col(f"thorax_{c}") for c in "xy")
    front_leg_ap_pos = np.einsum("...i,...i->...", front_leg_rel_pos, heading)
    df_features = df.select(
        -pl.col("ball_y").alias("Ball horizontal\ndisplacement\n(px)"),
        (pl.col("thorax_y") - pl.col("ball_y")).alias("Ball-fly\nhorizontal\ndistance (px)"),
        pl.Series("Front leg\nhorizontal\nposition (px)", front_leg_ap_pos),
        pl.Series("Heading/\ninclination\nangle (°)", np.angle(heading @ (1j, -1), deg=True)),
    )
    ball_onset_position = df_features["Ball horizontal\ndisplacement\n(px)"][frames_per_event // 2 :: frames_per_event]
    df_features = df_features.with_columns(
        (pl.col("Ball horizontal\ndisplacement\n(px)") - np.repeat(ball_onset_position, frames_per_event))
    )
    return df_features
