"""UMAP analysis driver: embedding fit + cluster maps + per-genotype kdes.

Reads the cached ``contacts.parquet`` / ``flies.parquet`` produced by
``preprocess.py``, computes a 4-channel kinematic feature matrix, fits
a UMAP embedding, clusters it, then writes the figure-3 panels and
per-genotype KDE pages into ``outputs/``.

The driver lives behind ``if __name__ == "__main__"`` so ``import
ballpushing_utils.umap.analysis`` is side-effect-free — only the pure
helpers (``get_features``, ``flip``, ``my_dist``) and constants are
exposed at module load time.
"""

from __future__ import annotations

from pathlib import Path

import numba
import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from mplex import Grid
from pynndescent.distances import euclidean
from tqdm import tqdm
from umap import UMAP  # the third-party umap-learn package, NOT this subpackage

from .utils import (
    brain_region_palette,
    cluster_points,
    determine_flip_needed,
    energy_test_fly,
    get_areas,
    get_cluster_palette,
    get_heading,
    get_kde,
    plot_map_regions,
    rotate_embedding,
    save_grid_video,
    sort_labels,
)

# --- Constants (no I/O) -----------------------------------------------------

FRAMES_PER_EVENT = 120
FPS = 29

EXCLUDED_GENOTYPES = ["Wild-type(PR)", "Wild-type(Canton-S)", "TH", "TH-2"]
CONTROL_GENOTYPES = {"y": "Empty-Split-Gal4", "n": "Empty-Gal4", "m": "TNT×PR"}

FEATURE_NAMES = [
    "Ball horizontal\ndisplacement\n(px)",
    "Ball-fly\nhorizontal\ndistance (px)",
    "Front leg\nhorizontal\nposition (px)",
    "Heading/\ninclination\nangle (°)",
]

#: Bundled CSV ships with the wheel; resolves regardless of cwd.
_DATA_DIR = Path(__file__).resolve().parent / "data"


# --- Pure helpers -----------------------------------------------------------


def get_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute the 4-channel kinematic feature matrix from per-frame coords."""
    heading = get_heading(
        df.select(f"{i}_{c}" for i in ["head", "thorax", "abdomen"] for c in "xy")
        .to_numpy()
        .reshape((-1, 3, 2))
    )
    front_leg_rel_pos = df.select(
        (pl.col(f"lf_{c}") + pl.col(f"rf_{c}")) / 2 - pl.col(f"thorax_{c}")
        for c in "xy"
    )
    front_leg_ap_pos = np.einsum("...i,...i->...", front_leg_rel_pos, heading)
    df_features = df.select(
        -pl.col("ball_y").alias("ball_horizontal_position"),
        (pl.col("thorax_y") - pl.col("ball_y")).alias("ball_fly_horizontal_distance"),
        pl.Series("front_leg_ap_pos", front_leg_ap_pos),
        pl.Series("heading", np.angle(heading @ (1j, -1), deg=True)),
    )
    ball_onset_position = df_features["ball_horizontal_position"][
        FRAMES_PER_EVENT // 2 :: FRAMES_PER_EVENT
    ]
    df_features = df_features.with_columns(
        (
            pl.col("ball_horizontal_position")
            - np.repeat(ball_onset_position, FRAMES_PER_EVENT)
        )
    )
    return df_features


@numba.njit
def flip(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    x[..., -120:] *= -1
    return x


@numba.njit
def my_dist(a: np.ndarray, b: np.ndarray) -> float:
    return min(euclidean(a, b), euclidean(a, flip(b)))


# --- Driver -----------------------------------------------------------------


def main() -> None:
    from ballpushing_utils.paths import get_cache_dir

    cache_dir = get_cache_dir()
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True, parents=True)

    t = (np.arange(FRAMES_PER_EVENT) - FRAMES_PER_EVENT // 2) / FPS

    df = pl.read_parquet(cache_dir / "contacts.parquet")
    df_fly = pl.read_parquet(cache_dir / "flies.parquet")

    genotype_names = dict(zip(*pl.read_csv(_DATA_DIR / "genotype_names.csv").to_dict().values()))
    df_fly = df_fly.with_columns(pl.col("genotype").replace(genotype_names))
    df_fly = df_fly.filter(~pl.col("genotype").is_in(EXCLUDED_GENOTYPES))
    df = df.filter(pl.col("fly").is_in(df_fly["fly"].implode()))

    df_features = get_features(df)
    n_features = df_features.shape[-1]
    feature_matrix = np.reshape(df_features, (-1, FRAMES_PER_EVENT, n_features)).transpose(
        0, 2, 1
    )
    feature_std = feature_matrix.std(axis=(0, 2), keepdims=True)
    feature_matrix = (feature_matrix / feature_std).reshape(
        (-1, n_features * FRAMES_PER_EVENT)
    )

    if Path(cache_dir / "umap_embedding.npy").exists():
        embedding = np.load(cache_dir / "umap_embedding.npy")
    else:
        umap = UMAP(n_components=2, random_state=0, metric=my_dist)
        embedding = umap.fit_transform(feature_matrix)
        np.save(cache_dir / "umap_embedding.npy", embedding)

    embedding = rotate_embedding(embedding)
    bound = np.abs(embedding).max() * 1.05
    kde_image = get_kde(embedding, bound=bound)[0]
    n_bins = len(kde_image)

    n_clusters = 20
    labels = cluster_points(embedding, n_clusters=n_clusters)
    labels = (labels - 1) % n_clusters
    labels = sort_labels(labels, embedding)

    density_threshold = 5e-5
    im_regions, xlim, ylim = get_areas(embedding, labels, bound, n_bins, density_threshold)
    xlim = xlim[::-1]

    g = plot_map_regions(im_regions, embedding, labels, xlim, ylim, bound)
    bar_length = 2.5
    ax = g.item()
    ax.add_scale_bars(
        xlim[1] + 1,
        ylim[0] + 0.3,
        -bar_length if xlim[1] > xlim[0] else bar_length,
        bar_length if ylim[1] > ylim[0] else -bar_length,
        xlabel="UMAP 1",
        ylabel="UMAP 2",
        fmt="",
        pad=(1.5, -4.5),
        size=4,
        text_kw=dict(color="k"),
        lw=0.3,
        c="k",
    )
    g.savefig(output_dir / "map.pdf")
    plt.close(g.fig)

    do_flip = determine_flip_needed(feature_matrix, embedding, labels, flip)
    feature_matrix_flipped = (
        np.reshape(df_features, (-1, FRAMES_PER_EVENT, n_features))
        .transpose(0, 2, 1)
        .reshape((-1, n_features * FRAMES_PER_EVENT))
    )
    feature_matrix_flipped[do_flip] = flip(feature_matrix_flipped[do_flip])
    for k in range(n_clusters):
        mask = labels == k
        mean_heading = feature_matrix_flipped[mask, -FRAMES_PER_EVENT:].mean()
        if mean_heading < 0:
            feature_matrix_flipped[mask] = flip(feature_matrix_flipped[mask])
            do_flip[mask] = ~do_flip[mask]

    feature_matrix_flipped = feature_matrix_flipped.reshape(
        (-1, n_features, FRAMES_PER_EVENT)
    )

    cluster_palette = get_cluster_palette(n_clusters)
    cluster_ids = np.arange(n_clusters)
    g = Grid(
        (8, 12),
        (n_features, len(cluster_ids)),
        sharex=True,
        sharey="row",
        spines="lb",
        space=(2, 2),
    )
    g[:, 1:].set_visible_sides("")
    g[:, 0].set_visible_sides("l")
    for i_cluster, cluster_id in enumerate(cluster_ids):
        ax = g[0, i_cluster]
        ax.set_title(cluster_id + 1, size=4, pad=0)
        for i, feature_name in enumerate(FEATURE_NAMES):
            mu = feature_matrix_flipped[labels == cluster_id, i].mean(0)
            s = feature_matrix_flipped[labels == cluster_id, i].std(0)
            ax = g[i, i_cluster]
            c = cluster_palette[cluster_id]
            ax.plot(t, mu, color=c, lw=1)
            ax.fill_between(t, mu - s, mu + s, alpha=0.5, color=c, lw=0)
            if i_cluster == 0:
                ax.add_text(
                    0,
                    0.5,
                    feature_name,
                    ha="r",
                    va="c",
                    transform="a",
                    pad=(-8, 0),
                    size=4,
                )
            ax.tick_params("y", labelsize=3)
            ax.yaxis.set_major_locator(MaxNLocator(3, integer=True))

    ax = g[-1, 0]
    ax.set_xlabel("Time (s)", size=4)
    ax.tick_params("x", labelsize=3)
    g.set_xlim(t[0], t[-1])
    ax.set_xticks([-2, 0, 2])
    ax.set_visible_sides("lb")
    g[0].make_ax().add_text(0.5, 1, "Cluster (#)", ha="c", va="b", pad=(0, 6), size=4)
    _ylims = np.array([ax.get_ylim() for ax in g.axs[:, 0]])
    g.savefig(output_dir / "features.pdf")
    plt.close(g.fig)

    kwargs = dict(
        save_dir=output_dir / "videos",
        t=t,
        embedding=embedding,
        feature_matrix_flipped=feature_matrix_flipped,
        feature_names=FEATURE_NAMES,
        labels=labels,
        im_regions=im_regions,
        bound=bound,
        xlim=xlim,
        ylim=ylim,
        df=df,
        df_fly=df_fly,
        n_rows=16,
        n_cols=6,
        size=(256, 64),
        zoom=1.5,
        thickness=1,
        do_flip=~do_flip,
    )

    for cluster_id in [3, 15]:
        save_grid_video(cluster_id=cluster_id, **kwargs)

    df_genotypes = (
        df_fly.select("genotype", "brain_region", "split").unique().sort("genotype")
    )
    df_events = df[::FRAMES_PER_EVENT, ["fly", "event_id"]].join(
        df_fly.select("fly", "genotype"), on="fly", how="left"
    )
    df_genotypes = df_genotypes.join(
        df_fly.group_by("genotype").agg(pl.len().alias("n_flies")),
        on="genotype",
        how="left",
    ).join(
        df_events.group_by("genotype", "fly")
        .len()
        .group_by("genotype")
        .agg(
            n_events_mean=pl.mean("len").round().cast(pl.Int32),
            n_events_std=pl.std("len").round().cast(pl.Int32),
        ),
        on="genotype",
        how="left",
    )

    from joblib import Parallel, delayed

    def run_energy_test(genotype: str) -> tuple[str, float, float]:
        control = CONTROL_GENOTYPES[
            df_genotypes.filter(pl.col("genotype").eq(genotype))["split"].item()
        ]
        results = energy_test_fly(
            embedding[df_events["genotype"] == control],
            embedding[df_events["genotype"] == genotype],
            df_events.filter(pl.col("genotype").eq(control))["fly"]
            .cast(pl.Categorical)
            .to_physical()
            .to_numpy(),
            df_events.filter(pl.col("genotype").eq(genotype))["fly"]
            .cast(pl.Categorical)
            .to_physical()
            .to_numpy(),
            n_samples=10000,
        )
        return (genotype, float(results[0]), float(results[1]))

    if (output_dir / "energy_test.parquet").exists():
        df_test = pl.read_parquet(output_dir / "energy_test.parquet")
    else:
        df_test = pl.DataFrame(
            Parallel(n_jobs=-1)(
                delayed(run_energy_test)(genotype)
                for genotype in tqdm(df_genotypes["genotype"])
            ),
            schema=["genotype", "E", "p"],
            orient="row",
        )
        df_test.write_parquet(output_dir / "energy_test.parquet")

    bar_length = 3
    n_cols = 10
    cmap = "gray_r"
    fig_height = (np.abs(np.diff(ylim) / np.diff(xlim)) * 60).item()

    for (split_type,), df_split in df_genotypes.join(
        df_test, on="genotype", how="left"
    ).group_by("split"):
        df_split = df_split.sort(["p", "E"], descending=[True, False])
        n_rows = int(np.ceil(len(df_split) / n_cols))
        if n_rows == 1:
            n_cols = len(df_split)
        g = Grid((60, fig_height), (n_rows, n_cols), space=(6, 24), facecolor="w")
        g.set_visible_sides("")
        for i, row in enumerate(df_split.iter_rows(named=True)):
            ax = g.axs.ravel()[i]
            im_kde = get_kde(
                embedding[df_events["genotype"] == row["genotype"]], bound=bound, bw=0.4
            )[0]
            im_kde /= im_kde.mean()
            ax.imshow(
                im_kde,
                cmap=cmap,
                extent=(-bound, bound, -bound, bound),
                origin="lower",
                vmin=0,
                vmax=20,
            )
            ax.contour(
                im_regions + 1,
                levels=np.arange(n_clusters + 1),
                colors=(0.3,) * 3,
                extent=(-bound, bound, -bound, bound),
                antialiased=True,
                linewidths=0.5,
                origin="lower",
            )
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect("equal")

            if row["genotype"] == CONTROL_GENOTYPES[split_type]:
                annotation_text = f"{row['genotype']}\n(control)"
            else:
                annotation_text = f"{row['genotype']}\np={row['p']:.3f}, E={row['E']:.3f}"

            annotation_text += f"\n{row['n_flies']} flies, {row['n_events_mean']}$\\pm${row['n_events_std']} events"
            ax.add_text(
                0.5,
                1,
                annotation_text,
                ha="c",
                va="b",
                transform="a",
                c=brain_region_palette[row["brain_region"]],
                size=6,
                pad=(0, 0),
            )
            ax = g[0, 0]
            ax.add_scale_bars(
                xlim[1] + 0.4,
                ylim[0] + 0.5,
                -bar_length if xlim[1] > xlim[0] else bar_length,
                bar_length if ylim[1] > ylim[0] else -bar_length,
                xlabel="UMAP 1",
                ylabel="UMAP 2",
                fmt="",
                pad=(2, -4.5),
                size=3.5,
            )
            for ax in g.axs.ravel():
                ax.set_rasterized(True)

            g[0, 0].set_zorder(100)
            g.set_visible_sides("")
            g.savefig(output_dir / f"genotype_{split_type}.pdf")
            plt.close(g.fig)


if __name__ == "__main__":
    main()
