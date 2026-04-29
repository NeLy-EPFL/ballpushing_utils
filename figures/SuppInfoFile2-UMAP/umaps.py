import polars as pl
import numpy as np
import matplotlib.pyplot as plt


def run_energy_tests(
    embedding: np.ndarray,
    df_events: pl.DataFrame,
    df_genotypes: pl.DataFrame,
    control_genotypes: pl.DataFrame,
    n_samples: int = 10000,
):
    from joblib import Parallel, delayed
    from ballpushing_utils.stats.energy_test import energy_test_fly
    from tqdm import tqdm

    def run_energy_test(embedding, df_events, control, genotype, n_samples):
        results = energy_test_fly(
            embedding[df_events["genotype"] == control],
            embedding[df_events["genotype"] == genotype],
            df_events.filter(pl.col("genotype").eq(control))["fly"].cast(pl.Categorical).to_physical().to_numpy(),
            df_events.filter(pl.col("genotype").eq(genotype))["fly"].cast(pl.Categorical).to_physical().to_numpy(),
            n_samples=n_samples,
        )
        return (genotype, float(results[0]), float(results[1]))

    return pl.DataFrame(
        Parallel(n_jobs=-1)(
            delayed(run_energy_test)(embedding, df_events, control_genotypes[row["split"]], row["genotype"], n_samples)
            for row in tqdm(df_genotypes.iter_rows(named=True), total=df_genotypes.height)
        ),
        schema=["genotype", "E", "p"],
        orient="row",
    )


def plot_umaps(
    df_events: pl.DataFrame,
    df_genotypes: pl.DataFrame,
    df_test: pl.DataFrame,
    control_genotypes: pl.DataFrame,
    split_type: str,
    embedding: np.ndarray,
    bound: float,
    im_regions: np.ndarray,
    xlim,
    ylim,
    bar_length: float = 3,
    n_cols: int = 10,
    cmap: str = "gray_r",
    panel_width=60.0,
):
    from mplex import Grid
    from ballpushing_utils.plotting.palette import BRAIN_REGION_COLORS
    from ballpushing_utils.stats.kde import get_kde

    n_clusters = int(im_regions.max() + 1)

    panel_height = (np.abs(np.diff(ylim) / np.diff(xlim)) * panel_width).item()
    df_split = (
        df_genotypes.filter(split=split_type)
        .join(df_test, on="genotype", how="left")
        .sort(["p", "E"], descending=[True, False])
    )
    n_rows = int(np.ceil(len(df_split) / n_cols))
    if n_rows == 1:
        n_cols = len(df_split)
    g = Grid((panel_width, panel_height), (n_rows, n_cols), space=(6, 24), facecolor="w")
    g.set_visible_sides("")
    for i, row in enumerate(df_split.iter_rows(named=True)):
        ax = g.axs.ravel()[i]
        im_kde = get_kde(embedding[df_events["genotype"] == row["genotype"]], bound=bound, bw=0.4)[0]
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

        if row["genotype"] == control_genotypes[split_type]:
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
            c=BRAIN_REGION_COLORS[row["brain_region"]],
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
    return g


if __name__ == "__main__":
    from ballpushing_utils.paths import get_cache_dir
    from ballpushing_utils.preprocess_screen_data import get_preprocessed_data
    from ballpushing_utils.paths import figure_output_dir

    control_genotypes = {"y": "Empty-Split-Gal4", "n": "Empty-Gal4", "m": "TNT×PR"}
    frames_per_event = 120
    cache_dir = get_cache_dir()
    df, df_fly = get_preprocessed_data(
        cache_dir,
        genotype_name_csv="../Fig3-Screen/data/genotype_names.csv",
        excluded_genotypes=["Wild-type(PR)", "Wild-type(Canton-S)", "TH", "TH-2"],
    )
    df_genotypes = df_fly.select("genotype", "brain_region", "split").unique().sort("genotype")
    df_events = df[::frames_per_event, ["fly", "event_id"]].join(df_fly.select("fly", "genotype"), on="fly", how="left")
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

    if (cache_dir / "umap.npz").exists():
        umap_data = {}
        with np.load(cache_dir / "umap.npz") as f:
            umap_data["embedding"] = f["embedding"]
            umap_data["bound"] = f["bound"].item()
            umap_data["xlim"] = f["xlim"]
            umap_data["ylim"] = f["ylim"]
            umap_data["im_regions"] = f["im_regions"]
    else:
        raise FileNotFoundError("UMAP embedding not found. Please run Fig3-Screen/fig3_umap.py first.")

    if (cache_dir / "energy_test.parquet").exists():
        df_test = pl.read_parquet(cache_dir / "energy_test.parquet")
    else:
        df_test = run_energy_tests(umap_data["embedding"], df_events, df_genotypes, control_genotypes)
        df_test.write_parquet(cache_dir / "energy_test.parquet")

    out_dir = figure_output_dir("SuppInfoFile2", __file__)

    split_type = "m"
    g = plot_umaps(
        df_events,
        df_genotypes,
        df_test,
        control_genotypes,
        split_type="m",
        **umap_data,
    )
    save_name = {"m": "mutant", "y": "split-gal4", "n": "gal4"}[split_type]
    g.savefig(out_dir / f"genotype_{split_type}.pdf")
    plt.close(g.fig)
