from pathlib import Path
import numpy as np
import polars as pl

frames_per_event = 120
n_clusters = 20
map_pad = 0.05
n_bins = 512
density_threshold = 5e-5


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


def get_features(df: pl.DataFrame) -> pl.DataFrame:
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


def get_embedding(df_features):
    from pynndescent.distances import euclidean
    from numba import njit
    from umap import UMAP

    @njit
    def flip(x: np.ndarray) -> np.ndarray:
        x = x.copy()
        x[..., -120:] *= -1
        return x

    @njit
    def my_dist(a: np.ndarray, b: np.ndarray) -> float:
        return min(euclidean(a, b), euclidean(a, flip(b)))

    def rotate_embedding(embedding):
        from sklearn.decomposition import PCA

        embedding = embedding - embedding.mean(0)
        return embedding @ PCA(n_components=2).fit(embedding).components_.T

    n_features = df_features.shape[1]
    feature_matrix = np.reshape(df_features, (-1, frames_per_event, n_features)).transpose(0, 2, 1)
    feature_matrix /= feature_matrix.std(axis=(0, 2), keepdims=True)
    feature_matrix = feature_matrix.reshape((-1, n_features * frames_per_event))

    umap = UMAP(n_components=2, random_state=0, metric=my_dist)
    embedding = umap.fit_transform(feature_matrix)
    embedding = rotate_embedding(embedding)
    return embedding


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


def cluster_points(embedding, n_clusters):
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(embedding)
    return labels


def sort_labels(labels, embedding, starting_cluster=0):
    from scipy.spatial import distance_matrix
    from python_tsp.exact import solve_tsp_dynamic_programming

    if starting_cluster:
        n_clusters = labels.max() + 1
        labels = (labels - starting_cluster) % n_clusters

    n_clusters = labels.max() + 1
    cluster_centers = np.array([embedding[labels == k].mean(0) for k in range(n_clusters)])

    mat = distance_matrix(cluster_centers, cluster_centers)
    mat[:, 0] = 0

    permutation, total_distance = solve_tsp_dynamic_programming(mat)
    order = np.argsort(permutation)
    assert (np.unique(order) == np.arange(n_clusters)).all()

    new_labels = np.zeros_like(labels)
    for k in range(n_clusters):
        new_labels[labels == k] = order[k]

    return new_labels


def get_bbox(a):
    from itertools import combinations

    def _get_bbox_1d(a):
        return np.array((a.argmax(), len(a) - a[::-1].argmax()))

    a = np.asarray(a).astype(bool)
    return np.array([_get_bbox_1d(a.any(i)) for i in combinations(reversed(range(a.ndim)), a.ndim - 1)])


def get_cluster_regions(embedding, labels, bound, n_bins, density_threshold):
    from scipy.ndimage import binary_fill_holes

    n_clusters = labels.max() + 1
    grid_y, grid_x = np.mgrid[-bound : bound : n_bins * 1j, -bound : bound : n_bins * 1j]
    coord_grid = np.stack((grid_x, grid_y), axis=-1)[:, :, None]
    centers = np.array([embedding[labels == k].mean(0) for k in range(n_clusters)])
    region_map = np.linalg.norm(coord_grid - centers, axis=-1).argmin(-1).astype(np.float32)
    region_map[~binary_fill_holes(get_kde(embedding, n_bins, bound)[0] > density_threshold)] = -1
    ylim, xlim = (get_bbox(region_map != -1) / n_bins - 0.5) * bound * 2 * 1.05
    return region_map, xlim, ylim


def get_cluster_palette(n_clusters: int):
    import colorcet as cc
    from matplotlib.colors import ListedColormap

    return ListedColormap(cc.rainbow)(np.linspace(0, 1, n_clusters))


def plot_map_regions(
    im_regions: np.ndarray,
    embedding: np.ndarray,
    labels: np.ndarray,
    xlim: tuple[int, int],
    ylim: tuple[int, int],
    bound: float,
):
    from mplex import Grid
    from matplotlib import patheffects
    from matplotlib.colors import to_hex

    n_clusters = labels.max() + 1
    cluster_palette = get_cluster_palette(n_clusters)
    g = Grid(100)
    ax = g.item()
    ax.contour(
        im_regions + 1,
        levels=np.arange(n_clusters + 1),
        colors=to_hex((0.3,) * 3),
        extent=(-bound, bound, -bound, bound),
        antialiased=True,
        linewidths=0.5,
        origin="lower",
    )
    ax.scatter(
        *embedding.T,
        s=2,
        c=cluster_palette[labels],
        alpha=0.05,
        marker=".",
        lw=0,
        rasterized=True,
    )
    for k in range(n_clusters):
        text = ax.text(
            *embedding[labels == k].mean(0),
            str(k + 1),
            ha="center",
            va="center",
            size=6,
            color="k",
        )
        text.set_path_effects(
            [
                patheffects.Stroke(linewidth=1, foreground="w"),
                patheffects.Normal(),
            ]
        )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis("off")
    ax.set_aspect("equal")

    bar_length = 2.5

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
    return g


def flip_if_needed(heading, embedding, labels=None):
    if labels is None:

        def sqeuclidean(a, b):
            return np.einsum("...i,...i->...", *(a - b,) * 2)

        heading = heading.copy()
        argmin = sqeuclidean(embedding, embedding.mean(0)).argmin()
        centroid = heading[argmin]
        flipped = sqeuclidean(heading, -centroid) < sqeuclidean(heading, centroid)
        heading[flipped] = -heading[flipped]
        if heading.mean() < 0:
            heading = -heading
            flipped = ~flipped
        return heading, flipped

    heading = heading.copy()
    flipped = np.zeros(len(heading), dtype=bool)
    for k in range(labels.max() + 1):
        mask = labels == k
        heading[mask], flipped[mask] = flip_if_needed(heading[mask], embedding[mask])
    return heading, flipped


def plot_features(df_features, labels):
    from mplex import Grid
    from matplotlib.ticker import MaxNLocator

    n_features = df_features.shape[1]
    n_clusters = labels.max() + 1
    cluster_palette = get_cluster_palette(n_clusters)
    t = (np.arange(frames_per_event) - frames_per_event // 2) / 29

    g = Grid((8, 12), (n_features, n_clusters), sharey="row", spines="lb", space=2)
    g[:, 1:].set_visible_sides("")
    g[:, 0].set_visible_sides("l")
    for i, feature in enumerate(df_features.columns):
        ax = g[i, 0]
        ax.add_text(0, 0.5, feature, ha="r", va="c", transform="a", pad=(-8, 0), size=4)
        ax.tick_params("y", labelsize=3)
        ax.yaxis.set_major_locator(MaxNLocator(3, integer=True))
        data = np.reshape(df_features[feature], (-1, frames_per_event))
        for k in range(n_clusters):
            color = cluster_palette[k]
            mean = data[labels == k].mean(0)
            std = data[labels == k].std(0)
            ax = g[i, k]
            ax.plot(t, mean, color=color, lw=1)
            ax.fill_between(t, mean - std, mean + std, alpha=0.5, color=color, lw=0)
            ax.set_ymargin(0)
            if i == 0:
                ax.set_title(k + 1, size=4, pad=0)

    ax = g[-1, 0]
    ax.set_xlabel("Time (s)", size=4, labelpad=0)
    ax.tick_params("x", labelsize=3)
    g.set_xlim(t[0], t[-1])
    ax.set_xticks([-2, 0, 2])
    ax.set_visible_sides("lb")
    g[0].make_ax().add_text(0.5, 1, "Cluster (#)", ha="c", va="b", pad=(0, 6), size=4)

    return g


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from ballpushing_utils.preprocess_screen_data import get_preprocessed_data
    from ballpushing_utils.paths import figure_output_dir, get_cache_dir

    cache_dir = get_cache_dir()
    out_dir = figure_output_dir("Figure3", __file__)

    df, df_fly = get_preprocessed_data(cache_dir)

    genotype_names = dict(zip(*pl.read_csv("data/genotype_names.csv").to_dict().values()))
    df_fly = df_fly.with_columns(pl.col("genotype").replace(genotype_names))
    excluded_genotypes = ["Wild-type(PR)", "Wild-type(Canton-S)", "TH", "TH-2"]
    df_fly = df_fly.filter(~pl.col("genotype").is_in(excluded_genotypes))
    df = df.filter(pl.col("fly").is_in(df_fly["fly"].implode()))

    df_features = get_features(df)
    heading = np.reshape(df_features[:, -1], (-1, frames_per_event))

    if (cache_dir / "umap.npz").exists():
        with np.load(cache_dir / "umap.npz") as data:
            embedding = data["embedding"]
            labels = data["labels"]
            im_regions = data["im_regions"]
            xlim = data["xlim"]
            ylim = data["ylim"]
            flipped = data["flipped"]
            bound = data["bound"].item()

        heading = heading.copy()
        heading[flipped] = -heading[flipped]
    else:
        embedding = get_embedding(df_features)

        labels = cluster_points(embedding, n_clusters=n_clusters)
        labels = sort_labels(labels, embedding, starting_cluster=1)

        bound = np.abs(embedding).max() * (1 + map_pad)
        im_regions, xlim, ylim = get_cluster_regions(embedding, labels, bound, n_bins, density_threshold)

        heading, flipped = flip_if_needed(heading, embedding, labels)

    df_features = df_features.with_columns(pl.Series(df_features.columns[-1], heading.ravel()))

    g = plot_map_regions(im_regions, embedding, labels, xlim[::-1], ylim, bound)
    g.savefig(out_dir / "fig3_umap.pdf")
    plt.close(g.fig)

    g = plot_features(df_features, labels)
    g.savefig(out_dir / "fig3_features.pdf")
    feature_ylims = g[:, 0].get_ylim()
    plt.close(g.fig)

    if not (cache_dir / "umap.npz").exists():
        cache_dir.mkdir(exist_ok=True)
        np.savez_compressed(
            cache_dir / "umap.npz",
            embedding=embedding,
            labels=labels,
            im_regions=im_regions,
            bound=bound,
            xlim=xlim,
            ylim=ylim,
            feature_ylims=feature_ylims,
            flipped=flipped,
        )
