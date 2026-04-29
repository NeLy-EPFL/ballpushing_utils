from pathlib import Path
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

edges = [
    ("thorax", "head"),
    ("thorax", "abdomen"),
    ("thorax", "lf"),
    ("thorax", "lm"),
    ("thorax", "lh"),
    ("thorax", "rf"),
    ("thorax", "rm"),
    ("thorax", "rh"),
]

fly_palette = {
    "lf": "#0f7399",
    "lm": "#188bad",
    "lh": "#76bcc9",
    "rf": "#b82032",
    "rm": "#c95750",
    "rh": "#d38279",
    "head": "green",
    "abdomen": "purple",
}

brain_region_palette = {
    "Control": "#000000",
    "Olfaction": "#a74399",
    "DN": "#987c56",
    "LH": "#653293",
    "CX": "#00aeef",
    "MB": "#38489e",
    "MB extrinsic neurons": "#12793d",
    "NM": "#b35c27",
    "Neuropeptide": "#746c2c",
    "Control": "#747474",
    "fchON": "#747474",
    "JON": "#747474",
    "Vision": "#761313",
}


def get_heading(amp: np.ndarray) -> np.ndarray:
    """Calculate the heading vector from the coordinates of the nodes by
    fitting a line through the neck, thorax, and abdomen nodes.

    Parameters
    ----------
    amp : np.ndarray
        Array of shape (..., 3, 2) containing the coordinates of the anterior,
        medial, and posterior nodes.

    Returns
    -------
    np.ndarray
        Array of shape (..., 2) containing the heading vectors.
    """
    cov = np.einsum(
        "...pi,...pj->...ij", *(amp - amp.mean(axis=-2, keepdims=True),) * 2
    )
    heading = np.linalg.eigh(cov)[1][..., -1]
    dot_prod = np.einsum("...i,...i->...", amp[..., 0, :] - amp[..., -1, :], heading)
    invert = dot_prod < 0
    heading[invert] = -heading[invert]
    return heading


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


def rotate_embedding(Z):
    from sklearn.decomposition import PCA

    Z = Z - Z.mean(0)
    return Z @ PCA(n_components=2).fit(Z).components_.T


def cluster_points(Z, n_clusters):
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(Z)
    return labels


def sort_labels(labels, Z):
    from scipy.spatial import distance_matrix
    from python_tsp.exact import solve_tsp_dynamic_programming

    n_clusters = labels.max() + 1
    cluster_centers = np.array([Z[labels == k].mean(0) for k in range(n_clusters)])

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
    return np.array(
        [
            _get_bbox_1d(a.any(i))
            for i in combinations(reversed(range(a.ndim)), a.ndim - 1)
        ]
    )


def get_areas(Z, labels, bound, n_bins, density_threshold):
    from scipy.ndimage import binary_fill_holes

    n_clusters = labels.max() + 1
    Y, X = np.mgrid[-bound : bound : n_bins * 1j, -bound : bound : n_bins * 1j]
    mg = np.stack((X, Y), axis=-1)[:, :, None]
    centers = np.array([Z[labels == k].mean(0) for k in range(n_clusters)])
    im_argmin = np.linalg.norm(mg - centers, axis=-1).argmin(-1).astype(np.float32)
    im_argmin[~binary_fill_holes(get_kde(Z, n_bins, bound)[0] > density_threshold)] = -1
    ylim, xlim = (get_bbox(im_argmin != -1) / n_bins - 0.5) * bound * 2 * 1.05
    return im_argmin, xlim, ylim


def determine_flip_needed(X, Z, labels, flip):
    from pynndescent.distances import euclidean

    n_clusters = np.max(labels) + 1
    need_flip = np.zeros(len(X), dtype=bool)
    centroid0 = None
    for k in range(n_clusters):
        bidx = labels == k
        idx = np.where(bidx)[0]
        Xk = X[bidx]
        Zk = Z[bidx]
        centroid_id = np.argmin(np.linalg.norm(Zk - Zk.mean(0), axis=1))
        centroid = Xk[centroid_id]
        if k == 0:
            centroid0 = centroid
        else:
            if euclidean(centroid0, flip(centroid)) < euclidean(centroid0, centroid):
                centroid = flip(centroid)

        for i, x in enumerate(Xk):
            need_flip[idx[i]] = euclidean(centroid, flip(x)) < euclidean(centroid, x)
    return need_flip


def get_cluster_palette(n_clusters):
    import colorcet as cc
    from matplotlib.colors import ListedColormap

    return ListedColormap(cc.rainbow)(np.linspace(0, 1, n_clusters))


def plot_map_regions(im_regions, Z, labels, xlim, ylim, bound):
    from mplex import Grid
    from mplex.text import set_text_outline
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
        *Z.T,
        s=2,
        c=cluster_palette[labels],
        alpha=0.05,
        marker=".",
        lw=0,
        rasterized=True,
    )
    for k in range(n_clusters):
        ax.add_text(
            *Z[labels == k].mean(0), str(k + 1), ha="c", va="c", size=6, color="k"
        )
        set_text_outline(ax.texts[-1], lw=1, c="w")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis("off")
    ax.set_aspect("equal")
    return g


def to_rgb_8bit(c):
    from matplotlib.colors import to_hex

    hex = to_hex(c)
    return tuple(int(hex[i : i + 2], 16) for i in (1, 3, 5))


def get_event_frames(
    fly: str,
    event_id: int,
    df: pl.DataFrame,
    df_fly: pl.DataFrame,
    size=(256, 64),
    ball_rel_pos=(0.7, 0.5),
    zoom=1.5,
    thickness=1,
    ball_radius=16,
    ball_color="#00aeef",
    do_flip=False,
) -> np.ndarray:
    from video_reader import PyVideoReader
    import cv2

    def get_warped_pos(df_event, i, name, mat):
        x = df_event[i, f"{name}_x"]
        y = df_event[i, f"{name}_y"]
        return tuple(map(round, mat @ (x, y, 1)))

    df_event = df.filter(pl.col("fly").eq(fly) & pl.col("event_id").eq(event_id))
    fly_dir = Path(df_fly.filter(pl.col("fly").eq(fly)).select("path").item())
    video_path = (fly_dir / f"{fly_dir.stem}_preprocessed.mp4").as_posix()
    images = PyVideoReader(video_path)[df_event["frame"]]

    raw_ball_pos = np.ravel(df_event.select("ball_x", "ball_y")[len(df_event) // 2])
    new_ball_pos = np.array(ball_rel_pos) * size
    mat = (
        np.array([[1, 0, new_ball_pos[0]], [0, 1, new_ball_pos[1]], [0, 0, 1]])
        @ np.array([[0, -zoom, 0], [zoom, 0, 0], [0, 0, 1]])
        @ np.array([[1, 0, -raw_ball_pos[0]], [0, 1, -raw_ball_pos[1]], [0, 0, 1]])
    )[:2]

    frames = np.zeros((len(images), size[1], size[0], 3), dtype=np.uint8)

    if do_flip:
        palette = dict(
            fly_palette,
            lf=fly_palette["rf"],
            rf=fly_palette["lf"],
            lm=fly_palette["rm"],
            rm=fly_palette["lm"],
            lh=fly_palette["rh"],
            rh=fly_palette["lh"],
        )
    else:
        palette = fly_palette
    ball_color = to_rgb_8bit(ball_color)

    for i, im in enumerate(images):
        im = cv2.warpAffine(
            im,
            mat,
            size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        for src, dst in edges:
            cv2.line(
                im,
                get_warped_pos(df_event, i, src, mat),
                get_warped_pos(df_event, i, dst, mat),
                color=to_rgb_8bit(palette[dst]),
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )
        cv2.circle(
            im,
            get_warped_pos(df_event, i, "ball", mat),
            radius=ball_radius,
            color=ball_color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
        frames[i] = im[::-1] if do_flip else im

    return frames


def choice(a, k, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(a), k, replace=False))
    return a[idx]


def save_grid_video(
    save_dir: str | Path,
    t: np.ndarray,
    Z: np.ndarray,
    X_flipped: np.ndarray,
    feature_names: list[str],
    labels: np.ndarray,
    cluster_id: int,
    do_flip: np.ndarray,
    im_regions: np.ndarray,
    bound: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    df: pl.DataFrame,
    df_fly: pl.DataFrame,
    n_rows: int,
    n_cols: int,
    size=(256, 64),
    seed=0,
    **kwargs,
):
    from imageio import get_writer

    frames_per_event = len(t)
    n_videos = n_rows * n_cols
    selected_events = choice(np.where(labels == cluster_id)[0], n_videos, seed=seed)
    df_events = df.select("fly", "event_id").unique(maintain_order=True)[
        selected_events
    ]
    frames = np.empty(
        (frames_per_event, size[1] * n_rows, size[0] * n_cols, 3), dtype=np.uint8
    )
    reshaped = frames.reshape((frames_per_event, n_rows, size[1], n_cols, size[0], 3))

    for i, (fly, event_id) in enumerate(df_events.iter_rows()):
        reshaped[:, i // n_cols, :, i % n_cols] = get_event_frames(
            fly,
            event_id,
            df,
            df_fly,
            size=size,
            do_flip=do_flip[selected_events[i]],
            **kwargs,
        )

    g, t_line, axt = get_left_panel_for_video(
        Z=Z,
        X_flipped=X_flipped,
        t=t,
        feature_names=feature_names,
        labels=labels,
        cluster_id=cluster_id,
        im_regions=im_regions,
        bound=bound,
        xlim=xlim,
        ylim=ylim,
        height=frames.shape[1],
    )

    plt.ion()
    g.fig.canvas.draw()
    bg = g.fig.canvas.copy_from_bbox(g.fig.bbox)
    plt.ioff()

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    im_pad = None
    with get_writer(
        (save_dir / f"cluster_{cluster_id + 1:02d}.mp4").as_posix(),
        fps=15,
        codec="libx264",
        ffmpeg_log_level="quiet",
    ) as writer:
        for i, im1 in enumerate(frames):
            g.fig.canvas.restore_region(bg)
            try:
                t_line.set_xdata([t[i]])
                axt.draw_artist(t_line)
            except AttributeError:
                for t_line_, axt_ in zip(t_line, axt):
                    t_line_.set_xdata([t[i]])
                    axt_.draw_artist(t_line_)
            g.fig.canvas.blit(g.fig.bbox)
            im0 = np.frombuffer(g.fig.canvas.buffer_rgba(), dtype=np.uint8)
            im0 = im0.reshape(g.fig.canvas.get_width_height()[::-1] + (4,))[..., :3]

            if im_pad is None:
                w_ = im0.shape[1] + im1.shape[1]
                pad = int(np.ceil((w_ / 32))) * 32 - w_
                im_pad = np.zeros((len(im0), pad, 3), np.uint8)

            writer.append_data(np.concatenate((im0, im_pad, im1), axis=1))

    plt.close(g.fig)


def get_left_panel_for_video(
    Z: np.ndarray,
    X_flipped: np.ndarray,
    t: np.ndarray,
    feature_names: list[str],
    labels: np.ndarray,
    cluster_id: int,
    im_regions: np.ndarray,
    bound: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    height: int,
):
    from matplotlib.colors import to_hex
    from mplex import Grid
    from matplotlib.ticker import MaxNLocator

    n_clusters = labels.max() + 1
    cluster_palette = get_cluster_palette(n_clusters)
    n_features = len(feature_names)
    dpi = height / (40 + 20 * n_features + 2 + 2 * (n_features - 1) + 2 + 15) * 72

    g = Grid(
        ([14, 20], [40] + [20] * n_features),
        facecolor="k",
        sharex=False,
        sharey=False,
        spines="l",
        space=(10, [2] + [2] * (n_features - 1)),
        border=(2, 2, 2, 15),
        dpi=dpi,
    )
    axs = g.axs[1:, 1]
    txt_kw = dict(color="w", transform="a", size=3, va="c")
    for i, feature_name in enumerate(feature_names):
        ax = axs[i]
        mu = X_flipped[labels == cluster_id, i].mean(0)
        s = X_flipped[labels == cluster_id, i].std(0)
        c = "w"
        ax.plot(t, mu, color=c, lw=1)
        ax.fill_between(t, mu - s, mu + s, alpha=0.5, color=c, lw=0)
        ax.add_text(0, 0.5, feature_name, pad=(-7, 0), rotation=0, ha="r", **txt_kw)
        ax.yaxis.set_major_locator(MaxNLocator(3, integer=True))
        ax.tick_params(axis="both", colors="w", labelsize=2.5)

        if i == n_features - 1:
            ax.set_xlabel("Time (s)", size=3, color="w")
            g.set_xlim(t[0], t[-1])
            ax.set_xticks([-2, 0, 2])
        else:
            ax.set_xticks([])

        for spine in ax.spines.values():
            spine.set_color("w")

    g[-1].set_visible_sides("lb")
    g[0].set_visible_sides("")

    ax = g[0].make_ax(behind=False)
    ax.add_text(
        0.5, 1, f"Cluster {cluster_id + 1}", pad=(0, 2), ha="c", **dict(txt_kw, size=4)
    )
    ax.scatter(*Z.T, s=1, c=cluster_palette[labels], alpha=0.05, lw=0, marker=".")
    cnt_kws = dict(antialiased=True, origin="lower", extent=(-bound, bound) * 2)
    ax.contour(
        im_regions + 1,
        levels=np.arange(n_clusters + 1),
        colors=to_hex((0.5,) * 3),
        **cnt_kws,
    )
    ax.contour(
        im_regions == cluster_id, colors="w", linewidths=0.5, zorder=100, **cnt_kws
    )
    bar_length = 3
    ax.add_scale_bars(
        xlim[1] + 0.5,
        ylim[0] - 0,
        -bar_length if xlim[1] > xlim[0] else bar_length,
        bar_length if ylim[1] > ylim[0] else -bar_length,
        xlabel="UMAP 1",
        ylabel="UMAP 2",
        fmt="",
        pad=(1.5, -2.5),
        size=2,
        text_kw=dict(color="w"),
        lw=0.3,
        c="w",
    )
    ax.set_aspect("equal")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    axt = g[1:, 1].make_ax(sharex=axs[0], behind=True)
    t_line = axt.axvline(t[0], color="w", lw=0.25, ls="--", animated=True)
    return g, t_line, axt


def energy_test_fly(
    X: np.ndarray,
    Y: np.ndarray,
    fly_ids_x: np.ndarray,
    fly_ids_y: np.ndarray,
    n_samples: int,
    random_state: int = 0,
):
    from scipy.spatial.distance import cdist

    n_flies_x = fly_ids_x.max() + 1
    n_flies_y = fly_ids_y.max() + 1
    n_flies = n_flies_x + n_flies_y
    fly_ids = np.concatenate((fly_ids_x, fly_ids_y + n_flies_x))
    XY = np.concatenate((X, Y))
    d = cdist(XY, XY)

    def estat(ix, iy):
        a = d[np.ix_(ix, ix)]
        b = d[np.ix_(iy, iy)]
        c = d[np.ix_(ix, iy)]
        return 2 * np.mean(c) - np.mean(a) - np.mean(b)

    Eobs = estat(np.arange(len(X)), np.arange(len(X), len(XY)))
    rng = np.random.default_rng(random_state)
    cnt = 0
    temp = np.arange(n_flies)

    for _ in range(n_samples):
        rng.shuffle(temp)
        ix = np.where(np.isin(fly_ids, temp[:n_flies_x]))[0]
        iy = np.where(np.isin(fly_ids, temp[n_flies_x:]))[0]
        if estat(ix, iy) >= Eobs:
            cnt += 1

    p = (cnt + 1) / (n_samples + 1)
    return Eobs, p
