from pathlib import Path
from matplotlib.colors import to_hex
from video_reader import PyVideoReader
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

edges = [
    ("t", "h"),
    ("t", "a"),
    ("t", "lf"),
    ("t", "lm"),
    ("t", "lh"),
    ("t", "rf"),
    ("t", "rm"),
    ("t", "rh"),
    ("t", "lw"),
    ("t", "rw"),
]

fly_palette = {
    "lf": "#0f7399",
    "lm": "#188bad",
    "lh": "#76bcc9",
    "rf": "#b82032",
    "rm": "#c95750",
    "rh": "#d38279",
    "h": "green",
    "a": "purple",
    "lw": "lightblue",
    "rw": "pink",
}

region_palette = {
    "Control": (0, 0, 0),
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


def c2xy(
    src: np.ndarray | pd.DataFrame | pd.Series,
    flatten=False,
):
    """Convert complex numbers in a dataframe to x, y coordinates.

    Parameters
    ----------
    src
        Complex numbers to be converted to x, y coordinates.
    flatten : bool, optional
        If True, reshape data from (..., n * 2) to (..., n, 2).
        Ignored if src is a DataFrame or Series.

    Returns
    -------
    DataFrame
        Dataframe containing the x, y coordinates.
    """
    if not isinstance(src, (pd.DataFrame, pd.Series)):
        src = np.asarray(src)
        dst = np.stack((src.real, src.imag), axis=-1)
        if flatten:
            return dst.reshape((len(src), -1))
        return dst

    if isinstance(src, pd.Series):
        return pd.DataFrame({"x": src.values.real, "y": src.values.imag}, src.index)

    if isinstance(src, pd.DataFrame):
        df = src
        data = np.stack([df.values.real, df.values.imag], axis=-1)
        data = data.reshape((len(df), -1))
        tuples = [i if isinstance(i, tuple) else (i,) for i in df.columns]
        tuples = [i + (j,) for i in tuples for j in "xy"]
        names = np.append(df.columns.names, "coord")
        columns = pd.MultiIndex.from_tuples(tuples, names=names)
        return pd.DataFrame(data, df.index, columns)
    else:
        return np.stack((src.real, src.imag), axis=-1)


def xy2c(src: np.ndarray | pd.DataFrame | pd.Series, coord_level=-1, unflatten=False):
    """Convert x, y coordinates in a dataframe to complex numbers.

    Parameters
    ----------
    src : np.ndarray or DataFrame or Series
        Dataframe or series containing x, y coordinates.
    coord_level : int or str
        Level of columns that represents the x, y coordinates.
        Ignored if src is not a DataFrame or Series.
    unflatten : bool, optional
        If True, reshape data from (..., n * 2) ro (..., n, 2) before conversion.
        Ignored if df is a DataFrame or Series.

    Returns
    -------
    DataFrame
        Dataframe in which each (x, y) pair is represented by
        a complex number x + y * 1j.
    """
    if isinstance(src, pd.Series):
        src = src.xs("x", level=coord_level) + src.xs("y", level=coord_level) * 1j
    elif isinstance(src, pd.DataFrame):
        try:
            src = (
                src.xs("x", axis=1, level=coord_level)
                + src.xs("y", axis=1, level=coord_level) * 1j
            )
        except TypeError:
            src = src[["x", "y"]] @ (1, 1j)
    else:
        if unflatten:
            src = src.reshape((*src.shape[:-1], -1, 2))
        src = src @ (1, 1j)

    return src


def intervals_to_mask(intervals, n=None):
    """Convert intervals to a boolean mask.

    Parameters
    ----------
    intervals : np.ndarray
        Array of intervals where each row represents the start and end indices
        of a True interval.
    n : int, optional
        Length of the mask. Defaults to the maximum interval end index.

    Returns
    -------
    np.ndarray
        Boolean mask where True values indicate the intervals.
    """

    if n is None:
        n = np.max(intervals)

    mask = np.zeros(n, dtype=bool)
    for a, b in intervals:
        mask[a:b] = True

    return mask


def mask_to_intervals(x: np.ndarray):
    """Convert a boolean mask to intervals of True values.

    Parameters
    ----------
    x : np.ndarray
        Boolean array where True values indicate the mask.

    Returns
    -------
    np.ndarray
        Array of intervals where each row represents the start and end indices
        of a True interval.
    """
    return np.column_stack(
        [np.argwhere(np.diff(np.pad(x, 1) * 1) == i).ravel() for i in (1, -1)]
    )


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
        Array of shape (...,) containing the heading vectors in the complex plane.
    """
    amp = amp.copy()
    isna = ~np.isfinite(amp).all(-1)
    nanmean = np.empty_like(amp)
    nanmean[:] = np.nanmean(amp, axis=-2, keepdims=True)
    amp[isna] = nanmean[isna]
    cov = np.einsum(
        "...pi,...pj->...ij", *(amp - amp.mean(axis=-2, keepdims=True),) * 2
    )
    heading = np.linalg.eigh(cov)[1][..., -1] @ (1, 1j)
    n, th, a = (amp[..., i, :] @ (1, 1j) for i in range(3))
    dot_prod = (np.stack((n - th, th - a, n - a)) * np.conj(heading)).real
    invert = (dot_prod < 0).sum(0) > (dot_prod > 0).sum(0)
    heading[invert] = -heading[invert]
    return heading


def reassign_labels(labels):
    """Reassign labels such that the largest cluster is labeled 0, the second largest is labeled 1, etc."""
    unique_labels = np.unique(labels)
    label_counts = {
        label: np.sum(labels == label) for label in unique_labels if label != -1
    }
    sorted_labels = sorted(label_counts, key=label_counts.get, reverse=True)
    label_map = {
        old_label: new_label for new_label, old_label in enumerate(sorted_labels)
    }
    return np.array([label_map.get(label, -1) for label in labels])


def preprocess(data_path: str) -> pd.DataFrame:
    df = pd.read_feather(data_path).set_index(["fly", "event_id"]).sort_index()

    col_slice = np.s_[4:22]
    to_drop = []

    for (fly, event_id), df_ in df.groupby(level=["fly", "event_id"]):
        if len(df_) != 120 or (~np.isfinite(df_.iloc[:, col_slice].values.real)).all():
            to_drop.append((fly, event_id))

    for key in to_drop:
        df.drop(key, inplace=True)

    X = df.iloc[:, col_slice].values.reshape((len(df), -1, 2)) @ (1, 1j)
    X = pd.DataFrame(
        X,
        columns=[i.removeprefix("x_") for i in df.columns[col_slice][::2]],
        index=df.index,
    )
    for key, df_ in X.groupby(level=["fly", "event_id"]):
        X.loc[key] = df_.interpolate(
            method="linear",
            limit_direction="both",
            limit=30,
            axis=0,
            limit_area="inside",
        )

    X.loc[X["Lfront"].isna(), "Lfront"] = X.loc[X["Lfront"].isna(), "Rfront"].values
    X.loc[X["Rfront"].isna(), "Rfront"] = X.loc[X["Rfront"].isna(), "Lfront"].values
    X.loc[X["Lmid"].isna(), "Lmid"] = X.loc[X["Lmid"].isna(), "Rmid"].values
    X.loc[X["Rmid"].isna(), "Rmid"] = X.loc[X["Rmid"].isna(), "Lmid"].values
    X.loc[X["Lhind"].isna(), "Lhind"] = X.loc[X["Lhind"].isna(), "Rhind"].values
    X.loc[X["Rhind"].isna(), "Rhind"] = X.loc[X["Rhind"].isna(), "Lhind"].values

    for key, df_ in X.groupby(level=["fly", "event_id"]):
        X.loc[key] = df_.interpolate(method="linear", limit=10, axis=0)

    X.dropna(inplace=True, axis=0)

    to_drop2 = []

    for (fly, event_id), df_ in X.groupby(level=["fly", "event_id"]):
        if len(df_) != 120:
            to_drop2.append((fly, event_id))

    for key in to_drop2:
        X.drop(key, inplace=True)
        df.drop(key, inplace=True)

    X = X[
        [
            "centre_preprocessed",
            "Thorax",
            "Head",
            "Abdomen",
            "Lfront",
            "Lmid",
            "Lhind",
            "Rfront",
            "Rmid",
            "Rhind",
        ]
    ]

    return df, X


def plot_skeleton(x, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    for src, dst in edges:
        if src not in x:
            if src == "Thorax":
                p1 = 0
            else:
                continue
        else:
            p1 = x[src]

        if dst not in x:
            if dst == "Thorax":
                p2 = 0
            else:
                continue
        else:
            p2 = x[dst]

        ax.plot(np.real([p1, p2]), np.imag([p1, p2]), color=fly_palette[dst], **kwargs)


def interactive_plot(
    X: pd.DataFrame,
    Z: np.ndarray,
    labels: np.ndarray,
):
    from bokeh.plotting import figure, show
    from bokeh.layouts import row
    from bokeh.models import HoverTool, CustomJS, ColumnDataSource
    from bokeh.io import output_notebook

    output_notebook()

    assert len(X) == len(Z) == len(labels)

    # Use all data instead of subset
    n_points = len(Z)  # Use all points
    indices = np.arange(n_points)  # Use all indices

    # Prepare pose data for all points
    pose_data = []
    for idx in indices:
        pose_row = X.iloc[idx]
        pose_dict = {}
        # Extract coordinates for each body part
        for part in [
            "Head",
            "Abdomen",
            "Lfront",
            "Lmid",
            "Lhind",
            "Rfront",
            "Rmid",
            "Rhind",
        ]:
            if part in pose_row:
                pose_dict[f"{part}_x"] = np.real(pose_row[part])
                pose_dict[f"{part}_y"] = np.imag(pose_row[part])
            else:
                pose_dict[f"{part}_x"] = 0
                pose_dict[f"{part}_y"] = 0

        pose_data.append(pose_dict)

    # Create main data source
    source_data = {"x": Z[:, 0], "y": Z[:, 1], "cluster": labels, "index": indices}

    # Add pose data to source
    for pose in pose_data:
        for key, value in pose.items():
            if key not in source_data:
                source_data[key] = []
            source_data[key].append(value)

    source = ColumnDataSource(source_data)

    # Create the left panel - UMAP scatter plot
    p1 = figure(
        width=400, height=400, title="UMAP", tools="pan,wheel_zoom,box_zoom,reset,save"
    )

    # Color map for clusters
    from bokeh.models import LinearColorMapper
    from bokeh.palettes import Category20

    color_mapper = LinearColorMapper(
        palette=Category20[20],
        low=min(labels),
        high=max(labels),
    )

    # Add scatter plot
    scatter = p1.scatter(
        "x",
        "y",
        size=1,
        source=source,
        alpha=0.6,
        color={"field": "cluster", "transform": color_mapper},
    )

    p1.title.text_font_size = "14pt"

    # Create the right panel - Pose visualization
    p2 = figure(
        width=400,
        height=400,
        title="Pose",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        x_range=(-60, 60),
        y_range=(-60, 60),
    )

    # Create line segments for skeleton
    skeleton_lines = []
    for _, dst in edges:
        line_source = ColumnDataSource({"x": [0, 0], "y": [0, 0]})
        color = fly_palette.get(dst, "black")
        line = p2.line(
            "x", "y", source=line_source, line_width=3, color=color, alpha=0.8
        )
        skeleton_lines.append((line, line_source))

    # Add a circle at thorax position
    thorax_circle = p2.circle([0], [0], size=8, color="red", alpha=0.8)

    p2.title.text_font_size = "14pt"
    # p2.xaxis.axis_label = "X Position"
    # p2.yaxis.axis_label = "Y Position"

    # Create hover callback
    hover_callback = CustomJS(
        args=dict(
            source=source,
            skeleton_lines=[line_source for _, line_source in skeleton_lines],
            thorax=thorax_circle.data_source,
        ),
        code="""
        const indices = cb_data.index.indices;
        if (indices.length > 0) {
            const idx = indices[0];
            const data = source.data;

            // Get pose data for this point
            const pose_data = {
                'Head': [data['Head_x'][idx], data['Head_y'][idx]],
                'Thorax': [0, 0],  // Always at origin
                'Abdomen': [data['Abdomen_x'][idx], data['Abdomen_y'][idx]],
                'Lfront': [data['Lfront_x'][idx], data['Lfront_y'][idx]],
                'Lmid': [data['Lmid_x'][idx], data['Lmid_y'][idx]],
                'Lhind': [data['Lhind_x'][idx], data['Lhind_y'][idx]],
                'Rfront': [data['Rfront_x'][idx], data['Rfront_y'][idx]],
                'Rmid': [data['Rmid_x'][idx], data['Rmid_y'][idx]],
                'Rhind': [data['Rhind_x'][idx], data['Rhind_y'][idx]]
            };

            // Update skeleton lines
            const edges = [
                ['Head', 'Thorax'],
                ['Thorax', 'Abdomen'],
                ['Thorax', 'Lfront'],
                ['Thorax', 'Lmid'],
                ['Thorax', 'Lhind'],
                ['Thorax', 'Rfront'],
                ['Thorax', 'Rmid'],
                ['Thorax', 'Rhind']
            ];

            for (let i = 0; i < edges.length && i < skeleton_lines.length; i++) {
                const [src, dst] = edges[i];
                const line_data = skeleton_lines[i].data;
                line_data['x'] = [pose_data[src][0], pose_data[dst][0]];
                line_data['y'] = [pose_data[src][1], pose_data[dst][1]];
                skeleton_lines[i].change.emit();
            }

            // Update thorax position
            thorax.data['x'] = [0];
            thorax.data['y'] = [0];
            thorax.change.emit();
        }
    """,
    )

    # Add hover tool to the scatter plot
    hover = HoverTool(
        tooltips=[
            ("Index", "@index"),
            ("Cluster", "@cluster"),
            # ("UMAP X", "@x{0.00}"),
            # ("UMAP Y", "@y{0.00}")
        ],
        callback=hover_callback,
        renderers=[scatter],
    )

    p1.add_tools(hover)

    # Create the complete layout
    layout = row(p1, p2)

    # Display the interactive plot
    show(layout)


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


def to_rgb_8bit(c):
    hex = to_hex(c)
    return tuple(int(hex[i : i + 2], 16) for i in (1, 3, 5))


def choice(a, k, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(a), k, replace=False))
    return a[idx]


def get_grid_videos(
    fly,
    event_id,
    frame_id,
    n_frames,
    df_flies,
    df_orig,
    need_flip,
    width=256,
    height=64,
    zoom=1.5,
    ball_rel_pos_x=0.7,
    ball_rel_pos_y=0.5,
    ball_color="lime",
    thickness=1,
    ball_radius=16,
):
    fly_dir = Path(df_flies.loc[fly, "path"])
    video_path = (fly_dir / f"{fly_dir.stem}_preprocessed.mp4").as_posix()

    if not Path(video_path).exists():
        video_path = sorted(fly_dir.glob("*preprocessed.mp4"))
        assert len(video_path) == 1
        video_path = video_path[0].as_posix()

    images = PyVideoReader(video_path)[frame_id : frame_id + n_frames]
    df_ = df_orig.loc[fly, event_id, frame_id : frame_id + n_frames]
    cx, cy = df_["b"].iloc[n_frames // 2].astype(int)
    ball_pos_x = ball_rel_pos_x * width
    ball_pos_y = ball_rel_pos_y * height
    mat = (
        np.array([[1, 0, ball_pos_x], [0, 1, ball_pos_y], [0, 0, 1]])
        @ np.array([[0, -zoom, 0], [zoom, 0, 0], [0, 0, 1]])
        @ np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
    )
    mat = mat[:2].astype(np.float32)
    results = np.zeros((n_frames, height, width, 3), dtype=np.uint8)

    for i_im, im in enumerate(images):
        im = cv2.warpAffine(
            im,
            mat,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        xy = df_.iloc[i_im].unstack()
        for src, dst in edges[:-2]:
            pt1 = tuple(map(round, mat @ (*xy.loc[src], 1)))
            pt2 = tuple(map(round, mat @ (*xy.loc[dst], 1)))
            if need_flip:
                dst = {"l": "r", "r": "l"}.get(dst[0], dst[0]) + dst[1:]
            color = to_rgb_8bit(fly_palette[dst])
            cv2.line(
                im, pt1, pt2, color=color, thickness=thickness, lineType=cv2.LINE_AA
            )

        xy = tuple(map(round, mat @ (*xy.loc["b"], 1)))
        color = to_rgb_8bit(ball_color)
        cv2.circle(
            im,
            xy,
            radius=ball_radius,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

        results[i_im] = im if not need_flip else im[::-1]

    return results


def get_non_outliers(Z, pad=0.05, threshold=5e-5):
    from scipy import ndimage as ndi

    bound = np.abs(Z).max() * (1 + pad)
    im = get_kde(Z, bound=bound)[0]
    n_bins = len(im)
    mask = im >= threshold
    im_labels = ndi.label(mask)[0]
    mask = im_labels == np.bincount(im_labels.ravel())[1:].argmax() + 1
    non_outlier = mask[tuple(((Z / bound + 1) / 2 * n_bins).astype(int).T)[::-1]]
    return non_outlier, mask, bound


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


def get_need_flip(X, Z, labels, flip):
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


def split_list(lst, k):
    import math

    n = len(lst)
    num_chunks = math.ceil(n / k)
    result = [lst[i * k : (i + 1) * k] for i in range(num_chunks)]
    return result


def make_video(
    k,
    labels,
    save_dir,
    n_frames,
    df,
    df_flies,
    df_orig,
    need_flip,
    get_plot,
    dpi,
    t,
    n_rows=16,
    n_cols=8,
    width=256,
    height=64,
    verbose=False,
    seed=0,
    **kwargs,
):
    from imageio import get_writer

    n_videos = n_rows * n_cols
    frames = np.full(
        (n_frames, height * n_rows, width * n_cols, 3), 255, dtype=np.uint8
    )
    reshaped = frames.reshape((n_frames, n_rows, height, n_cols, width, 3))
    selected_clips = choice(np.where(labels == k)[0], n_videos, seed=seed)

    if verbose:
        from tqdm import tqdm

        selected_clips = tqdm(selected_clips)

    for i_grid, i_clip in enumerate(selected_clips):
        fly, event_id, frame_id = df.index[i_clip]

        reshaped[:, i_grid // n_cols, :, i_grid % n_cols] = get_grid_videos(
            fly=fly,
            event_id=event_id,
            frame_id=frame_id,
            n_frames=n_frames,
            df_flies=df_flies,
            df_orig=df_orig,
            need_flip=not need_flip[i_clip],
            width=width,
            height=height,
            **kwargs,
        )

    g, t_line, axt = get_plot(k, dpi=dpi)
    plt.ion()
    g.fig.canvas.draw()
    bg = g.fig.canvas.copy_from_bbox(g.fig.bbox)
    plt.ioff()

    if verbose:
        frames = tqdm(frames)

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    im_pad = None
    with get_writer(
        (save_dir / f"cluster_{k + 1:02d}.mp4").as_posix(),
        fps=15,
        codec="libx264",
        ffmpeg_log_level="quiet",
        # output_params=["-x265-params", "log-level=quiet", "-tag:v", "hvc1"],
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


def concat_videos(video_paths, save_path):
    from subprocess import run

    temp_txt_file = Path("temp.txt")
    with open(temp_txt_file, "w") as f:
        for video_path in video_paths:
            video_path = Path(video_path).absolute().as_posix()
            f.write(f"file '{video_path}'\n")

    save_path = Path(save_path).absolute()
    save_path.parent.mkdir(exist_ok=True, parents=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            temp_txt_file,
            "-c",
            "copy",
            save_path.as_posix(),
        ]
    )
    temp_txt_file.unlink()


def plot_maps_for_lines(
    df_lines_path,
    Z,
    im_regions,
    df,
    df_flies,
    xlim,
    ylim,
    bound,
    output_dir,
    n_cols=10,
    n_rows=7,
    cmap="gray_r",
):
    from natsort import natsorted
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex
    from mplex import Grid

    df_lines = pd.read_csv(df_lines_path).set_index("line")
    lines_sorted = natsorted(df_lines["display_name"])
    df_lines["i"] = [lines_sorted.index(i) for i in df_lines["display_name"]]
    region_list = list(df_lines.loc[df_flies["line"], "region"].value_counts().index)
    region_list.remove("Control")
    region_list = ["Control"] + region_list
    df_lines["j"] = [region_list.index(i) for i in df_lines["region"]]
    df_lines.sort_values(["j", "i"], inplace=True)
    df_lines = df_lines.iloc[:, :2]
    line_list = list(df_lines["display_name"].drop_duplicates().values)
    lines = df_lines.loc[
        df_flies.loc[df.index.get_level_values(0), "line"].values, "display_name"
    ].values

    h = (np.abs(np.diff(ylim) / np.diff(xlim)) * 60).item()
    n_clusters = int(im_regions.max() + 1)

    for i_list, list_i in enumerate(split_list(line_list, n_cols * n_rows)):
        g = Grid((60, h), (n_rows, n_cols), space=(0, 10), facecolor="w")
        axs = g.axs.ravel()
        for ax, line in zip(axs, list_i):
            bidx = lines == line
            # x, y = Z[bidx].T
            im_kde = get_kde(Z[bidx], bound=bound, bw=0.4)[0]
            im_kde /= im_kde.mean()
            ax.imshow(
                im_kde,
                cmap=cmap,
                extent=(-bound, bound, -bound, bound),
                origin="lower",
                vmin=0,
                vmax=20,
            )
            # ret = sns.kdeplot(x=x, y=y, bw_method=0.15, cmap=cmap, fill=True, levels=100, ax=ax)
            # ax.scatter(x, y, s=0.5, c="k", lw=0, alpha=0.1, rasterized=True)
            ax.contour(
                im_regions + 1,
                levels=np.arange(n_clusters + 1),
                colors=to_hex((0.3,) * 3),
                extent=(-bound, bound, -bound, bound),
                antialiased=True,
                linewidths=0.5,
                origin="lower",
            )
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect("equal")
            ax.add_text(
                0.5,
                1,
                line,
                ha="c",
                va="b",
                transform="a",
                c=region_palette[
                    df_lines.loc[df_lines["display_name"].eq(line), "region"].unique()[
                        0
                    ]
                ],
                size=6,
                pad=(0, 0),
            )
        ax = g[0, 0]
        bar_length = 3
        ax.add_scale_bars(
            xlim[1] + 0.4,
            ylim[0] + 0.5,
            -bar_length if xlim[1] > xlim[0] else bar_length,
            bar_length if ylim[1] > ylim[0] else -bar_length,
            xlabel="UMAP 1",
            ylabel="UMAP 2",
            fmt="",
            pad=(2, -4.5),
            size=4,
        )
        for ax in g.axs.ravel():
            ax.set_rasterized(True)

        g[0, 0].set_zorder(100)
        g.set_visible_sides("")
        g.savefig(output_dir / f"lines_{i_list}.pdf")
        plt.close(g.fig)

    from pypdf import PdfWriter

    output_pdf = output_dir / "all_lines.pdf"

    if output_pdf.exists():
        output_pdf.unlink()

    with PdfWriter(output_pdf) as merger:
        for video_path in sorted(output_dir.glob("lines_*.pdf")):
            merger.append(video_path)


def energy_test_fly(X, Y, fly_ids_x, fly_ids_y, n_samples, random_state):
    from scipy.spatial.distance import cdist

    fly_ids_x = pd.Series(fly_ids_x).factorize()[0]
    fly_ids_y = pd.Series(fly_ids_y).factorize()[0]
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


def energy_test_event(X, Y, n_samples, random_state):
    from scipy.spatial.distance import cdist

    rng = np.random.default_rng(random_state)
    n, m = len(X), len(Y)
    Z = np.vstack([X, Y])
    D = cdist(Z, Z)  # pairwise distances

    # helpers to index blocks
    idxX = np.arange(n)
    idxY = np.arange(n, n + m)

    def estat(ix, iy):
        a = D[np.ix_(ix, ix)]
        b = D[np.ix_(iy, iy)]
        c = D[np.ix_(ix, iy)]
        return 2 * np.mean(c) - np.mean(a) - np.mean(b)

    E_obs = estat(idxX, idxY)

    # permutation null
    labels = np.arange(n + m)
    cnt = 0
    for _ in range(n_samples):
        rng.shuffle(labels)
        ix = labels[:n]
        iy = labels[n:]
        if estat(ix, iy) >= E_obs:  # upper tail
            cnt += 1
    p = (cnt + 1) / (n_samples + 1)
    return E_obs, p
