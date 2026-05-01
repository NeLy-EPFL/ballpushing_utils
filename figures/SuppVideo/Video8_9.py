from pathlib import Path
import polars as pl
import numpy as np
import matplotlib.pyplot as plt


def to_rgb_8bit(c):
    from matplotlib.colors import to_hex

    hex_color = to_hex(c)
    return tuple(int(hex_color[i : i + 2], 16) for i in (1, 3, 5))


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
    flipped=False,
) -> np.ndarray:
    from video_reader import PyVideoReader
    import cv2
    from ballpushing_utils.plotting.palette import FLY_COLORS

    def get_warped_pos(df_event, i, name, affine_mat):
        x = df_event[i, f"{name}_x"]
        y = df_event[i, f"{name}_y"]
        return tuple(map(round, affine_mat @ (x, y, 1)))

    df_event = df.filter(pl.col("fly").eq(fly) & pl.col("event_id").eq(event_id))
    fly_dir = Path(df_fly.filter(pl.col("fly").eq(fly)).select("path").item())
    video_path = (fly_dir / f"{fly_dir.stem}_preprocessed.mp4").as_posix()
    images = PyVideoReader(video_path)[df_event["frame"]]

    raw_ball_pos = np.ravel(df_event.select("ball_x", "ball_y")[len(df_event) // 2])
    new_ball_pos = np.array(ball_rel_pos) * size
    affine_mat = (
        np.array([[1, 0, new_ball_pos[0]], [0, 1, new_ball_pos[1]], [0, 0, 1]])
        @ np.array([[0, -zoom, 0], [zoom, 0, 0], [0, 0, 1]])
        @ np.array([[1, 0, -raw_ball_pos[0]], [0, 1, -raw_ball_pos[1]], [0, 0, 1]])
    )[:2]

    frames = np.zeros((len(images), size[1], size[0], 3), dtype=np.uint8)

    if flipped:
        palette = dict(
            FLY_COLORS,
            lf=FLY_COLORS["rf"],
            rf=FLY_COLORS["lf"],
            lm=FLY_COLORS["rm"],
            rm=FLY_COLORS["lm"],
            lh=FLY_COLORS["rh"],
            rh=FLY_COLORS["lh"],
        )
    else:
        palette = FLY_COLORS
    ball_color = to_rgb_8bit(ball_color)

    for i, im in enumerate(images):
        im = cv2.warpAffine(
            im,
            affine_mat,
            size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        thorax_pos = get_warped_pos(df_event, i, "thorax", affine_mat)
        for dst, color in palette.items():
            cv2.line(
                im,
                thorax_pos,
                get_warped_pos(df_event, i, dst, affine_mat),
                color=to_rgb_8bit(color),
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )
        cv2.circle(
            im,
            get_warped_pos(df_event, i, "ball", affine_mat),
            radius=ball_radius,
            color=ball_color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
        frames[i] = im[::-1] if flipped else im

    return frames


def choice(a, k, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(a), k, replace=False))
    return a[idx]


def get_left_panel_for_video(
    embedding: np.ndarray,
    df_features: pl.DataFrame,
    t: np.ndarray,
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
    from ballpushing_utils.plotting.palette import get_cluster_palette

    frames_per_event = len(t)
    n_clusters = labels.max() + 1
    cluster_palette = get_cluster_palette(n_clusters)
    n_features = df_features.shape[1]
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
    text_kwargs = dict(color="w", transform="a", size=3, va="c")
    for i, feature in enumerate(df_features.columns):
        ax = axs[i]
        data = np.reshape(df_features[feature], (-1, frames_per_event))
        mean = data[labels == cluster_id].mean(0)
        std = data[labels == cluster_id].std(0)
        c = "w"
        ax.plot(t, mean, color=c, lw=1)
        ax.fill_between(t, mean - std, mean + std, alpha=0.5, color=c, lw=0)
        ax.add_text(0, 0.5, feature, pad=(-7, 0), rotation=0, ha="r", **text_kwargs)
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
        0.5,
        1,
        f"Cluster {cluster_id + 1}",
        pad=(0, 2),
        ha="c",
        **dict(text_kwargs, size=4),
    )
    ax.scatter(*embedding.T, s=1, c=cluster_palette[labels], alpha=0.05, lw=0, marker=".")
    contour_kwargs = dict(antialiased=True, origin="lower", extent=(-bound, bound) * 2)
    ax.contour(
        im_regions + 1,
        levels=np.arange(n_clusters + 1),
        colors=to_hex((0.5,) * 3),
        **contour_kwargs,
    )
    ax.contour(
        im_regions == cluster_id,
        colors="w",
        linewidths=0.5,
        zorder=100,
        **contour_kwargs,
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

    timeline_ax = g[1:, 1].make_ax(sharex=axs[0], behind=True)
    time_cursor = timeline_ax.axvline(t[0], color="w", lw=0.25, ls="--", animated=True)
    return g, time_cursor, timeline_ax


def save_grid_video(
    save_path: str | Path,
    t: np.ndarray,
    embedding: np.ndarray,
    df_features: pl.DataFrame,
    labels: np.ndarray,
    cluster_id: int,
    flipped: np.ndarray,
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
    df_events = df.select("fly", "event_id").unique(maintain_order=True)[selected_events]
    frames = np.empty((frames_per_event, size[1] * n_rows, size[0] * n_cols, 3), dtype=np.uint8)
    reshaped = frames.reshape((frames_per_event, n_rows, size[1], n_cols, size[0], 3))

    for i, (fly, event_id) in enumerate(df_events.iter_rows()):
        reshaped[:, i // n_cols, :, i % n_cols] = get_event_frames(
            fly,
            event_id,
            df,
            df_fly,
            size=size,
            flipped=flipped[selected_events[i]],
            **kwargs,
        )

    g, time_cursor, timeline_ax = get_left_panel_for_video(
        embedding=embedding,
        df_features=df_features,
        t=t,
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
    canvas_bg = g.fig.canvas.copy_from_bbox(g.fig.bbox)
    plt.ioff()

    padding_frame = None
    with get_writer(
        Path(save_path).as_posix(),
        fps=15,
        codec="libx264",
        ffmpeg_log_level="quiet",
    ) as writer:
        for i, video_frame in enumerate(frames):
            g.fig.canvas.restore_region(canvas_bg)
            try:
                time_cursor.set_xdata([t[i]])
                timeline_ax.draw_artist(time_cursor)
            except AttributeError:
                for time_cursor_, timeline_ax_ in zip(time_cursor, timeline_ax):
                    time_cursor_.set_xdata([t[i]])
                    timeline_ax_.draw_artist(time_cursor_)
            g.fig.canvas.blit(g.fig.bbox)
            panel_frame = np.frombuffer(g.fig.canvas.buffer_rgba(), dtype=np.uint8)
            panel_frame = panel_frame.reshape(g.fig.canvas.get_width_height()[::-1] + (4,))[..., :3]

            if padding_frame is None:
                total_width = panel_frame.shape[1] + video_frame.shape[1]
                width_pad = int(np.ceil((total_width / 32))) * 32 - total_width
                padding_frame = np.zeros((len(panel_frame), width_pad, 3), np.uint8)

            writer.append_data(np.concatenate((panel_frame, padding_frame, video_frame), axis=1))

    plt.close(g.fig)


if __name__ == "__main__":
    from ballpushing_utils.paths import figure_output_dir, get_cache_dir
    from ballpushing_utils.preprocess_screen_data import get_preprocessed_data, get_features

    t = np.arange(-60, 60) / 29
    out_dir = figure_output_dir("SuppVideo", __file__)
    cache_dir = get_cache_dir()

    df, df_fly = get_preprocessed_data(
        cache_dir,
        genotype_name_csv="../Fig3-Screen/data/genotype_names.csv",
        excluded_genotypes=["Wild-type(PR)", "Wild-type(Canton-S)", "TH", "TH-2"],
    )
    df_features = get_features(df)

    if (cache_dir / "umap.npz").exists():
        umap_data = {}
        with np.load(cache_dir / "umap.npz") as f:
            umap_data["embedding"] = f["embedding"]
            umap_data["bound"] = f["bound"].item()
            umap_data["xlim"] = f["xlim"][::-1]
            umap_data["ylim"] = f["ylim"]
            umap_data["im_regions"] = f["im_regions"]
            umap_data["labels"] = f["labels"]
            umap_data["flipped"] = ~f["flipped"]
    else:
        raise FileNotFoundError("UMAP embedding not found. Please run Fig3-Screen/fig3_umap.py first.")

    kwargs = dict(
        t=t,
        df_features=df_features,
        df=df,
        df_fly=df_fly,
        n_rows=16,
        n_cols=6,
        size=(256, 64),
        zoom=1.5,
        thickness=1,
        **umap_data,
    )

    for cluster_id, video_num in zip([3, 15], [8, 9]):
        save_grid_video(
            cluster_id=cluster_id, save_path=out_dir / f"video_{video_num}_cluster{cluster_id + 1}.mp4", **kwargs
        )
