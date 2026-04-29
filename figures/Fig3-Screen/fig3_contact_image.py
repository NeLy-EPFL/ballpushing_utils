from pathlib import Path
import numpy as np


def load_data(
    cache_dir: Path,
    fly="231208_TNT_Fine_2_Videos_Tracked_arena6_corridor3",
    event_id=198,
    frame_within_event=38,
) -> tuple[dict[str, float], np.ndarray]:
    from pathlib import Path
    from video_reader import PyVideoReader
    from ballpushing_utils.preprocess_screen_data import get_preprocessed_data

    df, df_fly = get_preprocessed_data(cache_dir)
    data = df.filter(fly=fly, event_id=event_id)[frame_within_event].to_dicts()[0]
    fly_dir = Path(df_fly.filter(fly=fly)["path"].item())
    video_path = (fly_dir / f"{fly_dir.stem}_preprocessed.mp4").as_posix()
    im = PyVideoReader(video_path)[data["frame"]]
    return data, im


def plot_contact_image(
    data: dict[str, float],
    im: np.ndarray,
    line_width=1,
    ball_radius=12,
    ball_color="#00aeef",
):
    from matplotlib.patches import Circle
    from matplotlib.transforms import Affine2D
    from mplex import Grid
    from ballpushing_utils.plotting.palette import FLY_COLORS

    affine2d = Affine2D().translate(-data["ball_x"], -data["ball_y"]).rotate_deg(90)
    g = Grid()
    ax = g.item()
    transform = affine2d + ax.transData
    ax.imshow(im, transform=transform)
    for key, color in FLY_COLORS.items():
        ax.plot(
            [data["thorax_x"], data[f"{key}_x"]],
            [data["thorax_y"], data[f"{key}_y"]],
            color=color,
            transform=transform,
            linewidth=line_width,
            solid_capstyle="round",
        )
    ax.add_patch(
        Circle(
            (data["ball_x"], data["ball_y"]),
            ball_radius,
            edgecolor=ball_color,
            facecolor="none",
            lw=line_width,
            transform=transform,
        )
    )
    ax.set_xlim(-80, 40)
    ax.set_ylim(16, -16)
    ax.axis("off")
    return g


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from ballpushing_utils.paths import figure_output_dir, get_cache_dir

    cache_dir = get_cache_dir()
    out_dir = figure_output_dir("Figure3", __file__)

    data, im = load_data(cache_dir=cache_dir)
    g = plot_contact_image(data, im)
    g.savefig(out_dir / "fig3_contact.pdf")
    plt.close(g.fig)
