import functools
import operator
import typing

import cv2
import matplotlib.artist
import matplotlib.axes
import matplotlib.colors
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import plotnine as p9
import polars as pl

LEG_RIGHT_FRONT = (186, 30, 49)
LEG_RIGHT_MIDDLE = (201, 86, 79)
LEG_RIGHT_REAR = (213, 133, 121)
LEG_LEFT_FRONT = (15, 115, 153)
LEG_LEFT_MIDDLE = (26, 141, 175)
LEG_LEFT_REAR = (117, 190, 203)
BODY = (210, 210, 210)

colours = np.array(
    [
        BODY,
        BODY,
        BODY,
        LEG_LEFT_FRONT,
        LEG_RIGHT_FRONT,
        LEG_LEFT_MIDDLE,
        LEG_RIGHT_MIDDLE,
        LEG_LEFT_REAR,
        LEG_RIGHT_REAR,
        BODY,
    ]
)

DEFAULT_CROP_HALF_WINDOW = 368


def crop_image_from_centre(
    image: np.ndarray, centre: int, left_space: int, right_space: int
):
    x_min = centre - left_space
    x_max = centre + right_space - 1

    if x_min >= 0 and x_max < image.shape[1]:
        return image[:, x_min : x_max + 1]

    padding_left = -x_min if x_min < 0 else 0
    padding_right = x_max - image.shape[1] if x_max >= image.shape[1] else 0

    cropped_image = np.zeros_like(
        image, shape=(image.shape[0], left_space + right_space, image.shape[2])
    )
    x_min = max(0, x_min)
    x_max = min(image.shape[1], x_max)

    cropped_image[:, padding_left : -padding_right - 1] = image[:, x_min:x_max]
    return cropped_image


def get_best_thorax(keypoints: np.ndarray):
    thorax_top = keypoints[1]
    thorax_bottom = keypoints[9]
    if thorax_top[2] < 0.5 and thorax_bottom[2] < 0.5:
        return np.array([0, 0, 0])
    else:
        return thorax_top if keypoints[1, 2] > keypoints[9, 2] else thorax_bottom


def crop_image_from_thorax(image: np.ndarray, keypoints: np.ndarray):
    return crop_image_from_centre(
        image,
        round(get_best_thorax(keypoints)[0]),
        DEFAULT_CROP_HALF_WINDOW,
        DEFAULT_CROP_HALF_WINDOW,
    )


def plot_pose_on_image(image: np.ndarray, keypoints: np.ndarray, *, conf_threshold=0.5):
    """_summary_

    Args:
            image (np.ndarray): _description_
            keypoints (np.ndarray): [num_joints,3] array [x,y,conf]
    """
    main_thorax = get_best_thorax(keypoints)

    if np.all(main_thorax == 0):
        return image
    main_thorax_point = (round(main_thorax[0]), round(main_thorax[1]))

    not_thorax_indices = list(set(range(len(keypoints))).difference([1, 9]))

    for (x, y, conf), colour in zip(
        keypoints[not_thorax_indices], colours[not_thorax_indices]
    ):
        colour = [int(c) for c in colour]
        if conf > conf_threshold:
            image = cv2.line(image, main_thorax_point, (round(x), round(y)), colour, 2)
    return image


def padded_crop(
    image: np.ndarray, centre_point: tuple[int, int], width: int, height: int
):
    """
    Crop a box of size (`width`*`height`) from `image`, centred on the `centre_point`,
    while padding the output with zeros where this would be out of bounds of the original image.

    Therefore, the cropped image is guaranteed to be the right size and centred on the desired position.
    """
    x_start = int(centre_point[0] - width / 2)
    y_start = int(centre_point[1] - height / 2)
    x_end = int(centre_point[0] + width / 2)
    y_end = int(centre_point[1] + height / 2)

    if (
        x_start >= 0
        and y_start >= 0
        and x_end <= image.shape[1]
        and y_end <= image.shape[0]
    ):
        return image[y_start:y_end, x_start:x_end]
    else:
        output_shape = list(image.shape)
        output_shape[0] = height
        output_shape[1] = width
        output = np.zeros(output_shape, dtype=image.dtype)
        x_offset_start = -x_start if x_start < 0 else 0
        y_offset_start = -y_start if y_start < 0 else 0
        x_offset_end = x_end - image.shape[1] if x_end > image.shape[1] else 0
        y_offset_end = y_end - image.shape[0] if y_end > image.shape[0] else 0
        # we need to use None instead of -0 as the end index to mean "select until the end"
        none_if_zero = lambda x: None if x == 0 else x
        output[
            y_offset_start : none_if_zero(-y_offset_end),
            x_offset_start : none_if_zero(-x_offset_end),
        ] = image[
            y_start + y_offset_start : none_if_zero(y_end - y_offset_end),
            x_start + x_offset_start : none_if_zero(x_end - x_offset_end),
        ]
        return output


#
# PLOTNINE
#
paper_theme_plotnine = p9.theme(
    axis_line_x=p9.element_line(color="black"),
    axis_line_y=p9.element_line(color="black"),
    text=p9.element_text(family="Arial", color="black", size=12),
    title=p9.element_text(size=16),
    axis_title_x=p9.element_text(size=14),
    axis_title_y=p9.element_text(size=14),
    panel_grid=p9.element_blank(),
    panel_background=p9.element_blank(),
)


def boxplot_jitter(
    data: pl.DataFrame,
    y: str | list[str],
    x: str | None = None,
    *,
    color: str | None = None,
    figure_size: tuple[float, float] | None = None,
    cols_as_facet=False,
    dodge=False,
):
    aes_args = {}
    if color:
        aes_args["color"] = color
    if isinstance(y, str):
        ggplot = p9.ggplot(data, p9.aes(x=x, y=y, **aes_args))
        if dodge:
            ggplot = (
                ggplot
                + p9.stat_boxplot(
                    geom="errorbar", position=p9.position_dodge(preserve="single")
                )
                + p9.geom_boxplot(
                    outlier_shape="", position=p9.position_dodge(preserve="single")
                )
                + (
                    p9.geom_jitter(
                        position=p9.position_jitterdodge(
                            jitter_height=0, jitter_width=0.2
                        )
                    )
                )
            )

        else:
            ggplot = (
                ggplot
                + p9.stat_boxplot(geom="errorbar")
                + p9.geom_boxplot(outlier_shape="")
                + p9.geom_jitter(width=0.2, height=0)
            )
        return ggplot + p9.labs(y="", title=y) + p9.theme(figure_size=figure_size)
    elif cols_as_facet:
        return (
            boxplot_jitter(
                data.unpivot(on=y, index=x),
                "value",
                x,
                color=color,
                figure_size=figure_size,
                dodge=dodge,
            )
            + p9.facet_grid(cols="variable")
            + p9.labs(title="")
        )
    else:
        return functools.reduce(
            operator.or_,
            [
                boxplot_jitter(data, y_stat, x, color=color, dodge=dodge)
                + p9.theme(figure_size=figure_size or (6 * len(y), 6))
                for y_stat in y
            ],
        )


#
# MATPLOTLIB
#
def plot_annotated_heatmap_split(
    data: np.ndarray,
    row_names: list[str],
    col_names: list[str],
    annotations: np.ndarray | None = None,
    *,
    cmap: str | matplotlib.colors.Colormap = "Accent",
    vmin: float | None = None,
    vmax: float | None = None,
    draw_triangle_top_left=True,
    x_ticks_top=False,
    annotations_formatter: typing.Callable[..., str] = str,
    annotations_formatter_kwargs: typing.Callable[..., dict[str, typing.Any]],
    legend_entries: dict[str, typing.Any] | None = None,
):
    assert data.ndim == 3 and data.shape[2] == 2, (
        f"data should be of shape [num_rows, num_cols, 2] but is of shape {data.shape}"
    )
    height = data.shape[0] + 1
    width = data.shape[1] + 1
    xs, ys = np.meshgrid(range(width), range(height))

    def triangle_top_left(index: int):
        return (index, index + 1, index + width)

    def triangle_bottom_right(index: int):
        return (index + 1, index + width, index + width + 1)

    def triangle_bottom_left(index: int):
        return (index, index + width, index + width + 1)

    def triangle_top_right(index: int):
        return (index, index + 1, index + width + 1)

    if draw_triangle_top_left:
        get_triangles = lambda index: [
            triangle_top_left(index),
            triangle_bottom_right(index),
        ]
    else:
        get_triangles = lambda index: [
            triangle_bottom_left(index),
            triangle_top_right(index),
        ]

    xs = xs.flatten()
    ys = ys.flatten()
    tris = []
    for y in range(height - 1):
        for x in range(width - 1):
            index = x + y * width
            tris += get_triangles(index)

    if isinstance(cmap, str):
        cmap = matplotlib.colormaps.get_cmap(cmap)
    # plots nans as white triangles so the plot lines up nicely
    cmap.set_bad((1, 1, 1), 1)

    im = plt.tripcolor(xs, ys, tris, data.flatten(), cmap=cmap, vmin=vmin, vmax=vmax)
    if x_ticks_top:
        plt.gca().xaxis.tick_top()
        plt.gca().xaxis.set_label_position("top")

    plt.yticks(np.arange(len(row_names)) + 0.5, row_names)
    plt.xticks(np.arange(len(col_names)) + 0.5, col_names)
    plt.xlim(0, len(col_names))
    plt.ylim(len(row_names), 0)

    if annotations is not None:
        for y in range(height - 1):
            for x in range(width - 1):
                val = annotations[y, x]
                if val is None or np.isnan(val):
                    continue
                im.axes.text(
                    x + 0.5,
                    y + 0.5,
                    annotations_formatter(val),
                    horizontalalignment="center",
                    verticalalignment="center",
                    **annotations_formatter_kwargs(val),
                )

    if legend_entries is not None:
        plt.gca().legend(
            [
                matplotlib.patches.Patch(color=cmap(value))
                for value in legend_entries.values()
            ],
            legend_entries.keys(),
        )


def add_to_legend(
    handles: list[matplotlib.artist.Artist],
    labels: list[str],
    position: typing.Literal["before", "after"] = "after",
):
    legend = plt.gca().legend_
    if legend is not None:
        handles_original = legend.legend_handles
        labels_original = [label.get_text() for label in legend.texts]
    else:
        handles_original = []
        labels_original = []

    if position == "before":
        plt.legend(handles + handles_original, labels + labels_original)
    else:
        plt.legend(handles_original + handles, labels_original + labels)


def hide_axes_top_right(ax: matplotlib.axes.Axes | None = None):
    ax = ax or plt.gca()
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)


def xticks_show_nth(
    n: int, start_offset: int = 0, ax: matplotlib.axes.Axes | None = None
):
    xticks_slice(slice(start_offset, None, n), ax)


def xticks_slice(slice: slice, ax: matplotlib.axes.Axes | None = None):
    ax = ax or plt.gca()
    ticks = ax.get_xticks()
    labels = [label.get_text() for label in ax.get_xticklabels()]
    ax.set_xticks(ticks[slice], labels[slice])


def yticks_show_nth(
    n: int, start_offset: int = 0, ax: matplotlib.axes.Axes | None = None
):
    yticks_slice(slice(start_offset, None, n), ax)


def yticks_slice(slice: slice, ax: matplotlib.axes.Axes | None = None):
    ax = ax or plt.gca()
    ticks = ax.get_yticks()
    labels = [label.get_text() for label in ax.get_yticklabels()]
    ax.set_yticks(ticks[slice], labels[slice])
