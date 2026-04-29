import numpy as np


def arrow(p1, p2, ax, *args, **kwargs):
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    return ax.arrow(x1, y1, dx, dy, *args, **kwargs)


def fit_line_orthogonal(*points):
    pts = np.asarray(points)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    U, S, Vt = np.linalg.svd(centered)
    direction = Vt[0]
    return centroid, direction / np.linalg.norm(direction)


def plot_kinematic_features():
    from matplotlib.patches import Circle, Wedge
    from mplex import Grid
    from flyplotlib import add_fly

    arrow_kws = dict(
        width=0.06,
        overhang=0.25,
        fc="r",
        ec="none",
        head_width=0.4,
        head_length=0.4,
        length_includes_head=True,
        zorder=100,
    )
    line_kws = dict(
        color="black",
        lw=0.5,
        ls="-",
    )
    dash_kws = dict(
        color="black",
        lw=0.3,
        ls="--",
    )
    circle_kws = dict(
        radius=0.9,
        color=(0.75,) * 3,
        zorder=-9999,
        lw=0,
        fill=True,
    )
    scatter_kws = dict(
        s=20,
        color="black",
        marker=".",
        lw=0,
    )

    g = Grid((80, 20), (4, 1), dpi=600, facecolor="w", space=0)

    for ax in g.axs.ravel():
        add_fly(svg_path="pushing.svg", rotation=0, ax=ax)
        ax.set_aspect(1)
        ax.axis("off")

    ax = g[1, 0]
    thorax_xy = np.array([-0.2, 0.3])
    ball_xy = np.array([2.4, -0.1])
    ax.add_patch(Circle(ball_xy, **circle_kws))
    ax.scatter(*ball_xy, **scatter_kws)
    ax.scatter(*thorax_xy, **scatter_kws)
    y0 = 1
    ax.plot([thorax_xy[0], thorax_xy[0]], [y0, thorax_xy[1]], **dash_kws)
    ax.plot([ball_xy[0], ball_xy[0]], [y0, ball_xy[1]], **dash_kws)
    arrow((thorax_xy[0], y0), (ball_xy[0], y0), **arrow_kws, ax=ax)

    ax = g[0, 0]
    ball2_xy = np.array([3.6, -0.1])

    ax.add_patch(Circle(ball_xy, alpha=0.3, **circle_kws))
    ax.add_patch(Circle(ball2_xy, alpha=0.8, **circle_kws))
    ax.scatter(*ball_xy, **scatter_kws)
    ax.scatter(*ball2_xy, **scatter_kws)

    ax.plot([ball_xy[0], ball_xy[0]], [y0, ball_xy[1]], **dash_kws)
    ax.plot([ball2_xy[0], ball2_xy[0]], [y0, ball2_xy[1]], **dash_kws)
    arrow((ball_xy[0], y0), (ball2_xy[0], y0), **arrow_kws, ax=ax)

    abdomen_xy = np.array([-1.9, -0.55])
    head_xy = np.array([0.6, 0.5])
    ax = g[3, 0]
    ax.add_patch(Circle(ball_xy, **circle_kws))
    p0, u = fit_line_orthogonal(head_xy, thorax_xy, abdomen_xy)

    ax.plot(*np.transpose([thorax_xy - 1.5 * u, thorax_xy + 1.7 * u]), **line_kws)
    ax.plot(*np.transpose([thorax_xy, thorax_xy + (1.5, 0)]), **line_kws)
    ax.add_artist(Wedge(thorax_xy, 0.8, 0, np.angle(-u @ (1, 1j), deg=True), alpha=0.5, color="r"))

    ax = g[2, 0]
    ax.add_patch(Circle(ball_xy, **circle_kws))
    lf_xy = np.array([1.9, 0.65])
    rf_xy = np.array([1.69, -0.37])
    midpint_xy = (lf_xy + rf_xy) / 2

    ax.scatter(*lf_xy, **dict(scatter_kws, s=5), alpha=0.5)
    ax.scatter(*rf_xy, **dict(scatter_kws, s=5), alpha=0.5)
    ax.scatter(*midpint_xy, **dict(scatter_kws, s=20))
    ax.plot(*np.transpose([lf_xy, rf_xy]), **dict(line_kws, ls="-"), alpha=0.5)

    p1 = thorax_xy + u @ (midpint_xy - thorax_xy) * u
    ax.plot(*np.transpose([p1, thorax_xy + 1.7 * u]), **line_kws)
    ax.scatter(*p1, **dict(scatter_kws, s=20))
    v = u[::-1] * (1, -1) * 0.4
    arrow(thorax_xy + v, p1 + v, **arrow_kws, ax=ax)
    ax.plot(*np.transpose([midpint_xy, p1 + v]), **dash_kws)
    ax.plot(*np.transpose([thorax_xy, thorax_xy + v]), **dash_kws)
    ax.scatter(*thorax_xy, **scatter_kws)

    return g


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from ballpushing_utils.paths import figure_output_dir

    out_dir = figure_output_dir("Figure3", __file__)
    g = plot_kinematic_features()
    g.savefig(out_dir / "kinematic_features.pdf")
    plt.close(g.fig)
