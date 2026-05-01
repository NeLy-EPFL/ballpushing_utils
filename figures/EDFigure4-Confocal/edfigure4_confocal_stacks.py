from pathlib import Path
import ants
import cv2
import numpy as np
import yaml
import tifffile
import matplotlib.pyplot as plt
from tqdm import tqdm
from ballpushing_utils.paths import get_cache_dir, require_path

regions = ["brain", "vnc"]

# Co-author 2's lab-share path. Set BALLPUSHING_TL_CONFOCAL_DIR to override
# on machines with a different mount layout (lab members commonly mount
# the same NFS export at different prefixes — see paths.require_path).
raw_dir = require_path(
    "/mnt/upramdya/data/TL/affordance_confocal_stacks",
    description="confocal stack directory (ED Fig. 4)",
    env_var="BALLPUSHING_TL_CONFOCAL_DIR",
)
stack_infos_path = raw_dir / "stack_infos.yaml"
with open(stack_infos_path) as f:
    stack_infos = yaml.safe_load(f)

cache_dir = get_cache_dir() / "registration"
stacks_dir = cache_dir / "stacks"
lab_dir = cache_dir / "lab"
jrc_dir = cache_dir / "jrc"
lab_to_jrc_dir = cache_dir / "lab_to_jrc"

crop_size = {"brain": (826, 384), "vnc": (330, 671)}
default_pad = {"brain": (0.8, 0.8), "vnc": (1, 0.6)}

genotypes_for_building_templates = {
    "brain": [
        "mb247",
        "empty-split-gal4",
        "empty-gal4",
        "ir8a",
    ],
    "vnc": [
        "empty-gal4",
        "ir8a",
        "ir25a",
        "mb247",
        "lc10bc",
    ],
}

jrc_target_spacing = {"brain": (0.76, 0.76, 0.76), "vnc": (0.8, 0.8, 0.8)}
jrc_nrrd_paths = {
    "brain": raw_dir / "jrc2018" / "JRC2018_FEMALE_38um_iso_16bit.nrrd",
    "vnc": raw_dir / "jrc2018" / "JRC2018_VNC_FEMALE_4iso.nrrd",
}


def registration_exists(directory: str | Path):
    directory = Path(directory)
    ants_output_filenames = ["0GenericAffine.mat", "1InverseWarp.nii.gz", "1Warp.nii.gz"]
    return all((directory / i).exists() for i in ants_output_filenames)


def antsread(path):
    path = Path(path).as_posix()
    assert path.endswith((".nii", ".nii.gz", ".nrrd")), "Only support .nii, .nii.gz, .nrrd"
    return ants.image_read(path)


def read_max_proj(path):
    return antsread(path).numpy().T.max(0)


def read_mean_proj(path):
    return antsread(path).numpy().T.mean(0)


def winsorize(im, vmin, vmax):
    return np.clip((im - vmin) / (vmax - vmin), 0, 1)


def combine_gray(fg, bg, lims_fg, lims_bg):
    fg = np.power(winsorize(fg, *lims_fg), 1 / 1.2)
    bg = np.power(winsorize(bg, *lims_bg), 1 / 1.2)
    zeros = np.zeros_like(fg)
    bg_rgb = np.stack([bg, bg, bg], axis=-1) / 3
    fg_rgb = np.stack([zeros, fg, zeros], axis=-1)
    return np.clip(bg_rgb + fg_rgb, 0, 1)


def combine_magenta(fg, bg, lims_fg, lims_bg):
    fg = np.power(winsorize(fg, *lims_fg), 1 / 1.2)
    bg = np.power(winsorize(bg, *lims_bg), 1 / 1.2)
    zeros = np.zeros_like(fg)
    bg_rgb = np.stack([bg, zeros, bg], axis=-1) * 0.5
    fg_rgb = np.stack([zeros, fg, zeros], axis=-1)
    return np.clip(bg_rgb + fg_rgb, 0, 1)


def get_cropping_affine_matrix(size, p0, p1, pad):
    w, h = size
    pad0, pad1 = pad
    p0 = np.asarray(p0)
    p1 = np.asarray(p1)
    m = (p0 + p1) / 2
    u = p1 - p0
    d = np.linalg.norm(u)
    u = u / d
    v = np.array([u[1], -u[0]])

    p0 = m - (d * pad0) * u - (d * (pad0 + pad1) * (w / h) / 2) * v
    p1 = p0 + (d * (pad0 + pad1)) * u
    p2 = p0 + (d * (pad0 + pad1) * (w / h)) * v

    src = np.array([p0, p1, p2], dtype=np.float32)
    dst = np.array([[0, 0], [0, h], [w, 0]], dtype=np.float32)
    mat = cv2.getAffineTransform(src, dst)
    return mat


def crop_stack(stack_info):
    warp_kws = {"flags": cv2.INTER_LANCZOS4, "borderMode": cv2.BORDER_CONSTANT, "borderValue": 0}
    stack = tifffile.imread(raw_dir / stack_info["path"])
    for region in ["brain", "vnc"]:
        size = crop_size[region]
        p0 = stack_info[region]["p0"]
        p1 = stack_info[region]["p1"]
        pad = stack_info[region].get("pad", default_pad[region])
        matrix = get_cropping_affine_matrix(size, p0, p1, pad)
        s = np.sqrt(np.linalg.det(matrix[:, :2]))
        spacing = (stack_info["dx"] / s, stack_info["dx"] / s, stack_info["dz"] * 2)
        for channel in range(2):
            cropped = np.array([cv2.warpAffine(im, matrix, size, **warp_kws) for im in stack[:, channel]])
            if not stack_info[region]["ventral"]:
                cropped = cropped[::-1, ..., ::-1]
            yield region, channel, ants.from_numpy(cropped.T, spacing=spacing)


def crop_stacks(stack_infos: dict[str, dict], stacks_dir: Path):
    for genotype, stack_info in tqdm(stack_infos.items()):
        if all((stacks_dir / genotype / r / f"{c}g.nii.gz").exists() for r in regions for c in "fb"):
            print(f"All crops for {genotype} exist, skipping")
            continue
        for region, channel, cropped in crop_stack(stack_info):
            save_dir = stacks_dir / genotype / region
            save_dir.mkdir(parents=True, exist_ok=True)
            cropped.to_file(save_dir / f"{'fb'[channel]}g.nii.gz")


def build_lab_templates(
    stacks_dir: Path,
    genotypes_for_building_templates: dict[str, list[str]],
    lab_dir: Path,
):
    from tempfile import TemporaryDirectory

    for region, genotypes in genotypes_for_building_templates.items():
        output_path = lab_dir / f"{region}.nii.gz"
        if output_path.exists():
            print(f"{output_path} exists, skipping")
            continue

        with TemporaryDirectory() as temp_dir:
            stacks = [antsread(stacks_dir / genotype / region / "bg.nii.gz") for genotype in genotypes]
            stacks = sum([[stack, stack.reflect_image(axis=0)] for stack in stacks], [])
            output_path.parent.mkdir(exist_ok=True, parents=True)
            ants.build_template(
                image_list=stacks,
                output_dir=temp_dir,
            ).to_filename(output_path.as_posix())


def register_lab_to_jrc(
    lab_dir: Path,
    jrc_dir: Path,
    jrc_raw_paths: dict[str, Path],
    jrc_target_spacing: dict[str, tuple[float, float, float]],
    lab_to_jrc_dir: Path,
):
    for region in regions:
        fixed_path = jrc_dir / f"{region}.nii.gz"
        fixed_path.parent.mkdir(exist_ok=True, parents=True)

        if not fixed_path.exists():
            antsread(jrc_raw_paths[region]).resample_image(jrc_target_spacing[region], interp_type=3).to_file(
                fixed_path.as_posix()
            )
        else:
            print(f"{fixed_path} exists, skipping")

        fix_path = jrc_dir / f"{region}.nii.gz"
        mov_path = lab_dir / f"{region}.nii.gz"
        to_lab_dir = lab_to_jrc_dir / region

        if registration_exists(to_lab_dir):
            print(f"{to_lab_dir} exists, skipping")
            continue

        to_lab_dir.mkdir(parents=True, exist_ok=True)

        ants.registration(
            fixed=antsread(fix_path),
            moving=antsread(mov_path),
            outprefix=to_lab_dir.as_posix() + "/",
            verbose=True,
        )


def register_and_transform_stacks(
    stack_infos: dict[str, dict],
    stacks_dir: Path,
    lab_dir: Path,
    jrc_dir: Path,
    lab_to_jrc_dir: Path,
):
    for region in regions:
        lab = antsread(lab_dir / f"{region}.nii.gz")
        jrc = antsread(jrc_dir / f"{region}.nii.gz")

        for genotype, stack_info in stack_infos.items():
            bg_path = stacks_dir / genotype / region / "bg.nii.gz"
            to_lab_dir = bg_path.parent / "to_lab"

            if registration_exists(to_lab_dir):
                print(f"{to_lab_dir} exists, skipping")
            else:
                to_lab_dir.mkdir(exist_ok=True, parents=True)
                ants.registration(
                    fixed=lab,
                    moving=antsread(bg_path),
                    outprefix=to_lab_dir.as_posix() + "/",
                )

            transform_list = [
                lab_to_jrc_dir / region / "1Warp.nii.gz",
                lab_to_jrc_dir / region / "0GenericAffine.mat",
                to_lab_dir / "1Warp.nii.gz",
                to_lab_dir / "0GenericAffine.mat",
            ]
            transform_list = [i.as_posix() for i in transform_list]

            fg_path = stacks_dir / genotype / region / "fg.nii.gz"
            for moving_path in [bg_path, fg_path]:
                dst = stacks_dir / genotype / region / f"{moving_path.stem[:2]}_to_jrc.nii.gz"

                if dst.exists():
                    print(f"{dst} exists, skipping")
                else:
                    ants.apply_transforms(
                        fixed=jrc,
                        moving=antsread(moving_path),
                        transformlist=transform_list,
                        interpolator="lanczosWindowedSinc",
                    ).to_file(dst.as_posix())


def read_jrc(jrc_dir):
    projs = [read_mean_proj(jrc_dir / f"{region}.nii.gz") for region in regions]
    vmax = max(proj.max() for proj in projs)
    for proj in projs:
        proj[proj < 0] = 0
        proj /= vmax
    return tuple(np.stack((proj,) * 3, axis=-1) for proj in projs)


def read_stack(genotype_dir, stack_info):
    brain_fg = read_max_proj(genotype_dir / "brain" / "fg_to_jrc.nii.gz")
    brain_bg = read_max_proj(genotype_dir / "brain" / "bg_to_jrc.nii.gz")
    vnc_fg = read_max_proj(genotype_dir / "vnc" / "fg_to_jrc.nii.gz")
    vnc_bg = read_max_proj(genotype_dir / "vnc" / "bg_to_jrc.nii.gz")

    vmax_fg = stack_info.get("vmax0", max(brain_fg.max(), vnc_fg.max())) * stack_info.get("s0", 1)
    vmax_bg = np.concatenate([vnc_bg.ravel(), brain_bg.ravel()]).mean() * 4
    lims_fg = (vmax_fg * 0.05, vmax_fg)
    lims_bg = (vmax_bg * 0.05, vmax_bg)
    return (
        combine_gray(brain_fg, brain_bg, lims_fg, lims_bg),
        combine_gray(vnc_fg, vnc_bg, lims_fg, lims_bg),
    )


def get_combined_images(jrc_dir, stack_infos, stacks_dir):
    images = {
        "JRC2018 female": read_jrc(jrc_dir),
    }
    for genotype, stack_info in stack_infos.items():
        images[stack_info["name"]] = read_stack(stacks_dir / genotype, stack_info)
    return images


def plot_confocal_stacks(
    images: dict[str, tuple[np.ndarray, np.ndarray]],
    scale=0.1,
    scale_bar_pad=20,
    n_cols=5,
    brain_vnc_gap=10,
):
    from mplex import Grid

    n_images = len(images)
    n_rows = int(np.ceil(n_images / n_cols))
    brain_size = next(iter(images.values()))[0].shape[:2][::-1]
    vnc_size = next(iter(images.values()))[1].shape[:2][::-1]
    size = np.array(
        [
            max(brain_size[0], vnc_size[0]),
            brain_size[1] + brain_vnc_gap + vnc_size[1],
        ]
    )
    brain_extent = (-brain_size[0] / 2, brain_size[0] / 2, brain_size[1], 0)
    vnc_extent = (
        -vnc_size[0] / 2,
        vnc_size[0] / 2,
        brain_size[1] + brain_vnc_gap + vnc_size[1],
        brain_size[1] + brain_vnc_gap,
    )
    g = Grid(scale * size, (n_rows, n_cols), facecolor="w", space=(4, 12))
    for ax, (name, (brain, vnc)) in zip(g.axs.ravel(), images.items()):
        ax.imshow(brain, extent=brain_extent)
        ax.imshow([[0]], extent=(-size[0] / 2, size[0] / 2, vnc_extent[2], vnc_extent[3]), cmap="gray")
        ax.imshow(vnc, extent=vnc_extent)
        ax.set_xlim(-size[0] / 2, size[0] / 2)
        ax.set_ylim(size[1], 0)
        ax.set_visible_sides("")
        ax.set_title(name, fontsize=7)
        length = 50 / 0.76
        ax.add_scale_bars(
            scale_bar_pad - size[0] / 2, brain_extent[2] - scale_bar_pad, length, 0, c="w", xlabel="", fmt=""
        )
        ax.add_scale_bars(
            scale_bar_pad - size[0] / 2, vnc_extent[2] - scale_bar_pad, length, 0, c="w", xlabel="", fmt=""
        )
    return g


if __name__ == "__main__":
    from ballpushing_utils.paths import figure_output_dir

    crop_stacks(stack_infos, stacks_dir)
    build_lab_templates(stacks_dir, genotypes_for_building_templates, lab_dir)
    register_lab_to_jrc(lab_dir, jrc_dir, jrc_nrrd_paths, jrc_target_spacing, lab_to_jrc_dir)
    register_and_transform_stacks(stack_infos, stacks_dir, lab_dir, jrc_dir, lab_to_jrc_dir)
    images = get_combined_images(jrc_dir, stack_infos, stacks_dir)
    g = plot_confocal_stacks(images)
    out_dir = figure_output_dir("EDFigure4", __file__)
    g.savefig(out_dir / "edfigure4_confocal_stacks.pdf")
    plt.close(g.fig)
