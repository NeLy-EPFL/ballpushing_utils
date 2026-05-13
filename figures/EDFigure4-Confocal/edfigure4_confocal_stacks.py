from pathlib import Path
import ants
import cv2
import numpy as np
import yaml
import tifffile
import matplotlib.pyplot as plt
from tqdm import tqdm
from ballpushing_utils.paths import get_cache_dir

regions = ["brain", "vnc"]

# ---------------------------------------------------------------------------
# Data resolution helpers
# ---------------------------------------------------------------------------

# Janelia JRC2018 reference-brain NRRD filenames (exact names as published).
_JRC_FILENAMES = {
    "brain": "JRC2018_FEMALE_38um_iso_16bit.nrrd",
    "vnc": "JRC2018_VNC_FEMALE_4iso.nrrd",
}

# Landing pages for manual download of the Janelia templates.
_JRC_LANDING_URLS = {
    "brain": "https://figshare.com/s/afa673b1dcd163ad8f3f",
    "vnc": "https://figshare.com/s/8103fa90a5cded0509c4",
}

_CONFOCAL_SUBDIR = "confocal"


def _confocal_datasets_dir() -> Path:
    """Return ``<repo>/Datasets/confocal/``, the default local home for the
    Dataverse-downloaded confocal tiff files and ``stack_infos.yaml``."""
    from ballpushing_utils.paths import REPO_DATASETS_DIR

    d = REPO_DATASETS_DIR / _CONFOCAL_SUBDIR
    return d


def _download_confocal_from_dataverse(dest: Path) -> None:
    """Download the confocal-stacks Dataverse archive into *dest*.

    Uses the same unauthenticated Dataverse REST API as ``ballpushing-fetch``.
    All files from :data:`~ballpushing_utils.dataverse_naming.CONFOCAL_DOI`
    are downloaded (the tiff stack files + ``stack_infos.yaml``).
    """
    import json
    import urllib.parse
    import urllib.request
    import shutil
    import tempfile
    import os

    from ballpushing_utils.dataverse_naming import CONFOCAL_DOI
    from ballpushing_utils.dataverse_download import DEFAULT_API_BASE, _USER_AGENT

    dest.mkdir(parents=True, exist_ok=True)

    print(f"Querying Dataverse for confocal-stacks archive ({CONFOCAL_DOI})…")
    api_url = (
        f"{DEFAULT_API_BASE}/api/datasets/:persistentId/?"
        + urllib.parse.urlencode({"persistentId": CONFOCAL_DOI})
    )
    req = urllib.request.Request(
        api_url, headers={"Accept": "*/*", "User-Agent": _USER_AGENT}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.load(resp)

    files = {}
    for entry in payload["data"]["latestVersion"].get("files", []):
        df = entry.get("dataFile", {}) or {}
        name = df.get("filename") or entry.get("label")
        if name:
            files[name] = df.get("id")

    print(f"  Found {len(files)} files.  Downloading to {dest} …")
    for name, file_id in files.items():
        target = dest / name
        if target.exists():
            print(f"  [skip] {name} (already present)")
            continue
        dl_url = f"{DEFAULT_API_BASE}/api/access/datafile/{file_id}"
        dl_req = urllib.request.Request(
            dl_url, headers={"Accept": "*/*", "User-Agent": _USER_AGENT}
        )
        fd, tmp_name = tempfile.mkstemp(prefix=f".{name}.", suffix=".part", dir=dest)
        tmp_path = Path(tmp_name)
        try:
            os.close(fd)
            print(f"  Downloading {name} …")
            with (
                urllib.request.urlopen(dl_req, timeout=600) as resp,
                tmp_path.open("wb") as out,
            ):
                shutil.copyfileobj(resp, out, length=1 << 20)
            tmp_path.replace(target)
            print(f"  [done] {name}")
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
    print("Confocal data download complete.")


def resolve_confocal_dir(*, auto_download: bool = True) -> Path:
    """Return the directory containing the confocal tiff files and YAML.

    Resolution order:

    1. ``BALLPUSHING_TL_CONFOCAL_DIR`` env var (override for non-standard
       mount layouts — set this on lab machines where the NFS export is
       mounted at an unusual prefix).
    2. Hard-coded lab-server path
       ``/mnt/upramdya/data/TL/affordance_confocal_stacks`` — works
       out-of-the-box for EPFL NeLy lab members.
    3. ``<repo>/Datasets/confocal/`` — the default destination for the
       Dataverse download (flat layout: one ``.tiff`` per genotype key,
       plus ``stack_infos.yaml``).
    4. If *auto_download* is ``True`` and none of the above exist, attempt
       to download the archive from Dataverse
       (:data:`~ballpushing_utils.dataverse_naming.CONFOCAL_DOI`) into
       ``<repo>/Datasets/confocal/``.

    Raises ``FileNotFoundError`` if no path resolves and download is
    either disabled or fails.
    """
    import os

    # 1. env-var override
    override = os.environ.get("BALLPUSHING_TL_CONFOCAL_DIR")
    if override:
        p = Path(override).expanduser()
        if p.exists():
            return p

    # 2. hard-coded lab-server path
    lab = Path("/mnt/upramdya/data/TL/affordance_confocal_stacks")
    if lab.exists():
        return lab

    # 3. Datasets/confocal/ with yaml present
    datasets = _confocal_datasets_dir()
    if (datasets / "stack_infos.yaml").exists():
        return datasets

    # 4. auto-download from Dataverse
    if auto_download:
        print(
            "\nConfocal stack data not found locally.  Attempting to download from "
            "Dataverse…\n"
            "  DOI: doi:10.7910/DVN/MY4GN5\n"
            "  Destination: {}\n".format(datasets)
        )
        _download_confocal_from_dataverse(datasets)
        if (datasets / "stack_infos.yaml").exists():
            return datasets

    raise FileNotFoundError(
        "Confocal stack data not found.\n\n"
        "External users: download the confocal dataset from Harvard Dataverse:\n"
        "  DOI: doi:10.7910/DVN/MY4GN5\n"
        "  Direct link: https://doi.org/10.7910/DVN/MY4GN5\n\n"
        "  Place all downloaded files (*.tiff + stack_infos.yaml) in:\n"
        "    {datasets}\n"
        "  Or run:\n"
        "    python edfigure4_confocal_stacks.py\n"
        "  (the script will auto-download on first run if you have internet access)\n\n"
        "Lab members: mount the NFS share and/or set\n"
        "  export BALLPUSHING_TL_CONFOCAL_DIR=/your/mount/path\n".format(
            datasets=datasets
        )
    )


def resolve_stack_tiff(raw_dir: Path, genotype_key: str, stack_info: dict) -> Path:
    """Resolve the .tiff file for a given genotype.

    Tries two layouts:

    1. **Lab-server layout** — ``raw_dir / stack_info["path"]`` (nested,
       date-prefixed directory as stored on the EPFL NFS share).
    2. **Dataverse flat layout** — ``raw_dir / "{genotype_key}.tiff"`` (the
       name produced by ``prepare_flat_copy.py``'s date-stripping logic).

    Raises ``FileNotFoundError`` if neither exists.
    """
    # 1. Lab-server path from the YAML "path" field
    server_path = raw_dir / stack_info["path"]
    if server_path.exists():
        return server_path

    # 2. Flat Dataverse layout
    # need to strip date from path eg.
    # 250729_MB247-Gal4xUAS-spm/image.tif -> MB247-Gal4xUAS-spm
    filename = stack_info["path"].split("/")[0][7:]
    flat_path = raw_dir / f"{filename}.tiff"
    if flat_path.exists():
        return flat_path

    raise FileNotFoundError(
        f"Could not find tiff for genotype '{genotype_key}'.\n"
        f"  Tried (lab-server path): {server_path}\n"
        f"  Tried (Dataverse flat): {flat_path}\n"
        f"Download the confocal dataset from: https://doi.org/10.7910/DVN/MY4GN5"
    )


def resolve_jrc_nrrd_paths(raw_dir: Path) -> dict[str, Path]:
    """Return ``{region: Path}`` for the JRC2018 reference-brain NRRDs.

    Resolution order per region:

    1. ``BALLPUSHING_JRC2018_DIR`` env var (explicit override).
    2. ``raw_dir / "jrc2018" / <filename>`` — lab-server layout (the NRRDs
       live alongside the confocal stacks on the NFS share).
    3. ``<repo>/.cache/registration/jrc2018_src / <filename>`` — the local
       cache used when running off the Dataverse download.

    If any NRRD is not found after checking all locations, ``FileNotFoundError``
    is raised with instructions pointing to the Janelia download page.
    """
    import os

    jrc_src_cache = get_cache_dir() / "registration" / "jrc2018_src"

    search_dirs: list[Path] = []
    override = os.environ.get("BALLPUSHING_JRC2018_DIR")
    if override:
        search_dirs.append(Path(override).expanduser())
    search_dirs.append(raw_dir / "jrc2018")
    search_dirs.append(jrc_src_cache)

    paths: dict[str, Path] = {}
    missing: list[tuple[str, str]] = []
    for region, filename in _JRC_FILENAMES.items():
        found = None
        for d in search_dirs:
            candidate = d / filename
            if candidate.exists():
                found = candidate
                break
        if found is not None:
            paths[region] = found
        else:
            missing.append((region, filename))

    if missing:
        missing_str = "\n".join(f"  {region}: {fn}" for region, fn in missing)
        search_str = "\n".join(f"  {d}" for d in search_dirs)
        raise FileNotFoundError(
            f"Janelia JRC2018 reference-brain templates not found.\n\n"
            f"Missing files:\n{missing_str}\n\n"
            f"Searched in:\n{search_str}\n\n"
            f"To fix — download the templates from Janelia:\n"
            f"  Brain : {_JRC_LANDING_URLS['brain']}\n"
            f"  VNC   : {_JRC_LANDING_URLS['vnc']}\n\n"
            f"Place the downloaded NRRD files in one of the search directories\n"
            f"above, OR set the environment variable:\n"
            f"  export BALLPUSHING_JRC2018_DIR=/path/to/directory/containing/nrrds\n"
        )

    return paths


# ---------------------------------------------------------------------------
# Module-level data resolution
# ---------------------------------------------------------------------------

raw_dir = resolve_confocal_dir()
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


def registration_exists(directory: str | Path):
    directory = Path(directory)
    ants_output_filenames = [
        "0GenericAffine.mat",
        "1InverseWarp.nii.gz",
        "1Warp.nii.gz",
    ]
    return all((directory / i).exists() for i in ants_output_filenames)


def antsread(path):
    path = Path(path).as_posix()
    assert path.endswith(
        (".nii", ".nii.gz", ".nrrd")
    ), "Only support .nii, .nii.gz, .nrrd"
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


def crop_stack(genotype_key, stack_info):
    warp_kws = {
        "flags": cv2.INTER_LANCZOS4,
        "borderMode": cv2.BORDER_CONSTANT,
        "borderValue": 0,
    }
    stack = tifffile.imread(resolve_stack_tiff(raw_dir, genotype_key, stack_info))
    for region in ["brain", "vnc"]:
        size = crop_size[region]
        p0 = stack_info[region]["p0"]
        p1 = stack_info[region]["p1"]
        pad = stack_info[region].get("pad", default_pad[region])
        matrix = get_cropping_affine_matrix(size, p0, p1, pad)
        s = np.sqrt(np.linalg.det(matrix[:, :2]))
        spacing = (stack_info["dx"] / s, stack_info["dx"] / s, stack_info["dz"] * 2)
        for channel in range(2):
            cropped = np.array(
                [
                    cv2.warpAffine(im, matrix, size, **warp_kws)
                    for im in stack[:, channel]
                ]
            )
            if not stack_info[region]["ventral"]:
                cropped = cropped[::-1, ..., ::-1]
            yield region, channel, ants.from_numpy(cropped.T, spacing=spacing)


def crop_stacks(stack_infos: dict[str, dict], stacks_dir: Path):
    for genotype, stack_info in tqdm(stack_infos.items()):
        if all(
            (stacks_dir / genotype / r / f"{c}g.nii.gz").exists()
            for r in regions
            for c in "fb"
        ):
            print(f"All crops for {genotype} exist, skipping")
            continue
        for region, channel, cropped in crop_stack(genotype, stack_info):
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
            stacks = [
                antsread(stacks_dir / genotype / region / "bg.nii.gz")
                for genotype in genotypes
            ]
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
            antsread(jrc_raw_paths[region]).resample_image(
                jrc_target_spacing[region], interp_type=3
            ).to_file(fixed_path.as_posix())
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
                dst = (
                    stacks_dir
                    / genotype
                    / region
                    / f"{moving_path.stem[:2]}_to_jrc.nii.gz"
                )

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

    vmax_fg = stack_info.get(
        "vmax0", max(brain_fg.max(), vnc_fg.max())
    ) * stack_info.get("s0", 1)
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
        ax.imshow(
            [[0]],
            extent=(-size[0] / 2, size[0] / 2, vnc_extent[2], vnc_extent[3]),
            cmap="gray",
        )
        ax.imshow(vnc, extent=vnc_extent)
        ax.set_xlim(-size[0] / 2, size[0] / 2)
        ax.set_ylim(size[1], 0)
        ax.set_visible_sides("")
        ax.set_title(name, fontsize=7)
        length = 50 / 0.76
        ax.add_scale_bars(
            scale_bar_pad - size[0] / 2,
            brain_extent[2] - scale_bar_pad,
            length,
            0,
            c="w",
            xlabel="",
            fmt="",
        )
        ax.add_scale_bars(
            scale_bar_pad - size[0] / 2,
            vnc_extent[2] - scale_bar_pad,
            length,
            0,
            c="w",
            xlabel="",
            fmt="",
        )
    return g


if __name__ == "__main__":
    from ballpushing_utils.paths import figure_output_dir

    crop_stacks(stack_infos, stacks_dir)
    build_lab_templates(stacks_dir, genotypes_for_building_templates, lab_dir)
    jrc_nrrd_paths = resolve_jrc_nrrd_paths(raw_dir)
    register_lab_to_jrc(
        lab_dir, jrc_dir, jrc_nrrd_paths, jrc_target_spacing, lab_to_jrc_dir
    )
    register_and_transform_stacks(
        stack_infos, stacks_dir, lab_dir, jrc_dir, lab_to_jrc_dir
    )
    images = get_combined_images(jrc_dir, stack_infos, stacks_dir)
    g = plot_confocal_stacks(images)
    out_dir = figure_output_dir("EDFigure4", __file__)
    g.savefig(out_dir / "edfigure4_confocal_stacks.pdf")
    plt.close(g.fig)
