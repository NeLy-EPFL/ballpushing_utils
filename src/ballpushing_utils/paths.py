"""Central resolver for filesystem paths used by the paper scripts.

The ballpushing_utils figure and analysis scripts reference large datasets
(feathers, videos, pooled coordinates) that live outside the repository, and
they write figures/tables/stats to a shared output tree. To keep the scripts
portable across machines, all paths flow through this module.

Environment variables
---------------------
``BALLPUSHING_DATA_ROOT``
    Root directory containing datasets, videos, and region maps. Default:
    ``/mnt/upramdya_data/MD`` (the EPFL NeLy lab share used during paper
    development). Override per machine (``export BALLPUSHING_DATA_ROOT=...``)
    or via a :file:`.env` file loaded before running a script.

``BALLPUSHING_FIGURES_ROOT``
    Root directory for generated figures and stats. Default:
    ``$BALLPUSHING_DATA_ROOT/Affordance_Figures``.

Typical usage
-------------
::

    from ballpushing_utils.paths import data_root, figures_root, figure_output_dir

    DATASET = data_root() / "MagnetBlock/Datasets/.../pooled_summary.feather"
    OUTPUT_DIR = figure_output_dir("Figure2", __file__)

:func:`figure_output_dir` mirrors the idiom used in every figure script so
that outputs land under ``<figures_root>/<figure_name>/<script_stem>/``.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "data_root",
    "figures_root",
    "dataset",
    "detect_layout",
    "figure_output_dir",
    "get_cache_dir",
    "find_feather",
    "load_dotenv",
    "missing_data_message",
    "require_path",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_FIGURES_ROOT_NAME",
    "SAMPLE_DATA_RELATIVE",
]

#: Path (repo-relative) to the bundled sample fixtures. Used by the
#: friendly error breadcrumb so users with no server / no Dataverse
#: download still have an offline path to try.
SAMPLE_DATA_RELATIVE = "tests/fixtures/sample_data"

#: Default root if ``BALLPUSHING_DATA_ROOT`` is unset. This matches the path
#: used during paper development and keeps existing scripts working.
DEFAULT_DATA_ROOT = Path("/mnt/upramdya_data/MD")

#: Default subdirectory name under :func:`data_root` for generated figures
#: when ``BALLPUSHING_FIGURES_ROOT`` is unset.
DEFAULT_FIGURES_ROOT_NAME = "Affordance_Figures"


def _env_path(name: str) -> Path | None:
    """Read an env var and return it as a :class:`~pathlib.Path`, or ``None``."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    return Path(raw).expanduser()


def data_root() -> Path:
    """Return the root directory for datasets.

    Resolved from ``BALLPUSHING_DATA_ROOT`` if set, otherwise falls back to
    :data:`DEFAULT_DATA_ROOT`. The directory is **not** created automatically —
    scripts reading data should either find it there or fail loudly.
    """
    return _env_path("BALLPUSHING_DATA_ROOT") or DEFAULT_DATA_ROOT


def figures_root() -> Path:
    """Return the root directory for figure and stats outputs.

    Resolved from ``BALLPUSHING_FIGURES_ROOT`` if set; otherwise
    ``data_root() / DEFAULT_FIGURES_ROOT_NAME``.
    """
    return _env_path("BALLPUSHING_FIGURES_ROOT") or (data_root() / DEFAULT_FIGURES_ROOT_NAME)


def dataset(relative_path: str | os.PathLike[str]) -> Path:
    """Resolve a dataset path relative to :func:`data_root`.

    ``relative_path`` is appended to the data root and returned as an absolute
    :class:`~pathlib.Path`. Accepts any path-like.
    """
    return data_root() / Path(relative_path)


def figure_output_dir(
    figure_name: str,
    script_file: str | os.PathLike[str] | None = None,
    *,
    create: bool = True,
) -> Path:
    """Build and (optionally) create an output directory for a figure script.

    The resulting layout is
    ``figures_root()/<figure_name>/<Path(script_file).stem>/``. If
    ``script_file`` is ``None`` the subdirectory named after the script is
    omitted. Pass ``__file__`` from the calling script.
    """
    base = figures_root() / figure_name
    if script_file is not None:
        base = base / Path(script_file).stem
    if create:
        base.mkdir(parents=True, exist_ok=True)
    return base


def get_cache_dir() -> Path:
    """Return the project-level cache directory.

    The cache lives at ``<repo_root>/.cache`` so scripts run from any working
    directory share the same cached artifacts.
    """
    cache_dir = Path(__file__).resolve().parents[2] / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def find_feather(relative_or_absolute: str | os.PathLike[str]) -> Path | None:
    """Locate a feather (or any data file) the figure scripts ask for.

    Resolution order:

    1. ``relative_or_absolute`` taken at face value (absolute path or
       relative to the cwd) — the historical behaviour. If it resolves
       to an existing file, return it.
    2. ``BALLPUSHING_DATA_ROOT / relative_or_absolute`` — the Dataverse
       archive's canonical location. If a user extracted the whole
       archive, this is what figures already use today.
    3. Each path in the colon-separated ``BALLPUSHING_FEATHER_SEARCH``
       env var, joined with the *basename* of the requested feather.
       Use this when you've downloaded a single feather from the
       Dataverse and dropped it in a flat directory: set
       ``BALLPUSHING_FEATHER_SEARCH=/path/to/downloads`` and figure
       scripts find the file by name.
    4. Recursive ``rglob`` over ``BALLPUSHING_DATA_ROOT`` for the
       basename. A *single* match is returned; multiple matches return
       ``None`` (callers will produce the breadcrumb error so the user
       can disambiguate by setting ``BALLPUSHING_FEATHER_SEARCH`` more
       narrowly).

    Returns
    -------
    Path | None
        Resolved path on success; ``None`` if no candidate was found
        or the search was ambiguous. Callers (see
        :func:`ballpushing_utils.compat.read_feather`) raise the
        breadcrumb error in the ``None`` case so users get a clear
        next-step message.
    """
    rel = Path(relative_or_absolute)

    # Step 1: literal path or cwd-relative.
    if rel.exists():
        return rel.resolve()

    # Step 2: under BALLPUSHING_DATA_ROOT (existing dataset() semantics).
    primary = data_root() / rel
    if primary.exists():
        return primary.resolve()

    basename = rel.name

    # Step 3: explicit search dirs.
    search_env = os.environ.get("BALLPUSHING_FEATHER_SEARCH", "")
    for raw_dir in search_env.split(os.pathsep):
        raw_dir = raw_dir.strip()
        if not raw_dir:
            continue
        candidate = Path(raw_dir).expanduser() / basename
        if candidate.exists():
            return candidate.resolve()

    # Step 4: recursive search under BALLPUSHING_DATA_ROOT.
    root = data_root()
    if root.is_dir():
        matches = list(root.rglob(basename))
        if len(matches) == 1:
            return matches[0].resolve()
        # Ambiguous or empty — fall through to None so the caller can
        # surface a breadcrumb (or the basename-collision warning).
        if len(matches) > 1:
            import logging

            logging.warning(
                f"find_feather({relative_or_absolute!r}): {len(matches)} "
                f"files matched basename {basename!r} under {root}. "
                f"Set BALLPUSHING_FEATHER_SEARCH to a more specific dir, "
                f"or place the feather at {primary} to disambiguate."
            )

    return None


def detect_layout(root: str | os.PathLike[str]) -> str | None:
    """Classify a directory tree as ``"server"``, ``"dataverse"``, or ``None``.

    Used by :func:`missing_data_message` and the dataset builder to give
    the caller a useful hint about which mode to invoke. The detection
    is deliberately cheap — it scans only the first few levels of the
    tree and returns as soon as either signature is found.

    - ``"server"``: at least one descendant directly contains a
      ``Metadata.json`` file (the canonical on-rig experiment folder).
    - ``"dataverse"``: at least one corridor matches
      :func:`ballpushing_utils.dataverse.is_dataverse_layout` — i.e. it
      has a ``*ball*.h5`` HDF5 track and its grandparent matches the
      ``YYMMDD[-N]`` date pattern with no parent ``Metadata.json``.
    - ``None``: neither signature found within the first ~5 levels of
      the tree.
    """
    root = Path(root)
    if not root.is_dir():
        return None

    # Quick win: the root itself looks like an experiment folder.
    if (root / "Metadata.json").exists():
        return "server"

    # Bounded BFS — we stop the first time we see either signature, and
    # cap the depth so a misconfigured root doesn't hang the walker.
    max_depth = 5
    queue: list[tuple[Path, int]] = [(root, 0)]
    seen_h5_corridor: Path | None = None
    while queue:
        current, depth = queue.pop()
        if depth > max_depth:
            continue
        try:
            children = list(current.iterdir())
        except (PermissionError, OSError):
            continue
        for child in children:
            if not child.is_dir():
                continue
            if (child / "Metadata.json").exists():
                return "server"
            # Cheap probe for a Dataverse-style corridor (has *ball*.h5).
            # We defer the full ``is_dataverse_layout`` confirmation
            # until we've finished checking for ``Metadata.json``, which
            # is the preferred signature.
            if seen_h5_corridor is None and any(child.glob("*ball*.h5")):
                seen_h5_corridor = child
            queue.append((child, depth + 1))

    if seen_h5_corridor is not None:
        # Confirm with the strict Dataverse heuristic inlined here to
        # avoid pulling in dataverse.py from this low-level module:
        # date-shaped grandparent + no parent Metadata.json.
        import re

        date_re = re.compile(r"^\d{6}(?:-\d+)?$")
        grandparent = seen_h5_corridor.parent.parent
        if (
            not (grandparent / "Metadata.json").exists()
            and date_re.match(grandparent.name)
        ):
            return "dataverse"

    return None


def missing_data_message(
    requested: str | os.PathLike[str] | None = None,
    *,
    context: str = "data",
) -> str:
    """Build the standard three-option breadcrumb shown when data is missing.

    The message lists the three sources ballpushing_utils knows how to
    consume — the EPFL lab share, the published Dataverse archives, and
    the bundled sample fixtures — and tells the caller what to do for
    each. Keep it consistent across the package so users see the same
    pointer no matter which surface raises (figure script, dataset
    builder, notebook).

    Parameters
    ----------
    requested:
        Optional path that triggered the breadcrumb (e.g. the feather
        the figure script was trying to read). Included verbatim at the
        top of the message when supplied.
    context:
        One of ``"data"`` (the data root) or ``"feather"`` (a specific
        figure feather). Tweaks the wording so the message reads
        sensibly in either situation.

    Returns
    -------
    str
        Multi-line string ready to drop into a ``raise FileNotFoundError(…)``
        or ``logging.error(…)`` call.
    """
    header = (
        f"Could not find {context} at {requested}." if requested else
        f"Could not find {context}."
    )
    root = data_root()
    return (
        f"{header}\n\n"
        f"ballpushing_utils looks for one of three sources (in order of "
        f"preference):\n\n"
        f"  1. Lab share / your own server. Set BALLPUSHING_DATA_ROOT "
        f"to a directory containing\n"
        f"     <experiment>/Metadata.json folders. Currently:\n"
        f"       BALLPUSHING_DATA_ROOT = {root}\n"
        f"     For the pipeline rerun, pass --yaml <yaml> describing the\n"
        f"     experiment / fly directories.\n\n"
        f"  2. Published Dataverse archive (Affordance / Screen / "
        f"Exploration). Download the\n"
        f"     archive that covers the figure you want, extract it under "
        f"$BALLPUSHING_DATA_ROOT,\n"
        f"     and the figure scripts pick up the bundled feathers verbatim. "
        f"To regenerate\n"
        f"     feathers from the raw HDF5 tracks, call:\n"
        f"       python src/dataset_builder.py --dataverse-root "
        f"$BALLPUSHING_DATA_ROOT/<archive>/Videos \\\n"
        f"           --experiment-type <TNT|MagnetBlock|F1|Learning> "
        f"--datasets summary\n"
        f"     See README \"Rerunning the pipeline from raw HDF5 tracks\".\n\n"
        f"  3. Bundled sample data. The repo ships one F1 + one TNT + one "
        f"MagnetBlock fly under\n"
        f"       {SAMPLE_DATA_RELATIVE}/\n"
        f"     Use it to verify your install end-to-end. The notebooks under "
        f"notebooks/\n"
        f"     (especially ballpushing_utils_walkthrough.ipynb) are wired up "
        f"against it."
    )


def require_path(
    path: str | os.PathLike[str],
    *,
    description: str | None = None,
    env_var: str | None = None,
) -> Path:
    """Resolve a hardcoded lab-share path, raising a friendly error if absent.

    Use at the top of any script (typically in the silencing-screen,
    UMAP, or co-author-contributed analysis pipelines) that reads from
    a fixed absolute path on the EPFL NFS share. When the path
    doesn't resolve — either because the share isn't mounted, the
    path uses a different mount prefix (``/mnt/upramdya_data/`` vs
    ``/mnt/upramdya/data/``), or the dataset hasn't been built yet —
    raise ``FileNotFoundError`` with a structured message that points
    the user at how to fix it (mount the share, symlink, or override
    the env var).

    Different lab members tend to mount the same NFS export at
    different mount points (``/mnt/upramdya_data/``, ``/mnt/labshare/``,
    ``/Volumes/upramdya/`` on macOS, …); this helper keeps the failure
    mode consistent across scripts and makes the override path
    obvious.

    Parameters
    ----------
    path:
        The hardcoded lab-share path to check. Returned verbatim as a
        :class:`~pathlib.Path` if it exists.
    description:
        Optional one-line label describing what the path holds (e.g.
        ``"confocal stack directory"``). Surfaced in the error message
        so the user can reason about what's missing without reading
        the source.
    env_var:
        Optional environment variable name the user can set to
        override the hardcoded path (e.g. ``"BALLPUSHING_TL_DATA_ROOT"``).
        When provided AND set, takes precedence over ``path``.

    Returns
    -------
    pathlib.Path
        The resolved path (existence verified).

    Raises
    ------
    FileNotFoundError
        Multi-line message listing the three mount-mismatch fixes
        (mount the share / symlink / edit the script's hardcoded path
        or set ``env_var``).
    """
    if env_var:
        override = os.environ.get(env_var)
        if override:
            override_p = Path(override).expanduser()
            if override_p.exists():
                return override_p
            # Fall through to error with the override mentioned.

    p = Path(path).expanduser()
    if p.exists():
        return p

    label = f" ({description})" if description else ""
    override_hint = (
        f"\n  - Override via env var: ``export {env_var}=/your/mount/path``"
        if env_var
        else ""
    )
    raise FileNotFoundError(
        f"Required path not found{label}: {p}\n\n"
        f"This script expects a fixed lab-share path that may not "
        f"resolve on machines with a different mount layout (lab\n"
        f"members commonly mount the same NFS export at different "
        f"prefixes: ``/mnt/upramdya_data/``, ``/mnt/upramdya/data/``,\n"
        f"``/mnt/labshare/``, ``/Volumes/upramdya/`` on macOS, …).\n\n"
        f"To fix:\n"
        f"  - Mount the lab NFS share at the expected prefix, OR\n"
        f"  - Symlink your local mount: "
        f"``ln -s <your-mount> {p.parent}``, OR\n"
        f"  - Edit the script's hardcoded path to match your setup."
        f"{override_hint}"
    )


def load_dotenv(path: str | os.PathLike[str] = ".env") -> dict[str, str]:
    """Minimal ``.env`` loader with no external dependencies.

    Parses ``KEY=VALUE`` lines (ignoring blanks and ``#`` comments) from
    ``path`` and injects them into :data:`os.environ` if they are not already
    set. Returns the dict of values that were applied. This avoids a hard
    dependency on ``python-dotenv`` for the simple case of two env vars.
    """
    env_path = Path(path)
    applied: dict[str, str] = {}
    if not env_path.is_file():
        return applied
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied
