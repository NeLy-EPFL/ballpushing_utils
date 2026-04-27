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
    "figure_output_dir",
    "load_dotenv",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_FIGURES_ROOT_NAME",
]

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
