"""Execute the documentation notebooks against the bundled LFS fixtures.

Drops ``BALLPUSHING_DATA_ROOT`` and points the walkthrough env vars at
the Git-LFS sample fly tree under ``tests/fixtures/sample_data``, then
runs each doc notebook with ``jupyter nbconvert --execute --inplace``.

Usage::

    python tools/execute_doc_notebooks.py                # all three doc notebooks
    python tools/execute_doc_notebooks.py --include-diagnostics   # also diagnostics_demo
    python tools/execute_doc_notebooks.py walkthrough metrics      # subset by keyword

Exits non-zero on the first failure with the notebook name and the
underlying nbconvert error, so this is safe to wire into CI later.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "sample_data"

# Canonical (non-F1) fly fixture.
SAMPLE_FLY = (
    FIXTURES
    / "MultiMazeRecorder"
    / "Videos"
    / "231130_TNT_Fine_2_Videos_Tracked"
    / "arena2"
    / "corridor5"
)

# F1 paradigm fly fixture.
F1_SAMPLE_FLY = (
    FIXTURES
    / "F1_Tracks"
    / "Videos"
    / "250904_F1_New_Videos_Checked"
    / "arena5"
    / "Right"
)

NOTEBOOKS_DIR = REPO_ROOT / "notebooks"

DOC_NOTEBOOKS = {
    "walkthrough": NOTEBOOKS_DIR / "ballpushing_utils_walkthrough.ipynb",
    "metrics": NOTEBOOKS_DIR / "ballpushing_metrics_reference.ipynb",
    "dataset-types": NOTEBOOKS_DIR / "dataset_types_guide.ipynb",
}

OPTIONAL_NOTEBOOKS = {
    "diagnostics": NOTEBOOKS_DIR / "diagnostics_demo.ipynb",
}


def _check_fixture(path: Path, label: str) -> None:
    if not path.exists():
        sys.exit(
            f"[x] {label} fixture missing at {path}.\n"
            f"    Did you run `git lfs pull`?"
        )


def _nbconvert(notebook: Path) -> None:
    # --inplace rewrites outputs into the same .ipynb so it's ready to commit.
    # A long but bounded timeout keeps a hung cell from stalling forever.
    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.timeout=600",
        str(notebook),
    ]
    print(f"[•] {notebook.name}")
    print(f"    $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        sys.exit(f"[x] FAILED: {notebook.name} (nbconvert exit {result.returncode})")
    print(f"[✓] {notebook.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "only",
        nargs="*",
        help=(
            "Optional keyword filter on notebook keys "
            f"({', '.join(sorted({**DOC_NOTEBOOKS, **OPTIONAL_NOTEBOOKS}))}). "
            "Default: all doc notebooks."
        ),
    )
    parser.add_argument(
        "--include-diagnostics",
        action="store_true",
        help="Also execute notebooks/diagnostics_demo.ipynb (offline, uses stub fly).",
    )
    args = parser.parse_args()

    if shutil.which("jupyter") is None and not _has_importable("jupyter"):
        sys.exit(
            "[x] jupyter is not installed in this Python environment.\n"
            "    Install with: pip install jupyter nbconvert"
        )

    _check_fixture(SAMPLE_FLY, "canonical fly")
    _check_fixture(F1_SAMPLE_FLY, "F1 fly")

    # Make the notebook env vars point at the LFS fixtures and drop any
    # workstation-scoped BALLPUSHING_DATA_ROOT so DATA_ROOT fallbacks
    # don't sneak in a lab-share path.
    os.environ.pop("BALLPUSHING_DATA_ROOT", None)
    os.environ["BALLPUSHING_WALKTHROUGH_FLY"] = str(SAMPLE_FLY)
    os.environ["BALLPUSHING_WALKTHROUGH_F1_FLY"] = str(F1_SAMPLE_FLY)
    print(f"[env] BALLPUSHING_WALKTHROUGH_FLY    = {SAMPLE_FLY}")
    print(f"[env] BALLPUSHING_WALKTHROUGH_F1_FLY = {F1_SAMPLE_FLY}")

    targets: dict[str, Path] = dict(DOC_NOTEBOOKS)
    if args.include_diagnostics:
        targets.update(OPTIONAL_NOTEBOOKS)

    if args.only:
        selected = {k: v for k, v in targets.items() if any(tok in k for tok in args.only)}
        if not selected:
            sys.exit(
                f"[x] No notebooks matched {args.only!r}. "
                f"Available: {sorted(targets)}"
            )
        targets = selected

    for key, notebook in targets.items():
        if not notebook.exists():
            sys.exit(f"[x] Notebook missing: {notebook}")
        _nbconvert(notebook)

    print(f"\n[✓] Executed {len(targets)} notebook(s) in place. Ready to review & commit.")


def _has_importable(module: str) -> bool:
    try:
        __import__(module)
    except ImportError:
        return False
    return True


if __name__ == "__main__":
    main()
