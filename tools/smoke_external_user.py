"""Smoke test that walks the bundled Git-LFS fixtures end-to-end.

Mimics what an external user gets after::

    git clone ...
    git lfs pull
    pip install -e ".[dev]"
    unset BALLPUSHING_DATA_ROOT       # no lab share!
    python tools/smoke_external_user.py

The script unconditionally drops ``BALLPUSHING_DATA_ROOT`` from the
environment so it always exercises the fixture path, even if you forget
to ``unset`` it in your shell. Exits non-zero on the first failure with
a readable traceback so you can wire it into CI later if you want.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path


def _step(label: str, fn) -> None:
    print(f"\n[•] {label}")
    try:
        fn()
    except Exception:  # noqa: BLE001 — we want the trace, not the type
        print(f"[x] FAILED: {label}")
        traceback.print_exc()
        sys.exit(1)
    print(f"[✓] {label}")


def main() -> None:
    # Force the fixture path so this script is reproducible regardless of
    # the caller's shell state.
    os.environ.pop("BALLPUSHING_DATA_ROOT", None)

    # Imports live inside main() so the failure surfaces cleanly through
    # _step rather than as a bare ImportError at module load time.
    from ballpushing_utils import Experiment, Fly  # noqa: PLC0415

    repo_root = Path(__file__).resolve().parents[1]
    fixtures = repo_root / "tests" / "fixtures" / "sample_data"
    if not fixtures.exists():
        sys.exit(
            f"Fixture tree missing at {fixtures}. Did you run `git lfs pull`?"
        )

    nonf1_exp_path = fixtures / "MultiMazeRecorder/Videos/231130_TNT_Fine_2_Videos_Tracked"
    nonf1_fly_path = nonf1_exp_path / "arena2/corridor5"
    f1_exp_path = fixtures / "F1_Tracks/Videos/250904_F1_New_Videos_Checked"
    f1_fly_path = f1_exp_path / "arena5/Right"
    magnetblock_exp_path = fixtures / "MultiMazeRecorder/Videos/240711_MagnetBlock_Videos_Tracked"
    magnetblock_fly_path = magnetblock_exp_path / "arena4/corridor4"

    nonf1_fly: Fly | None = None
    nonf1_exp: Experiment | None = None
    f1_fly: Fly | None = None
    f1_exp: Experiment | None = None
    magnetblock_fly: Fly | None = None
    magnetblock_exp: Experiment | None = None

    def _assert_fly_loaded(fly: Fly, label: str) -> None:
        """Force ``Fly.tracking_data`` and complain loudly if it is invalid.

        ``Fly.__init__`` is lenient: it catches load errors, sets
        ``tracking_data.valid_data = False``, and returns a usable-looking
        object. That's the right behaviour for batch processing across
        thousands of flies, but it makes the smoke test misleading — we
        want a hard failure if a fixture fly doesn't actually load.
        """
        td = fly.tracking_data
        if td is None or not getattr(td, "valid_data", False):
            raise RuntimeError(
                f"{label}: tracking data is invalid (valid_data=False). "
                "Re-run with custom_config={'debugging': True} to see the "
                "underlying traceback. Most often this is a savgol/NaN "
                "issue from utils_behavior."
            )
        print(f"    valid_data=True")

    def _assert_experiment_has_flies(exp: Experiment, label: str) -> None:
        if len(exp.flies) == 0:
            raise RuntimeError(
                f"{label}: 0 flies loaded. Underlying Fly load probably "
                "failed silently — try the per-fly debug recipe in the "
                "smoke test docstring."
            )
        print(f"    {len(exp.flies)} fly/flies loaded, fps={exp.fps}")

    def load_nonf1_fly() -> None:
        nonlocal nonf1_fly
        nonf1_fly = Fly(nonf1_fly_path, as_individual=True)
        print(f"    {nonf1_fly!r}")
        _assert_fly_loaded(nonf1_fly, "Non-F1 fly")

    def load_nonf1_experiment() -> None:
        nonlocal nonf1_exp
        nonf1_exp = Experiment(nonf1_exp_path)
        _assert_experiment_has_flies(nonf1_exp, "Non-F1 experiment")

    def load_f1_fly() -> None:
        nonlocal f1_fly
        f1_fly = Fly(
            f1_fly_path,
            as_individual=True,
            custom_config={"experiment_type": "F1"},
        )
        print(f"    {f1_fly!r}")
        _assert_fly_loaded(f1_fly, "F1 fly")

    def load_f1_experiment() -> None:
        nonlocal f1_exp
        f1_exp = Experiment(
            f1_exp_path,
            custom_config={"experiment_type": "F1"},
        )
        _assert_experiment_has_flies(f1_exp, "F1 experiment")

    def load_magnetblock_fly() -> None:
        nonlocal magnetblock_fly
        magnetblock_fly = Fly(magnetblock_fly_path, as_individual=True)
        print(f"    {magnetblock_fly!r}")
        _assert_fly_loaded(magnetblock_fly, "MagnetBlock fly")

    def load_magnetblock_experiment() -> None:
        nonlocal magnetblock_exp
        magnetblock_exp = Experiment(magnetblock_exp_path)
        _assert_experiment_has_flies(magnetblock_exp, "MagnetBlock experiment")

    _step(f"Load Non-F1 fly  ({nonf1_fly_path.relative_to(repo_root)})", load_nonf1_fly)
    _step(f"Load Non-F1 experiment  ({nonf1_exp_path.relative_to(repo_root)})", load_nonf1_experiment)
    _step(f"Load F1 fly  ({f1_fly_path.relative_to(repo_root)})", load_f1_fly)
    _step(f"Load F1 experiment  ({f1_exp_path.relative_to(repo_root)})", load_f1_experiment)
    _step(f"Load MagnetBlock fly  ({magnetblock_fly_path.relative_to(repo_root)})", load_magnetblock_fly)
    _step(f"Load MagnetBlock experiment  ({magnetblock_exp_path.relative_to(repo_root)})", load_magnetblock_experiment)

    print("\n[✓] All fixture loads succeeded. The external-user path is healthy.")


if __name__ == "__main__":
    main()
