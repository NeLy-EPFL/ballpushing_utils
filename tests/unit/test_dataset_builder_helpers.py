"""Unit tests for the experiment-type / dataset-eligibility helpers in
``src/dataset_builder.py``.

These guard the routing logic that decides:
  * which experiment type a folder belongs to (path detection + CLI override),
  * which datasets a given experiment type can produce.

They run offline — they don't touch the lab share, so they fit the unit
suite. The dataset_builder module pulls in heavy lab dependencies at
import time (tqdm, pandas, ballpushing_utils, …); we stub the unused
ones with ``importlib`` machinery so the helpers can be exercised in
isolation.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_BUILDER = REPO_ROOT / "src" / "dataset_builder.py"


def _stub(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


@pytest.fixture(scope="module")
def db():
    """Import dataset_builder.py with heavy deps stubbed out.

    We only need the path-walking and eligibility helpers — not the
    actual data loaders — so we replace the imports it does at the top
    of the module with minimal stand-ins.
    """
    sys.modules.setdefault(
        "tqdm",
        _stub("tqdm", tqdm=lambda iterable=None, *a, **kw: iterable),
    )

    class _Proc:
        def memory_info(self):
            return type("m", (), {"rss": 0})()

    sys.modules.setdefault("psutil", _stub("psutil", Process=lambda pid=None: _Proc()))
    sys.modules.setdefault("yaml", _stub("yaml", safe_load=lambda *a, **k: {}, YAMLError=Exception))

    pandas_stub = _stub("pandas", DataFrame=type("DF", (), {}), concat=lambda *a, **k: None)
    sys.modules.setdefault("pandas", pandas_stub)

    bpu = _stub(
        "ballpushing_utils",
        Fly=lambda *a, **kw: None,
        Experiment=lambda *a, **kw: None,
        Dataset=lambda *a, **kw: None,
    )

    class _Cfg:
        def __init__(self):
            self.experiment_type = "F1"

        def print_config(self):
            pass

    bpu_config = _stub("ballpushing_utils.config", Config=_Cfg)
    bpu_utilities = _stub("ballpushing_utils.utilities")
    bpu.config = bpu_config
    bpu.utilities = bpu_utilities
    sys.modules.setdefault("ballpushing_utils", bpu)
    sys.modules.setdefault("ballpushing_utils.config", bpu_config)
    sys.modules.setdefault("ballpushing_utils.utilities", bpu_utilities)

    spec = importlib.util.spec_from_file_location("dataset_builder_module", DATASET_BUILDER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# detect_experiment_type / resolve_experiment_type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path, expected",
    [
        pytest.param(
            "/mnt/upramdya_data/MD/F1_Tracks/Videos/250904_F1_New_Videos_Checked",
            "F1",
            id="f1_canonical_dir",
        ),
        pytest.param(
            "/mnt/upramdya_data/MD/MagnetBlock/Videos/240711_MagnetBlock_Videos_Tracked",
            "MagnetBlock",
            id="magnetblock_canonical_dir",
        ),
        pytest.param(
            "/mnt/upramdya_data/MD/MultiMazeRecorder/Videos/231130_TNT_Fine_2_Videos_Tracked",
            "TNT",
            id="multimaze_catchall_defaults_to_TNT",
        ),
        pytest.param("/something/unrelated/Videos/x", None, id="no_match_returns_None"),
    ],
)
def test_detect_experiment_type(db, path, expected):
    assert db.detect_experiment_type(Path(path)) == expected


def test_resolve_experiment_type_path_detect(db):
    """Without override, detection wins."""
    p = Path("/mnt/upramdya_data/MD/F1_Tracks/Videos/x")
    assert db.resolve_experiment_type(p) == "F1"


def test_resolve_experiment_type_override_beats_detect(db):
    """Override wins even when detection would have returned a value."""
    p = Path("/mnt/upramdya_data/MD/F1_Tracks/Videos/x")
    assert db.resolve_experiment_type(p, override="MagnetBlock") == "MagnetBlock"


def test_resolve_experiment_type_unknown_path_no_override_raises(db):
    """No detection + no override is a hard error — never silently default."""
    p = Path("/mnt/upramdya_data/MD/SomethingNew/Videos/x")
    with pytest.raises(ValueError, match="Could not auto-detect experiment type"):
        db.resolve_experiment_type(p)


# ---------------------------------------------------------------------------
# filter_datasets_for_type
# ---------------------------------------------------------------------------


def test_filter_datasets_for_type_f1_keeps_all_compatible(db):
    out = db.filter_datasets_for_type(["summary", "F1_coordinates", "coordinates"], "F1")
    assert out == ["summary", "F1_coordinates", "coordinates"]


def test_filter_datasets_for_type_magnetblock_drops_f1_only(db, caplog):
    """F1-only datasets are skipped on MagnetBlock and a warning is logged."""
    with caplog.at_level("WARNING"):
        out = db.filter_datasets_for_type(
            ["summary", "F1_coordinates", "coordinates"], "MagnetBlock"
        )
    assert out == ["summary", "coordinates"]
    assert any("F1_coordinates" in rec.getMessage() for rec in caplog.records)


def test_filter_datasets_for_type_all_incompatible_returns_empty(db):
    """If every requested dataset is F1-only, TNT gets nothing back."""
    out = db.filter_datasets_for_type(["F1_coordinates", "F1_checkpoints"], "TNT")
    assert out == []


def test_filter_datasets_for_type_unknown_type_raises(db):
    with pytest.raises(ValueError, match="Unknown experiment type"):
        db.filter_datasets_for_type(["summary"], "NotARealType")


# ---------------------------------------------------------------------------
# Eligibility table sanity
# ---------------------------------------------------------------------------


def test_f1_only_datasets_not_in_other_eligibility_sets(db):
    """F1_coordinates / F1_checkpoints belong only in the F1 set.
    If someone adds them to TNT or MagnetBlock by accident, this catches it.
    """
    f1_only = {"F1_coordinates", "F1_checkpoints"}
    for exp_type, eligible in db.ELIGIBLE_DATASETS.items():
        if exp_type == "F1":
            assert f1_only.issubset(eligible), f"F1 missing F1-only datasets: {eligible}"
        else:
            leak = f1_only & eligible
            assert not leak, f"F1-only datasets leaked into {exp_type}: {leak}"


def test_all_dataset_types_is_union_of_eligibility(db):
    """``ALL_DATASET_TYPES`` should be exactly the union — used for argparse choices."""
    union = sorted({d for ds in db.ELIGIBLE_DATASETS.values() for d in ds})
    assert db.ALL_DATASET_TYPES == union
