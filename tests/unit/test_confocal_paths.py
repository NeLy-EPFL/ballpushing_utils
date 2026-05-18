"""Unit tests for the confocal-stacks data-resolution helpers.

Covers the three path-resolution functions introduced for ED Figure 4:

- :func:`resolve_confocal_dir` — four-rule resolution: env var →
  lab-server path → ``Datasets/confocal/`` → Dataverse auto-download.
- :func:`resolve_stack_tiff` — dual-layout lookup: lab-server nested
  path vs. Dataverse flat ``{key}.tiff``.
- :func:`resolve_jrc_nrrd_paths` — tri-source NRRD lookup: env var →
  ``raw_dir/jrc2018/`` → ``.cache/registration/jrc2018_src/``.

Also verifies:

- :data:`ballpushing_utils.dataverse_naming.CONFOCAL_DOI` is present
  and looks like a Harvard Dataverse DOI.
- ``_download_confocal_from_dataverse`` correctly skips files that
  already exist and writes others via atomic-rename (mocked network).

All tests are hermetic — no network, no lab share, no LFS fixtures.
"""

from __future__ import annotations

import importlib
import json
import sys
import types
import urllib.error
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import the helpers under test.
# The figure script lives outside the installable package, so we import it
# via importlib with the heavy dependencies (ants, cv2, tifffile, …) stubbed
# out first to keep the unit test offline and fast.
# ---------------------------------------------------------------------------

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "figures"
    / "EDFigure4-Confocal"
    / "edfigure4_confocal_stacks.py"
)


def _stub_module(name: str, **attrs) -> types.ModuleType:
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    return m


def _load_confocal_script(monkeypatch, confocal_dir: Path):
    """Import the figure script with heavy deps stubbed and the confocal dir
    pointed at *confocal_dir* (which must contain a ``stack_infos.yaml``)."""

    # Stub heavy C-extension deps that we don't need for the helpers.
    for mod in ("ants", "cv2", "tifffile", "mplex"):
        if mod not in sys.modules:
            monkeypatch.setitem(sys.modules, mod, _stub_module(mod))

    # Point BALLPUSHING_TL_CONFOCAL_DIR at the supplied directory so
    # resolve_confocal_dir() returns it without any fallback logic.
    monkeypatch.setenv("BALLPUSHING_TL_CONFOCAL_DIR", str(confocal_dir))
    monkeypatch.setenv("BALLPUSHING_JRC2018_DIR", str(confocal_dir / "jrc2018"))

    # Clear cached module if a previous test loaded it.
    sys.modules.pop("edfigure4_confocal_stacks", None)

    spec = importlib.util.spec_from_file_location("edfigure4_confocal_stacks", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def confocal_flat_dir(tmp_path):
    """A minimal Dataverse-style flat confocal directory.

    Contains:
    - stack_infos.yaml  (minimal valid YAML — one genotype key)
    - mb247.tiff        (zero-byte placeholder)
    - jrc2018/JRC2018_FEMALE_38um_iso_16bit.nrrd  (placeholder)
    - jrc2018/JRC2018_VNC_FEMALE_4iso.nrrd        (placeholder)
    """
    d = tmp_path / "confocal_flat"
    d.mkdir()
    # Minimal stack_infos YAML — the loader only needs it to be parseable.
    yaml_content = {
        "mb247": {
            "path": "260101_mb247/mb247.tiff",
            "name": "MB247",
            "dx": 0.5,
            "dz": 1.0,
            "brain": {"p0": [100, 100], "p1": [200, 200], "ventral": False},
            "vnc": {"p0": [100, 300], "p1": [200, 400], "ventral": True},
        }
    }
    import yaml

    (d / "stack_infos.yaml").write_text(yaml.safe_dump(yaml_content))
    (d / "mb247.tiff").write_bytes(b"")
    jrc = d / "jrc2018"
    jrc.mkdir()
    (jrc / "JRC2018_FEMALE_38um_iso_16bit.nrrd").write_bytes(b"")
    (jrc / "JRC2018_VNC_FEMALE_4iso.nrrd").write_bytes(b"")
    return d


@pytest.fixture()
def confocal_server_dir(tmp_path):
    """A lab-server-style nested confocal directory.

    Contains the tiff at the path stored in stack_infos.yaml's ``"path"``
    field (date-prefixed nested structure).
    """
    d = tmp_path / "confocal_server"
    d.mkdir()
    nested = d / "260101_mb247"
    nested.mkdir()
    (nested / "mb247.tiff").write_bytes(b"")

    import yaml

    yaml_content = {
        "mb247": {
            "path": "260101_mb247/mb247.tiff",
            "name": "MB247",
            "dx": 0.5,
            "dz": 1.0,
            "brain": {"p0": [100, 100], "p1": [200, 200], "ventral": False},
            "vnc": {"p0": [100, 300], "p1": [200, 400], "ventral": True},
        }
    }
    (d / "stack_infos.yaml").write_text(yaml.safe_dump(yaml_content))
    jrc = d / "jrc2018"
    jrc.mkdir()
    (jrc / "JRC2018_FEMALE_38um_iso_16bit.nrrd").write_bytes(b"")
    (jrc / "JRC2018_VNC_FEMALE_4iso.nrrd").write_bytes(b"")
    return d


# ---------------------------------------------------------------------------
# CONFOCAL_DOI constant
# ---------------------------------------------------------------------------


def test_confocal_doi_is_present():
    from ballpushing_utils.dataverse_naming import CONFOCAL_DOI

    assert isinstance(CONFOCAL_DOI, str)
    assert CONFOCAL_DOI.startswith("doi:10.7910/DVN/")


def test_confocal_doi_not_in_dataverse_dois():
    """CONFOCAL_DOI is a separate constant — not lumped into the feather
    downloader's DATAVERSE_DOIS dict."""
    from ballpushing_utils.dataverse_naming import CONFOCAL_DOI, DATAVERSE_DOIS

    assert CONFOCAL_DOI not in DATAVERSE_DOIS.values()


# ---------------------------------------------------------------------------
# resolve_confocal_dir
# ---------------------------------------------------------------------------


def test_resolve_confocal_dir_env_var_wins(tmp_path, monkeypatch, confocal_flat_dir):
    """Env var takes priority over all other paths."""
    # We import the function directly to test without executing module-level code.
    spec = importlib.util.spec_from_file_location("_confocal_helpers", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)

    # Stub heavy deps before loading.
    for dep in ("ants", "cv2", "tifffile", "mplex"):
        monkeypatch.setitem(sys.modules, dep, _stub_module(dep))

    # Point env var at our flat fixture dir.
    monkeypatch.setenv("BALLPUSHING_TL_CONFOCAL_DIR", str(confocal_flat_dir))
    # Also give a valid JRC dir so module-level code doesn't fail.
    monkeypatch.setenv("BALLPUSHING_JRC2018_DIR", str(confocal_flat_dir / "jrc2018"))

    sys.modules.pop("edfigure4_confocal_stacks", None)
    spec.loader.exec_module(mod)

    result = mod.resolve_confocal_dir()
    assert result == confocal_flat_dir


def test_resolve_confocal_dir_datasets_yaml(tmp_path, monkeypatch):
    """``Datasets/confocal/stack_infos.yaml`` present → return that dir."""
    import yaml
    from ballpushing_utils.paths import REPO_DATASETS_DIR

    # Build a temporary Datasets/confocal/ directory with the yaml.
    confocal = tmp_path / "confocal"
    confocal.mkdir()
    (confocal / "stack_infos.yaml").write_text(yaml.safe_dump({}))

    # Patch REPO_DATASETS_DIR so the helper resolves our tmp dir.
    monkeypatch.setattr(
        "ballpushing_utils.paths.REPO_DATASETS_DIR", tmp_path, raising=False
    )

    for dep in ("ants", "cv2", "tifffile", "mplex"):
        monkeypatch.setitem(sys.modules, dep, _stub_module(dep))

    # Clear env vars so the helper falls through to step 3.
    monkeypatch.delenv("BALLPUSHING_TL_CONFOCAL_DIR", raising=False)
    # Provide JRC placeholder so module-level resolve_jrc_nrrd_paths won't fail.
    jrc = tmp_path / "jrc2018"
    jrc.mkdir()
    for fname in ("JRC2018_FEMALE_38um_iso_16bit.nrrd", "JRC2018_VNC_FEMALE_4iso.nrrd"):
        (jrc / fname).write_bytes(b"")
    monkeypatch.setenv("BALLPUSHING_JRC2018_DIR", str(jrc))

    sys.modules.pop("edfigure4_confocal_stacks", None)
    spec = importlib.util.spec_from_file_location("edfigure4_confocal_stacks", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    result = mod.resolve_confocal_dir()
    assert result == confocal


def test_resolve_confocal_dir_no_download_raises(monkeypatch):
    """When nothing is found and auto_download=False, raise FileNotFoundError."""
    for dep in ("ants", "cv2", "tifffile", "mplex"):
        monkeypatch.setitem(sys.modules, dep, _stub_module(dep))
    monkeypatch.delenv("BALLPUSHING_TL_CONFOCAL_DIR", raising=False)

    # Patch _confocal_datasets_dir to return a non-existent path.
    sys.modules.pop("edfigure4_confocal_stacks", None)
    spec = importlib.util.spec_from_file_location("edfigure4_confocal_stacks", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)

    # We need a confocal dir for module-level to work; patch after load.
    # Instead, test the function in isolation without triggering module-level.
    # Load with a valid env dir first, then test the function directly.
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        import yaml

        (td_path / "stack_infos.yaml").write_text(yaml.safe_dump({}))
        jrc = td_path / "jrc2018"
        jrc.mkdir()
        for fname in (
            "JRC2018_FEMALE_38um_iso_16bit.nrrd",
            "JRC2018_VNC_FEMALE_4iso.nrrd",
        ):
            (jrc / fname).write_bytes(b"")
        monkeypatch.setenv("BALLPUSHING_TL_CONFOCAL_DIR", str(td_path))
        monkeypatch.setenv("BALLPUSHING_JRC2018_DIR", str(jrc))

        sys.modules.pop("edfigure4_confocal_stacks", None)
        spec.loader.exec_module(mod)

    # Now test with no valid paths and auto_download=False.
    monkeypatch.delenv("BALLPUSHING_TL_CONFOCAL_DIR", raising=False)
    monkeypatch.setattr(
        mod, "_confocal_datasets_dir", lambda: Path("/nonexistent/__confocal_test__")
    )

    with pytest.raises(FileNotFoundError, match="doi:10.7910/DVN/MY4GN5"):
        mod.resolve_confocal_dir(auto_download=False)


def test_resolve_confocal_dir_error_message_mentions_dataverse(monkeypatch, tmp_path):
    """The FileNotFoundError mentions the Dataverse DOI and Datasets path."""
    for dep in ("ants", "cv2", "tifffile", "mplex"):
        monkeypatch.setitem(sys.modules, dep, _stub_module(dep))

    fake_confocal = tmp_path / "confocal"
    import yaml

    fake_confocal.mkdir()
    (fake_confocal / "stack_infos.yaml").write_text(yaml.safe_dump({}))
    jrc = tmp_path / "jrc2018"
    jrc.mkdir()
    for fname in ("JRC2018_FEMALE_38um_iso_16bit.nrrd", "JRC2018_VNC_FEMALE_4iso.nrrd"):
        (jrc / fname).write_bytes(b"")
    monkeypatch.setenv("BALLPUSHING_TL_CONFOCAL_DIR", str(fake_confocal))
    monkeypatch.setenv("BALLPUSHING_JRC2018_DIR", str(jrc))

    sys.modules.pop("edfigure4_confocal_stacks", None)
    spec = importlib.util.spec_from_file_location("edfigure4_confocal_stacks", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    monkeypatch.delenv("BALLPUSHING_TL_CONFOCAL_DIR", raising=False)
    monkeypatch.setattr(
        mod, "_confocal_datasets_dir", lambda: Path("/nonexistent/__confocal_test__")
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        mod.resolve_confocal_dir(auto_download=False)
    msg = str(exc_info.value)
    assert "doi:10.7910/DVN/MY4GN5" in msg
    assert "BALLPUSHING_TL_CONFOCAL_DIR" in msg


# ---------------------------------------------------------------------------
# resolve_stack_tiff
# ---------------------------------------------------------------------------


def _load_mod(monkeypatch, confocal_dir):
    """Helper: load the script with heavy deps stubbed."""
    for dep in ("ants", "cv2", "tifffile", "mplex"):
        monkeypatch.setitem(sys.modules, dep, _stub_module(dep))
    monkeypatch.setenv("BALLPUSHING_TL_CONFOCAL_DIR", str(confocal_dir))
    monkeypatch.setenv("BALLPUSHING_JRC2018_DIR", str(confocal_dir / "jrc2018"))
    sys.modules.pop("edfigure4_confocal_stacks", None)
    spec = importlib.util.spec_from_file_location("edfigure4_confocal_stacks", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_resolve_stack_tiff_flat_layout(monkeypatch, confocal_flat_dir):
    """Flat layout: ``{key}.tiff`` is found when server path is absent."""
    mod = _load_mod(monkeypatch, confocal_flat_dir)
    stack_info = {"path": "260101_mb247/mb247.tiff"}
    result = mod.resolve_stack_tiff(confocal_flat_dir, "mb247", stack_info)
    assert result == confocal_flat_dir / "mb247.tiff"


def test_resolve_stack_tiff_server_layout(monkeypatch, confocal_server_dir):
    """Server layout: nested path in ``stack_info['path']`` is found first."""
    mod = _load_mod(monkeypatch, confocal_server_dir)
    stack_info = {"path": "260101_mb247/mb247.tiff"}
    result = mod.resolve_stack_tiff(confocal_server_dir, "mb247", stack_info)
    assert result == confocal_server_dir / "260101_mb247" / "mb247.tiff"


def test_resolve_stack_tiff_server_wins_over_flat(monkeypatch, tmp_path):
    """When both server-nested and flat files exist, server path wins."""
    import yaml

    d = tmp_path / "both"
    d.mkdir()
    nested = d / "260101_mb247"
    nested.mkdir()
    (nested / "mb247.tiff").write_bytes(b"server")
    (d / "mb247.tiff").write_bytes(b"flat")

    yaml_content = {
        "mb247": {
            "path": "260101_mb247/mb247.tiff",
            "name": "MB247",
            "dx": 0.5,
            "dz": 1.0,
            "brain": {"p0": [0, 0], "p1": [1, 1], "ventral": False},
            "vnc": {"p0": [0, 0], "p1": [1, 1], "ventral": True},
        }
    }
    (d / "stack_infos.yaml").write_text(yaml.safe_dump(yaml_content))
    jrc = d / "jrc2018"
    jrc.mkdir()
    for fname in ("JRC2018_FEMALE_38um_iso_16bit.nrrd", "JRC2018_VNC_FEMALE_4iso.nrrd"):
        (jrc / fname).write_bytes(b"")

    mod = _load_mod(monkeypatch, d)
    stack_info = {"path": "260101_mb247/mb247.tiff"}
    result = mod.resolve_stack_tiff(d, "mb247", stack_info)
    assert result == nested / "mb247.tiff"


def test_resolve_stack_tiff_missing_raises(monkeypatch, confocal_flat_dir):
    """FileNotFoundError with Dataverse URL when neither path exists."""
    mod = _load_mod(monkeypatch, confocal_flat_dir)
    stack_info = {"path": "260101_unknown/unknown.tiff"}
    with pytest.raises(FileNotFoundError, match="https://doi.org/10.7910/DVN/MY4GN5"):
        mod.resolve_stack_tiff(confocal_flat_dir, "unknown_genotype", stack_info)


# ---------------------------------------------------------------------------
# resolve_jrc_nrrd_paths
# ---------------------------------------------------------------------------


def test_resolve_jrc_nrrd_paths_env_var_override(
    monkeypatch, tmp_path, confocal_flat_dir
):
    """BALLPUSHING_JRC2018_DIR env var is consulted first."""
    mod = _load_mod(monkeypatch, confocal_flat_dir)

    # Put NRRDs in a custom location and re-set the env var after loading the module
    # (_load_mod points BALLPUSHING_JRC2018_DIR at the fixture's jrc2018/ dir).
    jrc_custom = tmp_path / "my_jrc"
    jrc_custom.mkdir()
    for fname in ("JRC2018_FEMALE_38um_iso_16bit.nrrd", "JRC2018_VNC_FEMALE_4iso.nrrd"):
        (jrc_custom / fname).write_bytes(b"")
    monkeypatch.setenv("BALLPUSHING_JRC2018_DIR", str(jrc_custom))

    result = mod.resolve_jrc_nrrd_paths(confocal_flat_dir)
    assert result["brain"] == jrc_custom / "JRC2018_FEMALE_38um_iso_16bit.nrrd"
    assert result["vnc"] == jrc_custom / "JRC2018_VNC_FEMALE_4iso.nrrd"


def test_resolve_jrc_nrrd_paths_alongside_confocal(monkeypatch, confocal_flat_dir):
    """NRRDs in ``raw_dir/jrc2018/`` are found without any env var."""
    monkeypatch.delenv("BALLPUSHING_JRC2018_DIR", raising=False)
    mod = _load_mod(monkeypatch, confocal_flat_dir)

    # Reset the env var that _load_mod set.
    monkeypatch.delenv("BALLPUSHING_JRC2018_DIR", raising=False)

    result = mod.resolve_jrc_nrrd_paths(confocal_flat_dir)
    assert result["brain"].name == "JRC2018_FEMALE_38um_iso_16bit.nrrd"
    assert result["vnc"].name == "JRC2018_VNC_FEMALE_4iso.nrrd"


def test_resolve_jrc_nrrd_paths_missing_raises_with_urls(
    monkeypatch, confocal_flat_dir
):
    """FileNotFoundError mentions both Janelia figshare URLs and env var."""
    mod = _load_mod(monkeypatch, confocal_flat_dir)
    monkeypatch.delenv("BALLPUSHING_JRC2018_DIR", raising=False)
    # Remove the NRRDs from the fixture.
    for fname in ("JRC2018_FEMALE_38um_iso_16bit.nrrd", "JRC2018_VNC_FEMALE_4iso.nrrd"):
        (confocal_flat_dir / "jrc2018" / fname).unlink()

    with pytest.raises(FileNotFoundError) as exc_info:
        mod.resolve_jrc_nrrd_paths(confocal_flat_dir)
    msg = str(exc_info.value)
    assert "figshare.com" in msg
    assert "BALLPUSHING_JRC2018_DIR" in msg
    assert "JRC2018_FEMALE_38um_iso_16bit.nrrd" in msg
    assert "JRC2018_VNC_FEMALE_4iso.nrrd" in msg


def test_resolve_jrc_nrrd_paths_cache_fallback(
    monkeypatch, tmp_path, confocal_flat_dir
):
    """NRRDs in ``.cache/registration/jrc2018_src/`` are found as last resort."""
    monkeypatch.delenv("BALLPUSHING_JRC2018_DIR", raising=False)
    # Remove them from the ``jrc2018/`` sibling dir.
    for fname in ("JRC2018_FEMALE_38um_iso_16bit.nrrd", "JRC2018_VNC_FEMALE_4iso.nrrd"):
        (confocal_flat_dir / "jrc2018" / fname).unlink()

    # Put them in the cache dir that get_cache_dir() returns.
    cache_jrc = tmp_path / ".cache" / "registration" / "jrc2018_src"
    cache_jrc.mkdir(parents=True)
    for fname in ("JRC2018_FEMALE_38um_iso_16bit.nrrd", "JRC2018_VNC_FEMALE_4iso.nrrd"):
        (cache_jrc / fname).write_bytes(b"")

    mod = _load_mod(monkeypatch, confocal_flat_dir)
    monkeypatch.delenv("BALLPUSHING_JRC2018_DIR", raising=False)
    # Patch get_cache_dir to return our tmp cache.
    monkeypatch.setattr(mod, "get_cache_dir", lambda: tmp_path / ".cache")

    result = mod.resolve_jrc_nrrd_paths(confocal_flat_dir)
    assert result["brain"].parent == cache_jrc
    assert result["vnc"].parent == cache_jrc


# ---------------------------------------------------------------------------
# _download_confocal_from_dataverse (mocked network)
# ---------------------------------------------------------------------------


def _make_fake_api_response(files: dict[str, int]) -> bytes:
    """Construct a minimal Dataverse API JSON payload."""
    file_entries = [
        {
            "dataFile": {
                "filename": name,
                "id": fid,
                "filesize": 4,
                "md5": None,
                "contentType": "image/tiff",
            }
        }
        for name, fid in files.items()
    ]
    return json.dumps({"data": {"latestVersion": {"files": file_entries}}}).encode()


class _FakeResponse:
    """Minimal urllib response-alike backed by bytes."""

    def __init__(self, data: bytes):
        self._io = BytesIO(data)
        self._headers = {"Content-Length": str(len(data))}

    def read(self, n=-1):
        return self._io.read(n)

    def info(self):
        return self._headers

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __iter__(self):
        return self._io


def test_download_confocal_skips_existing(monkeypatch, tmp_path, confocal_flat_dir):
    """Files already in the destination are skipped (no download attempted)."""
    mod = _load_mod(monkeypatch, confocal_flat_dir)

    dest = tmp_path / "dest"
    dest.mkdir()
    # Pre-populate one of the files.
    (dest / "mb247.tiff").write_bytes(b"existing")

    api_payload = _make_fake_api_response({"mb247.tiff": 101, "stack_infos.yaml": 102})
    yaml_bytes = b"mb247: {}"
    file_contents = {101: b"", 102: yaml_bytes}

    def fake_urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "persistentId" in url:
            return _FakeResponse(api_payload)
        for fid, data in file_contents.items():
            if str(fid) in url:
                return _FakeResponse(data)
        raise urllib.error.URLError(f"unexpected URL: {url}")

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        mod._download_confocal_from_dataverse(dest)

    # mb247.tiff should NOT have been overwritten.
    assert (dest / "mb247.tiff").read_bytes() == b"existing"
    # stack_infos.yaml should have been downloaded.
    assert (dest / "stack_infos.yaml").exists()


def test_download_confocal_writes_new_files(monkeypatch, tmp_path, confocal_flat_dir):
    """New files are downloaded and written atomically (no .part files left)."""
    mod = _load_mod(monkeypatch, confocal_flat_dir)

    dest = tmp_path / "dest2"
    dest.mkdir()

    api_payload = _make_fake_api_response({"stack_infos.yaml": 202})
    yaml_bytes = b"mb247: {name: MB247}"

    def fake_urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "persistentId" in url:
            return _FakeResponse(api_payload)
        if "202" in url:
            return _FakeResponse(yaml_bytes)
        raise urllib.error.URLError(f"unexpected URL: {url}")

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        mod._download_confocal_from_dataverse(dest)

    assert (dest / "stack_infos.yaml").read_bytes() == yaml_bytes
    # No .part files left behind.
    assert not any(dest.glob("*.part"))


def test_download_confocal_cleans_up_part_on_failure(
    monkeypatch, tmp_path, confocal_flat_dir
):
    """If the download raises, no ``.part`` temp file is left behind."""
    mod = _load_mod(monkeypatch, confocal_flat_dir)

    dest = tmp_path / "dest3"
    dest.mkdir()

    api_payload = _make_fake_api_response({"broken.tiff": 303})

    call_count = [0]

    def fake_urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "persistentId" in url:
            return _FakeResponse(api_payload)
        call_count[0] += 1
        raise urllib.error.URLError("simulated network failure")

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        with pytest.raises(urllib.error.URLError):
            mod._download_confocal_from_dataverse(dest)

    assert not any(dest.glob("*.part"))
