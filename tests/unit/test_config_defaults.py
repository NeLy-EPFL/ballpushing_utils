"""Lock the two recently-changed Config / Experiment defaults.

These defaults are load-bearing for the Dataverse path, where the
``.mp4`` and ``fps.npy`` files are absent and the package has to fall
back to constants:

- ``Config.default_video_size = (1280, 1024)`` is what
  :class:`SkeletonMetrics` uses to rescale ball coordinates when no
  video is around AND the affine fit can't be made.
- ``Experiment.load_fps()`` returning ``29`` when ``fps.npy`` is absent
  matches the canonical MultiMazeRecorder / F1 rig rate used throughout
  paper acquisition.

If either changes silently, time-based and skeleton-derived metrics
drift away from the published feathers — these tests pin the values.
"""

from __future__ import annotations

import json
from pathlib import Path

from ballpushing_utils import Config, Experiment


def test_config_default_video_size():
    """Locks ``Config.default_video_size = (1280, 1024)`` — used by
    :meth:`FlyMetadata.get_video_size` when no ``.mp4`` is on disk.
    """
    cfg = Config()
    assert cfg.default_video_size == (1280, 1024)


def test_experiment_load_fps_defaults_to_29(tmp_path):
    """When ``fps.npy`` is absent, :meth:`Experiment.load_fps` falls back
    to 29 (the canonical rig rate). The Dataverse archives ship h5
    tracks only — fps is implied by this constant.
    """
    # Build a minimal experiment dir: only Metadata.json + nothing else.
    # The ``Variable`` list is empty so :meth:`load_metadata` returns
    # an empty mapping — that's fine for the ``load_fps`` path.
    metadata = {"Variable": []}
    (tmp_path / "Metadata.json").write_text(json.dumps(metadata))

    exp = Experiment(tmp_path, metadata_only=True)
    assert exp.fps == 29


def test_experiment_load_fps_reads_fps_npy_when_present(tmp_path):
    """When ``fps.npy`` IS present, the on-disk value wins."""
    import numpy as np

    metadata = {"Variable": []}
    (tmp_path / "Metadata.json").write_text(json.dumps(metadata))
    np.save(tmp_path / "fps.npy", np.array(60.0))

    exp = Experiment(tmp_path, metadata_only=True)
    assert exp.fps == 60.0


def test_experiment_accepts_synthetic_metadata(tmp_path):
    """:class:`Experiment` accepts a pre-built ``metadata=`` dict and
    skips reading ``Metadata.json``. This is the entry point the
    Dataverse loader uses since the archives don't ship Metadata.json.
    """
    # NOTE: we do NOT create Metadata.json — the synthetic metadata path
    # must work without it.
    synthetic = {"Genotype": {"arena1": "MB247xTNT"}}
    exp = Experiment(tmp_path, metadata_only=True, metadata=synthetic)
    assert exp.metadata == synthetic
