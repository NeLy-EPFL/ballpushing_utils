# Test fixtures

Small, curated samples of the data `ballpushing_utils` operates on,
bundled so the data-gated tests, the integration suite, and the
walkthrough notebooks work out of the box on a fresh clone — no lab
share required.

## What's here

Everything lives under `tests/fixtures/sample_data/` and mirrors the
original layout under `$BALLPUSHING_DATA_ROOT`. That way the same
`paths.dataset(rel)` helper resolves both real data and bundled
fixtures, so the conftest plumbing stays minimal.

The canonical fixture set is three flies:

| Purpose | Relative path |
| --- | --- |
| Non-F1 (TNT_Fine) | `MultiMazeRecorder/Videos/231130_TNT_Fine_2_Videos_Tracked/arena2/corridor5/` |
| F1 paradigm | `F1_Tracks/Videos/250904_F1_New_Videos_Checked/arena5/Right/` |
| MagnetBlock | (add whichever MagnetBlock fly you find representative) |

Each corridor directory carries:

- `<corridor>.mp4` — the original recording, re-encoded with
  `libx264 -crf 28 -preset veryslow -an -movflags +faststart`.
  Resolution and frame rate are preserved; the SLEAP pixel
  coordinates in the H5 tracks therefore still line up with the
  compressed video.
- `*ball*.h5`, `*fly*.h5`, `*full_body*.h5` — the SLEAP tracks,
  copied verbatim.

Each **experiment** directory (one level up from the arena directories)
additionally carries:

- `Metadata.json` — the per-arena variable table the `Experiment`
  loader reads.
- `fps.npy` — the recording frame rate. Optional (`Experiment.load_fps`
  falls back to 30 fps without it), but bundled when available to keep
  time-axis reporting faithful.

## How to pull the fixtures

The binary assets (`*.mp4`, `*.h5`, `*.npy` under `tests/fixtures/`)
are tracked with [Git LFS](https://git-lfs.com). Install it once, then
a normal clone or pull fetches them automatically:

```bash
# One-time
git lfs install

# Fresh clone: LFS objects are downloaded alongside the working tree
git clone https://github.com/NeLy-EPFL/ballpushing_utils.git

# Existing clone: pull LFS objects
git lfs pull
```

If you only want the code (no binary assets), set
`GIT_LFS_SKIP_SMUDGE=1` before cloning; you'll end up with LFS
pointer files instead of real data, and the data-gated tests will
skip cleanly.

## How to regenerate a fixture

From the workstation (where the raw recordings live):

```bash
python tools/compress_sample_fly.py \
    --source /mnt/upramdya_data/MD/MultiMazeRecorder/Videos/231130_TNT_Fine_2_Videos_Tracked/arena2/corridor5 \
    --data-root /mnt/upramdya_data/MD
```

The script re-encodes every `*.mp4`, copies the `*.h5` tracks
verbatim, and pulls the experiment's `Metadata.json` + `fps.npy` into
the right subtree. Run it once per fly you want to bundle, then
`git add tests/fixtures .gitattributes && git commit`. Git LFS picks
up the binary files automatically via `.gitattributes`.

See `tools/compress_sample_fly.py --help` for all knobs (CRF,
destination, force overwrite).
