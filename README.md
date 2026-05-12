# ballpushing_utils

Analysis library and figure-reproduction code for the *Drosophila*
ball-pushing paradigm developed in the [Ramdya Lab](https://www.epfl.ch/labs/ramdya-lab/)
at EPFL. It computes behavioural metrics from SLEAP-tracked recordings of
flies interacting with a ball in a corridor, and contains the scripts that
generate every panel in the paper companion of this repository.

> **Paper citation:** *[Durrieu et al., 2026, "Object manipulation and affordance learning in Drosophila"](https://doi.org/10.64898/2026.04.28.721021).*
>
> **Dataset:** Three companion Dataverse archives host the data:
> *Affordance* (F1 + MagnetBlock, https://doi.org/10.7910/DVN/91R87T), *Screen* (TNT silencing
> screen, https://doi.org/10.7910/DVN/SPBKKJ), and *Exploration* (every other paradigm —
> wild-type baselines, ball types, ball scents, feeding state, dark
> olfaction, learning mutants, https://doi.org/10.7910/DVN/VB4UI5). Each archive bundles the
> **per-fly SLEAP HDF5 tracks** (sorted by condition), the **pooled
> feathers** the figure scripts read, and the `config.json` that was
> used to build the feathers. You can reproduce every paper figure from
> the feathers alone — see [Reproducing figures from
> Dataverse](#reproducing-figures-from-the-dataverse) — or rerun the
> entire pipeline from raw tracks with `dataset_builder.py
> --dataverse-root` (see [Rerunning the pipeline from raw HDF5
> tracks](#rerunning-the-pipeline-from-raw-hdf5-tracks)).

---

## What's in here

```
ballpushing_utils/
├── src/ballpushing_utils/      # The Python library (pip-installable).
│   ├── plotting/               # Shared figure helpers (rcParams, sig bars,
│   │                           #   cm-based axis sizing, paired boxplots).
│   ├── stats/                  # Permutation test, bootstrap CI, Cohen's d.
│   ├── ballpushing_metrics.py  # Per-fly metric definitions (see below).
│   ├── dataset.py              # Dataset loader / pooler.
│   ├── experiment.py, fly.py   # Domain objects (Experiment > Fly).
│   ├── fly_trackingdata.py     # SLEAP track wrapper.
│   ├── paths.py                # Data/figure path helpers (env-var driven).
│   └── ...
├── src/Screen_analysis/        # Brain-region screen analysis pipeline.
├── figures/                    # Paper figure scripts (Fig. 1 – Fig. 3 +
│   ├── Fig1-setup/             #   ED Fig. 6). One script per panel; each
│   ├── Fig2-Affordance/        #   reads a feather, runs stats, writes a
│   ├── Fig3-Screen/            #   PDF + a stats CSV.
│   └── EDFigure6-Dendrogram/
├── plots/                      # Exploratory + supplementary plots
│   ├── Ballpushing_PR/         #   (feeding state, wildtype push rate,
│   ├── F1_tracks/              #   F1 paradigm, ball scents, etc.).
│   └── Supplementary_exps/     # Supplementary-figure scripts.
├── experiments_yaml/           # YAML descriptors of every experiment
│                               #   batch (genotype, replicate dates, ...).
├── notebooks/                  # Jupyter walkthroughs (Fly/Experiment/
│                               #   Dataset tour + diagnostics demo).
├── tools/                      # CLI / dashboard entry points
│                               #   (e.g. tools/diagnostics_dashboard.py,
│                               #   tools/compress_sample_fly.py).
├── tests/                      # pytest suite + Git-LFS sample fixtures
│   ├── unit/                   #   (tests/fixtures/sample_data/ — real fly
│   ├── integration/            #   videos + SLEAP tracks; see
│   └── fixtures/               #   tests/fixtures/README.md).
├── run_all_figures.py          # Run every script under figures/.
├── pyproject.toml              # Package + dev-tool config.
└── .env.example                # Template for local data/figure paths.
```

A complete description of every per-fly metric (interaction events,
significant pushes, "aha moment", chamber/corridor metrics, leg
visibility, etc.) lives in
[`src/ballpushing_utils/README_Ballpushing_metrics.md`](src/ballpushing_utils/README_Ballpushing_metrics.md).

---

## Installation

Requires **Python ≥ 3.10**.

Installation should take between 30 seconds and 5 minutes depending on your package manager, computer, and internet speed.

All python dependencies are specified in the [`pyproject.toml`](pyproject.toml) file and will be automatically installed when following the instructions below.

This has been tested on Ubuntu 24.04 with Python 3.10, and 3.12, but should be compatible with other operating systems and Python versions.

```bash
# Sample videos + SLEAP tracks under tests/fixtures/ are stored via Git LFS.
# Install it once (https://git-lfs.com), otherwise `git clone` will download
# LFS-pointer stubs and the data-gated tests + notebooks will stay skipped.
git lfs install

git clone https://github.com/Nely-EPFL/ballpushing_utils.git
cd ballpushing_utils

# Create an environment (conda or venv — either is fine).
python -m venv .venv
source .venv/bin/activate

# Install the package in editable mode.
pip install -e .

# Optional extras:
pip install -e ".[interactive]"   # bokeh / panel / shiny dashboards
pip install -e ".[video]"         # moviepy / pygame for video overlays
pip install -e ".[dev]"           # pytest, black, ruff
pip install -e ".[docs]"          # jupyter / nbconvert (for the walkthrough notebooks)
pip install -e ".[all]"           # everything
```

If you only want the source code (no binary assets), prefix the clone with
`GIT_LFS_SKIP_SMUDGE=1 git clone …` — data-gated tests will skip cleanly.
See [`tests/fixtures/README.md`](tests/fixtures/README.md) for what's in
the sample fixture and how to regenerate it from your own recordings.

`ballpushing_utils` depends on
[`utils_behavior`](https://github.com/NeLy-EPFL/utils_behavior), declared
as a direct `git+https://` dependency in `pyproject.toml` so
`pip install -e .` resolves without a private index.

> **Troubleshooting.** If `pip install` inside the venv fails with
> `No module named pip`, the venv was created without pip (some distros
> ship a `python` without ensurepip). Bootstrap it once with
> `.venv/bin/python -m ensurepip --upgrade`, then retry. If `pip` outside
> the venv still points at a system / conda interpreter, call the venv's
> pip explicitly: `.venv/bin/pip install …`.

---

## Configuration: data & figure paths

All scripts resolve dataset paths relative to **`BALLPUSHING_DATA_ROOT`**
and write outputs under **`BALLPUSHING_FIGURES_ROOT`**. Set them however
you prefer — `.env`, shell `export`, or your launcher of choice.

Copy the template and edit:

```bash
cp .env.example .env
$EDITOR .env
```

```ini
# .env
BALLPUSHING_DATA_ROOT=/path/to/dataverse/extract
BALLPUSHING_FIGURES_ROOT=/path/where/figures/should/land
```

To pick up `.env` automatically inside Python:

```python
from ballpushing_utils import load_dotenv
load_dotenv()  # reads ./.env
```

If unset, `BALLPUSHING_DATA_ROOT` resolves in this order: EPFL lab
share (`/mnt/upramdya_data/MD`) if mounted → `<repo>/Datasets/` (the
default destination for `ballpushing-fetch`). `BALLPUSHING_FIGURES_ROOT`
defaults to `<data root>/Affordance_Figures`.

`ballpushing-fetch` populates `<repo>/Datasets/` with the published
Dataverse feathers (see [Reproducing figures from
Dataverse](#reproducing-figures-from-the-dataverse)). After it runs,
every figure script reproduces unmodified — `ballpushing_utils.dataset(…)`
maps the on-server paths the scripts ask for to the Dataverse-published
basenames via
[`src/ballpushing_utils/dataverse_naming.py`](src/ballpushing_utils/dataverse_naming.py).

---

## Where data comes from

Scripts in this repo look for experiment data in three places, in
order of preference:

1. **Published Dataverse archive** (everything you need to reproduce
   the paper without lab access — see the next section). Fetch with
   `ballpushing-fetch`; the feathers land in `<repo>/Datasets/` by
   default (or `$BALLPUSHING_DATA_ROOT` if set). The figure scripts
   run unchanged via the Dataverse-alias resolver.
2. **Lab share / your own server** (the on-rig layout, with
   `Metadata.json` next to per-arena/corridor SLEAP tracks). Set
   `BALLPUSHING_DATA_ROOT` to the share, point `dataset_builder.py` at
   it via `--yaml`, and figure scripts read the bundled feathers via
   `dataset(...)`. This is what was used to build the paper. The
   pipeline rerun from raw HDF5 tracks uses `--dataverse-root` instead
   of `--yaml`.
3. **Bundled sample data** at `tests/fixtures/sample_data/` (one F1 +
   one TNT + one MagnetBlock fly via Git LFS). The walkthrough
   notebooks under `notebooks/` are wired up against it. Use this to
   sanity-check your install before downloading anything.

When a script can't find the data it expects (a missing feather or a
`--dataverse-root` that's empty), it raises a structured error
listing all three options — produced by
`ballpushing_utils.missing_data_message(...)`. Use
`ballpushing_utils.detect_layout(<dir>)` to classify a directory
yourself: it returns `"server"`, `"dataverse"`, or `None`.

## Dataverse archive

> **See [`DATAVERSE.md`](DATAVERSE.md) for the full per-archive
> reference**: every archive name → which feather column(s) it
> populates, the rebuild-from-h5 invocation per paradigm, and the
> mapping cheat-sheet. The section below is the at-a-glance summary;
> `DATAVERSE.md` is the lookup you'll want open while doing a rerun.

Three Dataverse datasets accompany the paper. Each ships per-condition
`.tar` archives of SLEAP HDF5 tracks, plus pre-computed feathers and
the `Config.json` snapshot used to build them.

| Dataset | Paradigm coverage | Conditions sorted by | Top-level folders |
|---------|-------------------|----------------------|-------------------|
| **Affordance** | F1 pretraining + MagnetBlock affordance learning (Fig. 2, ED Fig. 2) | `F1_condition` (F1) / `Magnet` (MagnetBlock) | `MagnetBlock/`, `F1_Tracks/` |
| **Screen** | Brain-region TNT silencing screen + F1-TNT (Fig. 3, ED Fig. 6) | `Genotype` | `Ballpushing_TNTScreen/`, `F1_Tracks/` |
| **Exploration** | Wild-type baseline, ball types, ball scents, feeding state, dark olfaction, learning mutants, broad TNT screen (Fig. 1, ED Figs. 3 / 7–10) | per-paradigm column (see invocation table below) | `Ballpushing_Exploration/`, `Ballpushing_Balltypes/`, `Ballpushing_Ballscents/`, … |

### Layout inside each Dataverse dataset

Each archive is **flat** — every feather lives at the dataset root,
named after the paradigm it covers:

```
<dataverse-dataset>/
├── <Paradigm>_ballpushing_metrics.feather   # Per-fly metrics (one row per fly).
├── <Paradigm>_trajectories.feather          # Pooled trajectories (one row per frame).
│   # Files >2.5 GiB are split into <Paradigm>_trajectories-1.feather,
│   # <Paradigm>_trajectories-2.feather, … (each part holds whole flies).
├── <Paradigm>_config.json                   # `Config` snapshot used to build the feathers.
└── <Condition>.tar                          # Per-condition SLEAP tracks (only
                                             #   needed for the rerun-from-raw path).
```

`<Paradigm>` is e.g. `Magnetblock`, `Generalisation-Wild-type`,
`Ballscents`, `Wild-Type`. Inside each tar, files are organised **by
condition**: `<Condition>/<YYMMDD>[-N]/arenaN/<corridorM|Left|Right>/`
with the SLEAP `*ball*.h5`, `*fly*.h5`, and (where present)
`*full_body*.h5` tracks. The `ballpushing_utils.dataverse` module knows
how to walk this tree and synthesise the per-arena metadata `Fly()`
would normally read from `Metadata.json` — see
[Rerunning the pipeline from raw HDF5 tracks](#rerunning-the-pipeline-from-raw-hdf5-tracks).

### Reproducing figures from the Dataverse

Three commands from a fresh clone:

```bash
pip install -e .
ballpushing-fetch                  # downloads ~9 GB of feathers into ./Datasets/
python run_all_figures.py          # writes every panel under $BALLPUSHING_FIGURES_ROOT
```

`ballpushing-fetch` reads the file list from `figures/**/*.py`, queries
the three published Dataverse archives (Affordance, Screen, Exploration
— see DOIs at the top of the README), and downloads only the feathers
the figures actually need. Subsequent calls are idempotent (skip files
already on disk with matching size). Useful options:

```bash
ballpushing-fetch --dry-run                    # show what would be fetched, exit
ballpushing-fetch --archive affordance         # restrict to one archive (repeatable)
ballpushing-fetch --dest /path/to/feathers     # override the destination
ballpushing-fetch --include-raw                # also fetch the raw HDF5 track tars +
                                               #   grid videos (~tens of GB)
ballpushing-fetch --verify-md5                 # checksum each downloaded file
```

Files land in `./Datasets/` by default. Set `BALLPUSHING_DATA_ROOT` to
download somewhere else; the figure scripts honour the same variable
and fall back to `./Datasets/` when it's unset. The handle for the
filename translation is
[`src/ballpushing_utils/dataverse_naming.py`](src/ballpushing_utils/dataverse_naming.py)
— add an entry there if you publish a new feather. See the figure ↔
feather mapping below for the exact per-panel correspondence.

### Rerunning the pipeline from raw HDF5 tracks

If you want to regenerate the feathers from scratch (for example to
change a threshold in `Config`, add a new metric, or audit the pipeline
end-to-end), call `dataset_builder.py` with `--dataverse-root` pointing
at one of the `Videos/` subtrees. The builder walks
`<root>/<condition>/<date>/arenaN/corridorM/` automatically and
synthesises the per-fly metadata from the condition folder name.

The condition folder name expands into one or more feather columns,
depending on the archive. Below is the canonical invocation per
paradigm — pass these verbatim and the column you'd filter on in a
figure script will be populated:

| Archive subtree | Invocation | Columns the condition folder populates |
|---|---|---|
| `MagnetBlock/Videos/` | `--experiment-type MagnetBlock` | `Magnet` (`y` / `n`) |
| `F1_Tracks/Videos/` | `--experiment-type F1` | `F1_condition` *and* `Pretraining` (the F1 transformer derives both: `control` → `Pretraining=n`; anything else → `Pretraining=y`) |
| `Ballpushing_TNTScreen/Videos/` | `--experiment-type TNT` | `Genotype` |
| `Ballpushing_Balltypes/Videos/` | `--experiment-type TNT --condition-field BallType` | `BallType` |
| `Ballpushing_Ballscents/Videos/` | `--experiment-type TNT --condition-field FeedingState` | `FeedingState` |
| Wild-type × light archives | `--experiment-type TNT --condition-field Light` | `Light` |
| Feeding state archives | `--experiment-type TNT --condition-field FeedingState` | `FeedingState` |
| Period archives | `--experiment-type TNT --condition-field Period` | `Period` |

```bash
# MagnetBlock subset of the Affordance archive (one column: Magnet)
python src/dataset_builder.py \
    --dataverse-root $BALLPUSHING_DATA_ROOT/MagnetBlock/Videos \
    --experiment-type MagnetBlock \
    --datasets summary coordinates

# F1 subset of the Affordance archive (transformer: F1_condition + Pretraining)
python src/dataset_builder.py \
    --dataverse-root $BALLPUSHING_DATA_ROOT/F1_Tracks/Videos \
    --experiment-type F1 \
    --datasets summary F1_coordinates fly_positions

# Screen (one column: Genotype)
python src/dataset_builder.py \
    --dataverse-root $BALLPUSHING_DATA_ROOT/Ballpushing_TNTScreen/Videos \
    --experiment-type TNT \
    --datasets summary

# Balltype (one column: BallType — same experiment_type as the screen)
python src/dataset_builder.py \
    --dataverse-root $BALLPUSHING_DATA_ROOT/Ballpushing_Balltypes/Videos \
    --experiment-type TNT --condition-field BallType \
    --datasets summary coordinates
```

Per-paradigm defaults live in
`ballpushing_utils.dataverse.DEFAULT_CONDITION_FIELD`; the F1
transformer that derives `Pretraining` from `F1_condition` lives in
`ballpushing_utils.dataverse.CONDITION_TRANSFORMERS`. Inspect the
columns a folder will populate without launching a build:

```python
from ballpushing_utils.dataverse import expand_condition
expand_condition("F1", "pretrained_unlocked")
# {'Pretraining': 'y', 'F1_condition': 'pretrained_unlocked'}
expand_condition("TNT", "Marble", condition_field="BallType")
# {'BallType': 'Marble'}
```

#### Caveats when running from the Dataverse layout

The `Videos/` tree intentionally ships only the SLEAP HDF5 files. A few
ancillary assets that exist on the recording server are *not* in the
archive; the builder degrades gracefully but you should know what's
missing:

- **No `Metadata.json`.** Per-arena metadata is synthesised from the
  condition folder name (see invocation table above). For F1 the
  publisher already split flies by `pretrained` / `pretrained_unlocked`
  / `control` (i.e. the on-rig Left/Right + Unlocked logic was applied
  *before* upload, so the folder name is the right `F1_condition` for
  every fly inside it). Secondary fields the on-rig `Metadata.json`
  would normally carry (`Unlocked`, plus replicate-level annotations
  like `Period` / `Light` / `FeedingState` when those aren't the
  primary condition for the paradigm) are absent. Panels that need to
  *cross* two annotations (e.g. `Genotype × Light`,
  `Genotype × FeedingState`) won't reproduce from a single-archive
  rerun: regenerate them against the bundled pooled feathers, which
  were built from the on-rig data before sorting.
- **No `.mp4` videos.** The Dataverse archives ship tracks only — raw
  videos are too large. Sample / grid videos may be uploaded for some
  paradigms (e.g. balltype demos) separately; if you need raw
  recordings, contact the lab. The package detects the missing video
  and skips video-export helpers (`Fly.generate_clips`,
  `Fly.generate_preview`) cleanly. `SkeletonMetrics` no longer needs
  the original video dimensions: it derives the per-fly raw → template
  affine transform empirically from matched thorax positions in the
  fly tracker (raw video coords) and the skeleton tracker (template
  coords). See
  `SkeletonMetrics._estimate_raw_to_template_transform`. The legacy
  geometry-based math (`Config.default_video_size`) remains as a
  fallback when the skeleton track is missing or has too few
  overlapping frames.
- **No `fps.npy`.** The package defaults to **29 fps**, which is the
  canonical MultiMazeRecorder / F1 rig rate used throughout paper
  acquisition. All time-based metrics in the bundled feathers assume
  this, so the rerun matches bit-for-bit.
- **`Region_map_*.csv` (Screen archive only).** Used by the screen
  heatmap, the dendrogram, and `FlyMetadata.load_brain_regions` to
  resolve genotype → nickname / brain-region. The Screen Dataverse
  archive ships it at the root of its extract; download and place it
  under `$BALLPUSHING_DATA_ROOT` to populate the `Nickname` and
  `Brain region` columns. If absent, `load_brain_regions` skips
  silently with default values (`Nickname="PR"`, `Brain region="Control"`)
  — non-screen paradigms work fine without it.

If your goal is figure reproduction, prefer the bundled feathers. If
your goal is downstream custom analysis on individual flies, the
`ballpushing_utils.Fly(directory, dataverse_condition={…})` constructor
is the canonical entry point — it wires up the same synthetic-metadata
path the builder uses.

#### Downloading a single feather without the full archive

You don't have to recreate the `<archive>/Datasets/<timestamp>/...`
hierarchy. `read_feather` falls back to a basename search via
`paths.find_feather`:

1. Drop the feather anywhere — under `BALLPUSHING_DATA_ROOT` or in a
   separate folder.
2. If you used a separate folder, point at it with
   `BALLPUSHING_FEATHER_SEARCH`:

   ```bash
   export BALLPUSHING_FEATHER_SEARCH=$HOME/Downloads/dataverse_feathers
   python figures/Fig2-Affordance/fig2_magnetblock_first_major_push_time.py
   ```

3. The figure script's `dataset(...)` call still names the canonical
   relative path, but the resolver will pick up your file by
   basename. If two feathers share a basename in the search tree, the
   warning identifies them so you can disambiguate by setting
   `BALLPUSHING_FEATHER_SEARCH` more narrowly or by placing one of
   them at the canonical path.

---

## Quickstart

### Reproduce a single paper panel from the dataverse feathers

```bash
# Once: fetch the Dataverse feathers (skips files already on disk).
ballpushing-fetch

python figures/Fig2-Affordance/fig2_magnetblock_first_major_push_time.py
# -> writes  $BALLPUSHING_FIGURES_ROOT/Figure2/<script-stem>/*.pdf
#         + a *_stats.csv with the published p-value alongside it.
```

Each script accepts `--test` to run on a 200-row subsample for a quick
smoke test.

### Reproduce *every* paper panel in one shot

```bash
python run_all_figures.py
```

This auto-discovers all `*.py` under `figures/`, runs each in its own
subprocess, and prints a green/red pass-fail summary. Figures land under
`BALLPUSHING_FIGURES_ROOT`.

### Use the library on your own recordings

For a guided tour of how `Fly`, `Experiment`, and `Dataset` bind
tracking data, metadata, and config together — with runnable cells
against a real fly folder — start with
[`notebooks/ballpushing_utils_walkthrough.ipynb`](notebooks/ballpushing_utils_walkthrough.ipynb).
Two companion notebooks drill into the two things most users want
next:

- [`notebooks/ballpushing_metrics_reference.ipynb`](notebooks/ballpushing_metrics_reference.ipynb)
  is a live, per-metric reference paired with
  `src/ballpushing_utils/README_Ballpushing_metrics.md` — every
  metric in `fly.event_summaries` is printed with its current value
  and a one-line description.
- [`notebooks/dataset_types_guide.ipynb`](notebooks/dataset_types_guide.ipynb)
  tours every `dataset_type` you can request
  (`summary`, `coordinates`, `fly_positions`, `event_metrics`,
  `F1_coordinates`, `F1_checkpoints`, `contact_data`,
  `Skeleton_contacts`, `standardized_contacts`, `transformed`,
  `transposed`, `behavior_umap`) with subsections for the
  preconditions (F1 experiment type, skeleton tracks,
  Learning paradigm, …).

A quick taster:

```python
from ballpushing_utils import Experiment

# Point at a folder containing one experiment (multiple arenas of flies).
exp = Experiment("/path/to/experiment_directory")

for fly in exp.flies:
    # Each metric family is a cached dict on the Fly object — touching the
    # property triggers the computation the first time and caches it.
    summaries = fly.event_summaries        # ball-pushing summary metrics
    print(fly.metadata.name, summaries.get("first_major_event_time"))
```

Other metric families exposed on `Fly`: `event_metrics` (per-event tables),
`f1_metrics` (F1-paradigm only), `learning_metrics`, and the underlying
`tracking_data` (a `FlyTrackingData`). See
[`src/ballpushing_utils/README_Ballpushing_metrics.md`](src/ballpushing_utils/README_Ballpushing_metrics.md)
for the full metric reference.

A worked example for a single panel using the shared plotting/stats
helpers:

```python
import pandas as pd
import matplotlib.pyplot as plt
from ballpushing_utils import dataset, figure_output_dir
from ballpushing_utils.plotting import (
    paired_boxplot_with_significance, resize_axes_cm, set_illustrator_style,
)
from ballpushing_utils.stats import permutation_test

set_illustrator_style()
df = pd.read_feather(dataset("MagnetBlock/.../pooled_summary.feather"))
control = df.loc[df.Magnet == "n", "first_major_event_time"].to_numpy()
test    = df.loc[df.Magnet == "y", "first_major_event_time"].to_numpy()

perm = permutation_test(control, test, statistic="median", n_permutations=10_000)

fig, ax = plt.subplots()
paired_boxplot_with_significance(ax, [control, test], p_value=perm.p_value)
resize_axes_cm(fig, ax, width_cm=1.75, height_cm=2.25)
fig.savefig(figure_output_dir("MyFig", __file__) / "panel.pdf", dpi=300)
```

The permutation test seeds the legacy NumPy `RandomState(42)` so the
p-values it returns match the published values bit-for-bit.

### Diagnostics

When a recording looks wrong (events mis-classified, metrics out of
range, NaNs appearing) start with the diagnostics layer. The builders
under `ballpushing_utils.diagnostics` return plain `DataFrame`s +
`matplotlib.Figure`s, so they're equally at home in a script, a
notebook, or a dashboard:

- `notebooks/diagnostics_demo.ipynb` walks through `build_event_timeline`
  and `build_metric_report` against a **stub fly** — runs offline so
  it doubles as a smoke test for new installs.
- `python tools/diagnostics_dashboard.py <fly_path>` serves an
  interactive Panel app with the event table, the Gantt-style timeline
  (thresholds tweakable via sliders), and the metric-range report.
- `write_report(...)` materialises any report into a per-run folder
  with `summary.md`, per-section CSVs, and `plots/*.png`.

The hermetic invariants of these builders are locked down in
`tests/unit/diagnostics/`, which is what CI runs on every push.

---

## Figure ↔ feather mapping

Each figure script resolves its dataset path through
`ballpushing_utils.dataset(...)`. The "Server path" column lists the
literal each script asks for (paths are relative to
`$BALLPUSHING_DATA_ROOT`); the "Dataverse filename" column lists the
basename `ballpushing-fetch` downloads to satisfy that request. The
translation table lives in
[`src/ballpushing_utils/dataverse_naming.py`](src/ballpushing_utils/dataverse_naming.py)
— add an entry there if you publish a new feather.

| Paper figure | Script(s) | Server path | Dataverse filename | Archive |
|---|---|---|---|---|
| **Fig. 1** — setup & wild-type baseline | `figures/Fig1-setup/learning_trials_duration.py` | `BallPushing_Learning/.../250320_Annotated_data.feather` | `Multi-trials_trajectories.feather` (split) | Exploration |
|  | `figures/Fig1-setup/plot_wildtype_trajectories.py`<br>`figures/Fig1-setup/compute_distribution_stats.py` | per-fly coordinates (see *Dual-workflow scripts*) | `Wild-type_Lights-*_*_trajectories.feather` (sliced by experiment) | Exploration |
|  | `figures/Fig1-setup/plot_simulation_trajectories.py` | (synthetic; reads no Dataverse feather) | — | — |
| **Fig. 2** — affordance (MagnetBlock + F1) | `figures/Fig2-Affordance/fig2_magnetblock_first_major_push_time.py`<br>`figures/Fig2-Affordance/fig2_magnetblock_first_major_push_index.py` | `MagnetBlock/.../summary/pooled_summary.feather` | `Magnetblock_ballpushing_metrics.feather` | Affordance |
|  | `figures/Fig2-Affordance/plot_magnetblock_trajectories.py` | `MagnetBlock/.../coordinates/pooled_coordinates.feather` | `Magnetblock_trajectories.feather` | Affordance |
|  | `figures/Fig2-Affordance/fig2_f1_control_conditions.py` | `F1_Tracks/.../F1_New_Data/summary/pooled_summary.feather` | `Generalisation-Wild-type_ballpushing_metrics.feather` | Affordance |
| **Fig. 3** — neural silencing screen | `figures/Fig3-Screen/fig3_screen_heatmap.py` | `Ballpushing_TNTScreen/.../summary/pooled_summary.feather` + `Region_map_*.csv` | `ballpushing_metrics_silencing_screen.feather` | Screen |
|  | `figures/Fig3-Screen/fig3_f1_tnt.py` | `F1_Tracks/.../F1_TNT_Full_Data/summary/pooled_summary.feather` | `Generalisation-TNT_ballpushing_metrics.feather` | Affordance |
| **ED Fig. 2** — MagnetBlock speeds | `figures/EDFigure2-Magnetblock_speeds/edfigure2_abc_speeds.py` | `MagnetBlock/.../{summary,coordinates}` | `Magnetblock_ballpushing_metrics.feather` + `Magnetblock_trajectories.feather` | Affordance |
| **ED Fig. 3** — wild-type × light state | `figures/EDFigure3-Wild-type_Light/edfigure3_b_summary_metrics.py` | `Ballpushing_Exploration/.../summary/pooled_summary.feather` | `Wild-Type_ballpushing_metrics.feather` | Exploration |
|  | `figures/EDFigure3-Wild-type_Light/edfigure3_a_trajectories.py` | coordinates directory (see *Dual-workflow scripts*) | `Wild-type_Lights-{on,off}_{Fed,Starved,Starved-without-water}_trajectories.feather` | Exploration |
| **ED Fig. 6** — behavioural dendrogram | `figures/EDFigure6-Dendrogram/edfigure6_dendrogram.py` | `Ballpushing_TNTScreen/.../summary/pooled_summary.feather` | `ballpushing_metrics_silencing_screen.feather` | Screen |
| **ED Fig. 7** — ball types | `figures/EDFigure7-Balltypes/edfigure7_b_first_push_time.py` | `Ballpushing_Balltypes/.../summary/pooled_summary.feather` | `Balltypes_ballpushing_metrics.feather` | Exploration |
|  | `figures/EDFigure7-Balltypes/edfigure7_c_trajectories.py` | `Ballpushing_Balltypes/.../coordinates/pooled_coordinates.feather` | `Balltypes_trajectories.feather` | Exploration |
| **ED Fig. 8** — ball scents | `figures/EDFigure8-Ballscents/edfigure8_abc_metrics.py` | `Ball_scents/.../summary/pooled_summary.feather` + slice of pooled wild-type | `Ballscents_ballpushing_metrics.feather` + `Wild-Type_ballpushing_metrics.feather` | Exploration |
|  | `figures/EDFigure8-Ballscents/edfigure8_def_trajectories.py` | `Ball_scents/.../coordinates/pooled_coordinates.feather` | `Ballscents_trajectories.feather` | Exploration |
| **ED Fig. 9** — IR8a × light | `figures/EDFigure9-Ir8a_Light/edfigure9_b_pulling_ratio.py` | `TNT_Olfaction_Dark/.../summary/pooled_summary.feather` | `TNTxIR8a-dark_ballpushing_metrics.feather` | Exploration |
|  | `figures/EDFigure9-Ir8a_Light/edfigure9_a_trajectories.py` | `TNT_Olfaction_Dark/.../coordinates/pooled_coordinates.feather` | `TNTxIR8a-dark_trajectories.feather` | Exploration |
| **ED Fig. 10** — feeding state | `figures/EDFigure10-FeedingStates/edfigure10_c_metrics_significant.py`<br>`figures/EDFigure10-FeedingStates/edfigure10_d_metrics_nonsignificant.py` | `Ballpushing_Exploration/.../summary/pooled_summary.feather` | `Wild-Type_ballpushing_metrics.feather` | Exploration |
|  | `figures/EDFigure10-FeedingStates/edfigure10_a_trajectories.py`<br>`figures/EDFigure10-FeedingStates/edfigure10_b_final_position.py` | coordinates directory (see *Dual-workflow scripts*) | `Wild-type_Lights-{on,off}_{Fed,Starved,Starved-without-water}_trajectories.feather` | Exploration |

### Dual-workflow scripts (per-fly coordinate iteration)

Five figure scripts originally iterated per-experiment
`*_coordinates.feather` files inside `Ballpushing_Exploration/.../coordinates/`.
The Dataverse archive publishes the same data pooled by condition
(`Wild-type_Lights-{on,off}_{Fed,Starved,Starved-without-water}_trajectories.feather`),
so these scripts now route through
[`ballpushing_utils.iter_coordinate_feathers`](src/ballpushing_utils/compat.py)
which transparently picks the right layout:

- **On-server**: yields one `(file_stem, df)` per `*_coordinates.feather`
  in the directory.
- **Dataverse**: opens each per-condition pool, splits it by the
  `experiment` column, and yields one `(experiment_name, df)` per
  experiment.

The downstream filter / downsample / fly-namespacing logic in each
script is identical for both layouts. The scripts:

- `figures/Fig1-setup/plot_wildtype_trajectories.py` (Fig 1e wild-type
  trajectories — default mode reads a specific FeedingState cohort via
  `load_wildtype_experiment(...)` when the per-fly file is missing).
- `figures/Fig1-setup/compute_distribution_stats.py` (Fig 1 stats on
  the same cohort, same fallback).
- `figures/EDFigure3-Wild-type_Light/edfigure3_a_trajectories.py`,
  `figures/EDFigure10-FeedingStates/edfigure10_a_trajectories.py`, and
  `figures/EDFigure10-FeedingStates/edfigure10_b_final_position.py`
  (directory iteration).

The pattern lookup table is in
[`src/ballpushing_utils/dataverse_naming.py`](src/ballpushing_utils/dataverse_naming.py)
under `SERVER_DIRECTORY_TO_DATAVERSE`; add an entry there if a new
script needs to iterate a different on-server coordinates directory.

Supplementary panels (feeding-state, ball scents, ball types, dark
olfaction, learning mutants, broad TNT screen, etc.) live under
`plots/Supplementary_exps/` and `plots/Ballpushing_PR/`. They follow the
same `script.py → PDF + stats.csv` convention as the figure scripts.

---

## Building feathers from raw H5 SLEAP tracks

The Dataverse archives ship the per-fly summary feathers used by every
figure script, so most users will never need this step. If you do want
to re-process raw tracks, two paths are available:

### From the Dataverse archive (`Condition/date/arenaN/corridorM/`)

```bash
python src/dataset_builder.py \
    --dataverse-root $BALLPUSHING_DATA_ROOT/<archive>/Videos \
    --experiment-type {TNT|MagnetBlock|F1|Learning} \
    --datasets summary [coordinates …]
```

See [Rerunning the pipeline from raw HDF5
tracks](#rerunning-the-pipeline-from-raw-hdf5-tracks) for the full set
of paradigm invocations and the documented limitations (no per-fly
`Light`/`Period`/`Unlocked` columns; no video/fps assets — defaults
apply).

### From the on-rig server layout (`<experiment>/Metadata.json + arenaN/corridorM/`)

This is what `dataset_builder.py` was written for and is what the
`config.json` shipped in the Dataverse `Datasets/` folders documents.
You only need this path if you have your own recordings:

1. **Drop SLEAP `.h5` exports** under `$BALLPUSHING_DATA_ROOT/<experiment>/`,
   alongside a `Metadata.json` describing each arena's condition.
2. **Describe the experiment batch** in a YAML file under
   `experiments_yaml/` (genotypes, replicate dates, conditions). See
   any of the existing files for the schema.
3. **Build the dataset:**
   `python src/dataset_builder.py --yaml <yaml> --datasets summary` produces
   per-fly metric tables and a pooled `pooled_<metric>.feather` per
   metric.
4. Run the figure scripts as usual.

---

## Tests

Each test suite should take ~5-30 seconds to run.

```bash
pip install -e ".[dev]"
pytest tests/unit         # hermetic suite — no SLEAP data required
pytest tests/integration  # integration suite — needs BALLPUSHING_DATA_ROOT
```

`tests/unit/` runs against stub flies/experiments and is what CI
executes on every push
(see [`.github/workflows/tests.yml`](.github/workflows/tests.yml)).
It covers the diagnostics builders
(`ballpushing_utils.diagnostics.{event_timeline,metric_report,report}`)
and the reproducibility contracts of the permutation test
(`ballpushing_utils.stats.permutation_test`, both the legacy
`RandomState` / median path and the screen-panel
`default_rng` / mean / `plus_one` path).

`tests/integration/` is currently mid-triage — see
[`tests/integration/REVIEW.md`](tests/integration/REVIEW.md) for the
per-file plan.

Configuration lives in `pyproject.toml` under `[tool.pytest.ini_options]`.

---

## Development

The project uses Black, Ruff, and pytest, all configured in
`pyproject.toml`. Recommended workflow:

```bash
black src tests figures
ruff check src tests figures
pytest
```

Hardcoded data paths in any new script will fail review — always go
through `ballpushing_utils.paths.dataset(...)` and
`ballpushing_utils.paths.figure_output_dir(...)`.

---

## Contributors

The package and the paper figures it generates were authored by
**Matthias Durrieu** (lead, Ramdya Lab @ EPFL) with the following
contributions from collaborators inside the lab:

- **Tommy Lam** ([@tkclam](https://github.com/tkclam)) — UMAP analysis
  pipeline of contact-event kinematics and the figures that build on
  it. Specifically:
  - `src/ballpushing_utils/umap/` — `preprocess.py`, `analysis.py`,
    `utils.py` (feature matrix, UMAP fit + custom flip-aware metric,
    cluster maps, KDE overlays, energy-test pipeline, grid-video
    helper).
  - `src/ballpushing_utils/stats/{energy_test,kde}.py` — energy
    statistic for two-sample comparisons in embedding space and
    KDE helpers used by the UMAP figures.
  - `figures/Fig3-Screen/{fig3_umap,fig3_contact_image,fig3_kinematic_features}.py`
    + `figures/EDFigure5-UMAP/edfigure5_umap.py` +
    `figures/SuppInfo/File2_umaps.py` +
    `figures/SuppVideo/Video8_9.py`.
  - `figures/EDFigure4-Confocal/edfigure4_confocal_stacks.py` —
    confocal stack registration / quantification for ED Fig. 4.
  - `notebooks/ball_tracking/` — companion ball-tracking pipeline
    (template matching + homography) used during method development.
  - `src/ballpushing_utils/preprocess_screen_data.py` — pre-aggregation
    feeding into the UMAP feature matrix.
- **Dominic Dall'Osto**
  ([@Dominic-DallOsto](https://github.com/Dominic-DallOsto)) —
  high-resolution ball-pushing analysis used for ED Fig. 1.
  Specifically:
  - `src/ball_pushing_high_res/` — sibling package (`config.py`,
    `df_utils.py`, `plot_utils.py`, `stat_utils.py`) holding the
    polars-/plotnine-based dataframe helpers, fly-pose plotting, and
    permutation-test helpers used to classify early contact events.
  - `figures/EDFigure1-HighRes/edfigure1_early_contact_classification.ipynb`
    — the notebook that produces the panel.

If you reuse a specific figure or pipeline component, please include
the relevant contributor in your acknowledgements alongside the paper
citation.

---

## License & citation

Source code: MIT. © 2024–2026 Neuroengineering Laboratory @EPFL — Ramdya
Lab. See [`LICENSE`](LICENSE).

Please cite [the paper above]([https://doi.org/10.64898/2026.04.28.721021]) when using the library, the metrics, or the
dataset in your work.
