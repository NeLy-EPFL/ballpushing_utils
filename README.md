# ballpushing_utils

Analysis library and figure-reproduction code for the *Drosophila*
ball-pushing paradigm developed in the [Ramdya Lab](https://www.epfl.ch/labs/ramdya-lab/)
at EPFL. It computes behavioural metrics from SLEAP-tracked recordings of
flies interacting with a ball in a corridor, and contains the scripts that
generate every panel in the paper companion of this repository.

> **Paper citation:** *TODO — paste BioRxiv / journal DOI when available
> (Durrieu et al., 2026, "Object manipulation and affordance learning in
> Drosophila").*
>
> **Dataset:** Three companion Dataverse archives host the data:
> *Affordance* (F1 + MagnetBlock, *TODO DOI*), *Screen* (TNT silencing
> screen, *TODO DOI*), and *Exploration* (every other paradigm —
> wild-type baselines, ball types, ball scents, feeding state, dark
> olfaction, learning mutants, *TODO DOI*). Each archive bundles the
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

```bash
# Sample videos + SLEAP tracks under tests/fixtures/ are stored via Git LFS.
# Install it once (https://git-lfs.com), otherwise `git clone` will download
# LFS-pointer stubs and the data-gated tests + notebooks will stay skipped.
git lfs install

git clone https://github.com/<TODO-org>/ballpushing_utils.git
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

If unset, `BALLPUSHING_DATA_ROOT` defaults to the EPFL lab share
(`/mnt/upramdya_data/MD`) and `BALLPUSHING_FIGURES_ROOT` defaults to
`<data root>/Affordance_Figures`.

When you point `BALLPUSHING_DATA_ROOT` at the unpacked Dataverse bundles
(see next section), every figure script runs unmodified — paths under
the data root are resolved through `ballpushing_utils.dataset(…)` and
mirror the layout the scripts expect.

---

## Where data comes from

Scripts in this repo look for experiment data in three places, in
order of preference:

1. **Lab share / your own server** (the on-rig layout, with
   `Metadata.json` next to per-arena/corridor SLEAP tracks). Set
   `BALLPUSHING_DATA_ROOT` to the share, point `dataset_builder.py` at
   it via `--yaml`, and figure scripts read the bundled feathers via
   `dataset(...)`. This is what was used to build the paper.
2. **Published Dataverse archive** (everything you need to reproduce
   the paper without lab access — see the next section). Extract under
   `BALLPUSHING_DATA_ROOT`. The figure scripts run unchanged; the
   pipeline rerun uses `--dataverse-root` instead of `--yaml`.
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

Three Dataverse datasets accompany the paper. Each ships the raw SLEAP
tracks for every fly that contributed to a panel, the pooled feathers
the figure scripts read, and the `config.json` produced by
`dataset_builder.py` when the feathers were built (so you can audit or
reproduce the metric thresholds, time-range cutoffs, etc.).

| Dataset | Paradigm coverage | Conditions sorted by | Top-level folders |
|---------|-------------------|----------------------|-------------------|
| **Affordance** | F1 pretraining + MagnetBlock affordance learning (Fig. 2, ED Fig. 2) | `F1_condition` (F1) / `Magnet` (MagnetBlock) | `MagnetBlock/`, `F1_Tracks/` |
| **Screen** | Brain-region TNT silencing screen + F1-TNT (Fig. 3, ED Fig. 6) | `Genotype` | `Ballpushing_TNTScreen/`, `F1_Tracks/` |
| **Exploration** | Wild-type baseline, ball types, ball scents, feeding state, dark olfaction, learning mutants, broad TNT screen (Fig. 1, ED Figs. 3 / 7–10) | per-paradigm column (see invocation table below) | `Ballpushing_Exploration/`, `Ballpushing_Balltypes/`, `Ballpushing_Ballscents/`, … |

### Layout inside each Dataverse dataset

Two parallel trees live side by side:

```
<dataverse-dataset>/
├── Datasets/                       # Pooled feathers (drop-in for figures)
│   └── <YYMMDD_HH>_<metric>_<source>_Data/
│       ├── config.json             # CONFIG dict + ballpushing Config snapshot
│       ├── summary/pooled_summary.feather
│       ├── coordinates/pooled_coordinates.feather
│       └── …                       # one subfolder per dataset_type built
└── Videos/                         # Raw SLEAP tracks, sorted by condition
    └── <Condition>/                #   e.g. MB247xTNT/, Magnet_y/, Trained/
        └── <YYMMDD>[-N]/           #   acquisition date; -2/-3 if the same
            └── arenaN/             #   day produced multiple experiments
                └── corridorM/
                    ├── *_ball*.h5    # SLEAP ball track
                    ├── *_fly*.h5     # SLEAP fly track
                    └── *_full_body*.h5  # SLEAP skeleton track
```

The `Videos/` tree is what the package calls the "Dataverse layout":
files are organised **by condition**, not by experiment. There is no
`Metadata.json` — the only per-fly metadata is the condition folder
name. The `ballpushing_utils.dataverse` module knows how to walk this
tree and synthesise the per-arena metadata `Fly()` would normally read
from `Metadata.json` (see
[Rerunning the pipeline from raw HDF5 tracks](#rerunning-the-pipeline-from-raw-hdf5-tracks)).

### Reproducing figures from the Dataverse

This is the lowest-friction path. Download the archives, extract them
under a single `BALLPUSHING_DATA_ROOT`, and the figure scripts will pick
up the bundled feathers verbatim:

```bash
mkdir -p $BALLPUSHING_DATA_ROOT
# Extract each archive in place — they share top-level folders
# (MagnetBlock/, F1_Tracks/, Ballpushing_TNTScreen/, Ballpushing_Exploration/, …)
unzip Affordance_Dataverse.zip   -d $BALLPUSHING_DATA_ROOT
unzip Screen_Dataverse.zip       -d $BALLPUSHING_DATA_ROOT
unzip Exploration_Dataverse.zip  -d $BALLPUSHING_DATA_ROOT

python run_all_figures.py
# -> writes every panel under $BALLPUSHING_FIGURES_ROOT
```

The exact feather each panel consumes is listed in the
[Figure ↔ feather mapping](#figure--feather-mapping) table below.

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
- **`F1_New_Template.png` (F1 heatmap only).** Background image for
  `figures/Fig2-Affordance/fig2_f1_heatmaps_pretraining.py`. Ships in
  the Affordance archive at `F1_Tracks/F1_New_Template.png`.

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
# Once (after editing .env)
export $(grep -v '^#' .env | xargs)

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
`ballpushing_utils.dataset(...)`, so paths in the table below are
relative to `$BALLPUSHING_DATA_ROOT`. The "Dataverse archive" column
points at the archive that ships each feather.

| Paper figure | Script(s) | Reads | Dataverse archive |
|---|---|---|---|
| **Fig. 1** — setup & wild-type baseline | `figures/Fig1-setup/plot_wildtype_trajectories.py`<br>`figures/Fig1-setup/plot_simulation_trajectories.py`<br>`figures/Fig1-setup/learning_trials_duration.py`<br>`figures/Fig1-setup/compute_distribution_stats.py` | wild-type trajectory + summary feathers under `Ballpushing_Exploration/Datasets/` | Exploration |
| **Fig. 2** — affordance (MagnetBlock + F1) | `figures/Fig2-Affordance/fig2_magnetblock_first_major_push_time.py`<br>`figures/Fig2-Affordance/fig2_magnetblock_first_major_push_index.py` | `MagnetBlock/Datasets/.../summary/pooled_summary.feather` | Affordance |
|  | `figures/Fig2-Affordance/plot_magnetblock_trajectories.py` | `MagnetBlock/Datasets/.../coordinates/pooled_coordinates.feather` | Affordance |
|  | `figures/Fig2-Affordance/fig2_f1_control_conditions.py` | `F1_Tracks/Datasets/.../summary/pooled_summary.feather` | Affordance |
|  | `figures/Fig2-Affordance/fig2_f1_heatmaps_pretraining.py` | `F1_Tracks/Datasets/.../fly_positions/pooled_fly_positions.feather` + `F1_Tracks/F1_New_Template.png` | Affordance |
| **Fig. 3** — neural silencing screen | `figures/Fig3-Screen/fig3_screen_heatmap.py` | `Ballpushing_TNTScreen/Datasets/.../summary/pooled_summary.feather` + `Region_map_*.csv` | Screen |
|  | `figures/Fig3-Screen/fig3_f1_tnt.py` | `F1_Tracks/Datasets/260123_*_F1_TNT_Full_Data/summary/pooled_summary.feather` | Screen |
| **ED Fig. 2** — MagnetBlock speeds | `figures/EDFigure2-Magnetblock_speeds/edfigure2_abc_speeds.py` | MagnetBlock coordinates feather | Affordance |
| **ED Fig. 3** — wild-type × light state | `figures/EDFigure3-Wild-type_Light/*.py` | `Ballpushing_Exploration/Datasets/.../summary/pooled_summary.feather` | Exploration |
| **ED Fig. 6** — behavioural dendrogram | `figures/EDFigure6-Dendrogram/edfigure6_dendrogram.py` | wild-type metric matrix from screen pooled summary | Screen |
| **ED Fig. 7** — ball types | `figures/EDFigure7-Balltypes/*.py` | `Ballpushing_Balltypes/Datasets/.../coordinates/pooled_coordinates.feather` | Exploration |
| **ED Fig. 8** — ball scents | `figures/EDFigure8-Ballscents/*.py` | `Ballpushing_Ballscents/Datasets/.../*.feather` | Exploration |
| **ED Fig. 9** — IR8a × light | `figures/EDFigure9-Ir8a_Light/*.py` | `Ballpushing_Exploration/Datasets/.../*.feather` | Exploration |
| **ED Fig. 10** — feeding state | `figures/EDFigure10-FeedingStates/*.py` | `Ballpushing_Exploration/Datasets/.../summary/pooled_summary.feather` | Exploration |

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

## License & citation

Source code: MIT. © 2024–2026 Neuroengineering Laboratory @EPFL — Ramdya
Lab. See [`LICENSE`](LICENSE).

Please cite the paper above when using the library, the metrics, or the
dataset in your work.
