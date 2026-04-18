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
> **Dataset:** *TODO — paste Harvard Dataverse DOI / URL once published.*
> The dataverse hosts both the **raw HDF5 SLEAP tracks** and the **pooled
> per-fly summary feathers** used by the figure scripts. You can reproduce
> the paper figures from the feathers alone (no SLEAP re-processing
> required).

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
├── tests/                      # pytest suite.
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
pip install -e ".[all]"           # everything
```

`ballpushing_utils` depends on
[`utils_behavior`](https://github.com/labramdya/utils_behavior), the
lab's general-purpose behavioural-analysis utilities. It is declared as a
PyPI dependency in `pyproject.toml`; if your environment cannot resolve
it from PyPI, install it from source first.

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
BALLPUSHING_DATA_ROOT=/path/to/dataverse/download
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

The dataverse archive mirrors the layout the scripts expect, so after
`unzip`ing the bundle into `BALLPUSHING_DATA_ROOT` you can run any figure
script unmodified.

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

---

## Figure ↔ script mapping

| Paper figure                   | Script(s)                                                                                                                                                                  | Reads                                                  |
|--------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------|
| **Fig. 1** — setup & wild-type baseline | `figures/Fig1-setup/plot_wildtype_trajectories.py`<br>`figures/Fig1-setup/plot_simulation_trajectories.py`<br>`figures/Fig1-setup/learning_trials_duration.py`<br>`figures/Fig1-setup/compute_distribution_stats.py` | wild-type trajectory + summary feathers                |
| **Fig. 2** — affordance (MagnetBlock + F1) | `figures/Fig2-Affordance/fig2_magnetblock_first_major_push_time.py`<br>`figures/Fig2-Affordance/fig2_magnetblock_first_major_push_index.py`<br>`figures/Fig2-Affordance/plot_magnetblock_trajectories.py`<br>`figures/Fig2-Affordance/fig2_f1_control_conditions.py`<br>`figures/Fig2-Affordance/fig2_f1_heatmaps_pretraining.py` | MagnetBlock pooled summary; F1 pre-training datasets   |
| **Fig. 3** — neural silencing screen | `figures/Fig3-Screen/fig3_screen_heatmap.py`<br>`figures/Fig3-Screen/fig3_f1_tnt.py`                                                                                       | TNT screen + F1-TNT pooled summaries                   |
| **ED Fig. 6** — behavioural dendrogram | `figures/EDFigure6-Dendrogram/edfigure6_dendrogram.py`                                                                                                                     | wild-type metric matrix                                |

Supplementary panels (feeding-state, ball scents, ball types, dark
olfaction, learning mutants, broad TNT screen, etc.) live under
`plots/Supplementary_exps/` and `plots/Ballpushing_PR/`. They follow the
same `script.py → PDF + stats.csv` convention as the figure scripts.

---

## Building feathers from raw H5 SLEAP tracks

The dataverse exposes the per-fly summary feathers used by every figure
script, so most users will never need this step. If you do want to
re-process raw tracks, the pipeline is:

1. **Drop SLEAP `.h5` exports** under `$BALLPUSHING_DATA_ROOT/<experiment>/`.
2. **Describe the experiment batch** in a YAML file under
   `experiments_yaml/` (genotypes, replicate dates, conditions). See
   any of the existing files for the schema.
3. **Build the dataset:** `python src/dataset_builder.py <yaml>` produces
   per-fly metric tables.
4. **Pool feathers:** `python src/pool_feather_files.py` concatenates
   per-experiment feathers into the `pooled_summary.feather` files the
   figure scripts read.
5. Run the figure scripts as usual.

---

## Tests

```bash
pip install -e ".[dev]"
pytest                    # unit tests (fast, no SLEAP data needed)
pytest tests/integration  # integration tests (require sample data)
```

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
