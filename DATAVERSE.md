# Using `ballpushing_utils` with the Dataverse archives

Companion guide to the three Harvard Dataverse datasets that accompany
Durrieu et al. (2026). The repo's main [`README.md`](README.md) covers
installation and the high-level "data sources" hierarchy; this file is
the precise per-archive reference: what's in each archive, how its
folders are named, and exactly which CLI invocation rebuilds a
feather from raw HDF5 tracks.

If you only want to *reproduce paper figures*, you don't need to
understand any of the naming conventions — extract any archive (or
subset of it) under `$BALLPUSHING_DATA_ROOT` and the figure scripts
pick the bundled feathers up automatically. This document is for the
case where you want to **rerun the metric pipeline** on the raw tracks.

---

## Archive layout — at a glance

Each Dataverse dataset is a flat collection of `.tar` archives plus a
small set of pre-computed feathers and a `config.json`. Each `.tar`
contains the SLEAP HDF5 tracks for one *condition* of one *paradigm*.
When extracted, the archive becomes a directory whose name encodes
the paradigm + condition; inside, flies are sorted by acquisition
date and corridor:

```
<archive-name>/
└── <YYMMDD>[-N]/                 # acquisition date; -2/-3 for same-day duplicates
    └── arenaN/
        └── corridorM/             # or "Left" / "Right" for F1
            ├── *_ball*.h5
            ├── *_fly*.h5
            └── *_full_body*.h5    # absent for F1 — no skeleton model yet
```

`<archive-name>` is **the only metadata** that travels with the SLEAP
tracks. Everything the package needs (`Genotype`, `Magnet`, `BallType`,
`F1_condition`, etc.) is decoded from this name by
`ballpushing_utils.dataverse.parse_archive_name`.

---

## The three Dataverse datasets

### 1. Silencing screen (TNT)

Each archive is a single genotype:

```
LC6.tar               → LC6/<date>/<arena>/<corridor>/{ball,fly,full_body}.h5
MB247xTNT.tar         → MB247xTNT/...
LC10-2.tar            → LC10-2/...
…
```

The archive name **is** the value of the `Genotype` column. The
`parse_archive_name("LC6")` call returns `("TNT", {"Genotype": "LC6"})`.

**Pre-built feathers** (top-level, alongside the tars):

| File | Used by |
|---|---|
| `ballpushing_metrics_silencing_screen.feather` | summary / per-genotype heatmap (Fig. 3, ED Fig. 6) |
| `<region>_trajectories.feather` (and `<region>_trajectories-2.feather` for big regions split in two) | regional trajectory plots — concatenate the `-2` halves before reading |
| `Config.json` | the `ballpushing_utils.Config` snapshot used to build the feathers |

**Reproducing figures** — extract whichever feathers you need under
`$BALLPUSHING_DATA_ROOT/Ballpushing_TNTScreen/Datasets/...` (or
anywhere if you set `BALLPUSHING_FEATHER_SEARCH`); `python
run_all_figures.py` picks them up.

**Rerunning the pipeline** for one or more genotypes:

```bash
# Extract one or more genotypes side by side under a scratch root
mkdir -p ~/dv_screen
tar xf LC6.tar       -C ~/dv_screen
tar xf MB247xTNT.tar -C ~/dv_screen

python src/dataset_builder.py \
    --dataverse-root ~/dv_screen \
    --datasets summary
# experiment_type is auto-detected as TNT; --experiment-type only needed
# to override.
```

### 2. Affordance (Generalisation + MagnetBlock)

Two paradigms share this dataset.

#### MagnetBlock

Two archives only:

```
MagnetBlock-Blocked.tar    → Magnet=y in the feather
MagnetBlock-Control.tar    → Magnet=n in the feather
```

(The parser is case-insensitive on this prefix, so `Magnetblock-…`
also works if you find an older draft.)

**Pre-built feathers**:

| File | Used by |
|---|---|
| `Magnetblock_ballpushing_metrics.feather` | Fig. 2 first-major-push panels |
| `Magnetblock_trajectories.feather` | Fig. 2 trajectories panel |
| `Magnetblock_config.json` | metric thresholds used to build them |

Rerun:

```bash
mkdir -p ~/dv_magnetblock
tar xf MagnetBlock-Blocked.tar -C ~/dv_magnetblock
tar xf MagnetBlock-Control.tar -C ~/dv_magnetblock

python src/dataset_builder.py \
    --dataverse-root ~/dv_magnetblock \
    --datasets summary coordinates
# experiment_type auto-detected as MagnetBlock from "Magnetblock-*"
# folder names; condition mapped to {Magnet: "y"|"n"} per archive.
```

#### F1 (published as "Generalisation")

> The F1 paradigm is referred to as "Generalisation" on Dataverse to
> reflect the question it asks. Internally the package keeps the
> `experiment_type="F1"` name and the `F1_condition` column — the
> alias is translated at the Dataverse boundary only.

Archive naming: `Generalisation-<genotype>-<f1_condition>` where:

- `<genotype>` is the genotype as it appears in `arena_metadata`:
  `Wild-type`, `MB247xTNT`, `TNTxLC10-2`, `TNTxLC6`, …. Note:
  the wild-type strain is published as `Wild-type` here, not the
  rig codename `PR`.
- `<f1_condition>` is one of `control`, `pretrained`,
  `pretrained_unlocked`. The publisher applied the on-rig Left/Right
  + Unlocked logic before packaging, so each fly is in the right
  condition folder by construction.

All published F1 archives (one per `<genotype, condition>` pair):

```
F1_wildtype/raw_h5/
├── Generalisation-Wild-type-control.tar
├── Generalisation-Wild-type-pretrained.tar
└── Generalisation-Wild-type-pretrained_unlocked.tar

F1_tnt/raw_h5/                    # 3 conditions × 8 silencing genotypes = 24 tars
├── Generalisation-TNTxDDC-{control,pretrained,pretrained_unlocked}.tar
├── Generalisation-TNTxEmptyGal4-{...}.tar
├── Generalisation-TNTxEmptySplit-{...}.tar
├── Generalisation-TNTxLC10-2-{...}.tar       # genotype contains a hyphen — parsed from the right
├── Generalisation-TNTxLC16-1-{...}.tar
├── Generalisation-TNTxMB247-{...}.tar
├── Generalisation-TNTxTH-{...}.tar
└── Generalisation-TNTxTRH-{...}.tar
```

`parse_archive_name("Generalisation-TNTxLC10-2-pretrained")` returns
`("F1", {"Pretraining": "y", "F1_condition": "pretrained", "Genotype":
"TNTxLC10-2"})` — three columns populated from one folder name.
Genotypes containing hyphens (`Wild-type`, `TNTxLC10-2`) are split
from the right so the last hyphen-segment is always the F1 condition.

**Pre-built feathers**:

| File | Used by |
|---|---|
| `Generalisation_ballpushing_metrics.feather` | Fig. 2 F1 control panel |
| `Generalisation_fly_positions.feather` | Fig. 2 F1 heatmaps panel |
| `Generalisation-TNT_ballpushing_metrics.feather` | Fig. 3 F1-TNT panel |
| `Generalisation_config.json` | metric thresholds |

Rerun for any subset of conditions:

```bash
mkdir -p ~/dv_f1
tar xf Generalisation-Wild-type-control.tar    -C ~/dv_f1
tar xf Generalisation-Wild-type-pretrained.tar -C ~/dv_f1
tar xf Generalisation-Wild-type-pretrained_unlocked.tar -C ~/dv_f1

python src/dataset_builder.py \
    --dataverse-root ~/dv_f1 \
    --datasets summary fly_positions
```

### 3. Supplementary experiments

Each archive packs one paradigm and (one or more) condition value(s).
The published Dataverse tree organises them into per-paradigm
subdirectories so the file listing on the Dataverse page stays
navigable:

```
ball_scents/raw_h5/
├── Ballscents-New.tar
├── Ballscents-Pre-exposed.tar
├── Ballscents-Washed.tar
├── Ballscents-New-plus-Pre-exposed.tar           # combined-source ball
└── Ballscents-Washed-plus-Pre-exposed.tar

ball_types/raw_h5/
├── Balltype-Manufactured.tar
├── Balltype-Rusty.tar
├── Balltype-Sandpaper.tar
├── Balltype-Silicone.tar
└── Balltype-Steel.tar

feedingstate_wildtype/raw_h5/
├── Wild-type_Lights-off_Fed-1.tar
├── Wild-type_Lights-off_Fed-2.tar
├── Wild-type_Lights-off_Starved.tar
├── Wild-type_Lights-off_Starved-without-water-1.tar
├── Wild-type_Lights-off_Starved-without-water-2.tar
├── Wild-type_Lights-on_Fed-1.tar
├── Wild-type_Lights-on_Fed-2.tar
├── Wild-type_Lights-on_Starved-1.tar
├── Wild-type_Lights-on_Starved-2.tar
├── Wild-type_Lights-on_Starved-without-water-1.tar
└── Wild-type_Lights-on_Starved-without-water-2.tar

TNT_olfaction_dark/raw_h5/
├── TNTxEmptyGal4-Light-off.tar
├── TNTxEmptyGal4-Light-on.tar
├── TNTxIR8a-Light-off.tar
└── TNTxIR8a-Light-on.tar

trial_duration/raw_h5/
├── Multi-trials-1.tar
├── Multi-trials-2.tar
└── Multi-trials-3.tar
```

The naming patterns and their effect on the rebuilt feather:

| Archive name pattern | `experiment_type` | Columns populated |
|---|---|---|
| `Ballscents-<value>` | `TNT` | `BallScent` |
| `Balltype-<value>` (singular!) | `TNT` | `BallType` |
| `Wild-type_Lights-{on,off}_<FeedingState>[-<replicate>]` | `TNT` | `Genotype` (= `Wild-type`) + `Light` + `FeedingState` (replicate suffix stripped) |
| `<Genotype>-Light-{on,off}` | `TNT` | `Genotype` + `Light` |
| `Multi-trials-<N>` | `Learning` | `Trial_duration` |

Pre-built feathers follow the convention
`<Condition>_ballpushing_metrics.feather` and
`<Condition>_trajectories.feather`. A `Config.json` per paradigm
captures the build state.

Rerun for, say, all ball types:

```bash
mkdir -p ~/dv_balltypes
for f in Balltype-*.tar; do tar xf "$f" -C ~/dv_balltypes; done

python src/dataset_builder.py \
    --dataverse-root ~/dv_balltypes \
    --datasets summary
```

Or for the feeding-state archives (the parser handles all 11 archives
in one go — Genotype + Light + FeedingState all populate from the
folder name):

```bash
mkdir -p ~/dv_feeding
for f in Wild-type_Lights-*.tar; do tar xf "$f" -C ~/dv_feeding; done
python src/dataset_builder.py --dataverse-root ~/dv_feeding --datasets summary
```

---

## How auto-detection works

When you pass `--dataverse-root <root>`, `dataset_builder.py`:

1. Walks `<root>/<archive>/<date>/arena*/<corridor>/` looking for
   any directory containing `*ball*.h5`. Junk top-level dirs (a
   `Config.json`, a stray `README`, half-extracted folders…) are
   ignored.
2. For each unique archive folder name found, calls
   `dataverse.parse_archive_name(name)` to derive
   `(experiment_type, fields)`.
3. Logs a "Archive → recipe map" table so you can sanity-check the
   parsing before the long-running build kicks off:

   ```
   Archive → recipe map:
     'Magnetblock-Blocked'        → ('MagnetBlock', {'Magnet': 'y'})
     'Magnetblock-Control'        → ('MagnetBlock', {'Magnet': 'n'})
     'Generalisation-MB247xTNT-pretrained' → ('F1', {'Pretraining': 'y', 'F1_condition': 'pretrained', 'Genotype': 'MB247xTNT'})
   ```

4. Builds each fly's synthetic `arena_metadata` from the matching
   recipe and runs the metric pipeline.

Two CLI flags act as overrides:

- `--experiment-type {TNT|MagnetBlock|F1|Learning}` — forces the
  paradigm for every archive. Useful only if your archive folder
  names don't follow the convention; you'll get a warning per
  archive whose parsed type differs.
- `--condition-field <column>` — for paradigms without a multi-column
  transformer, override the column the archive value lands in. With
  archives that already carry multi-column meaning (F1,
  TNT-olfaction-dark) this is silently ignored.

If your archive names don't fit any pattern, edit
`src/ballpushing_utils/dataverse.py:ARCHIVE_PREFIX_RECIPES` (for
single-column paradigms) or add a special case to `parse_archive_name`.

---

## Mapping cheat-sheet

| Archive folder name | `(experiment_type, fields)` |
|---|---|
| `MagnetBlock-Blocked` | `("MagnetBlock", {"Magnet": "y"})` |
| `MagnetBlock-Control` | `("MagnetBlock", {"Magnet": "n"})` |
| `Generalisation-Wild-type-control` | `("F1", {"Pretraining": "n", "F1_condition": "control", "Genotype": "Wild-type"})` |
| `Generalisation-Wild-type-pretrained` | `("F1", {"Pretraining": "y", "F1_condition": "pretrained", "Genotype": "Wild-type"})` |
| `Generalisation-Wild-type-pretrained_unlocked` | `("F1", {"Pretraining": "y", "F1_condition": "pretrained_unlocked", "Genotype": "Wild-type"})` |
| `Generalisation-TNTxLC10-2-pretrained` | `("F1", {"Pretraining": "y", "F1_condition": "pretrained", "Genotype": "TNTxLC10-2"})` |
| `Generalisation-TNTxMB247-pretrained_unlocked` | `("F1", {"Pretraining": "y", "F1_condition": "pretrained_unlocked", "Genotype": "TNTxMB247"})` |
| `Ballscents-New` | `("TNT", {"BallScent": "New"})` |
| `Ballscents-Washed-plus-Pre-exposed` | `("TNT", {"BallScent": "Washed-plus-Pre-exposed"})` |
| `Balltype-Rusty` | `("TNT", {"BallType": "Rusty"})` |
| `Wild-type_Lights-off_Fed-1` | `("TNT", {"Genotype": "Wild-type", "Light": "off", "FeedingState": "Fed"})` |
| `Wild-type_Lights-on_Starved` | `("TNT", {"Genotype": "Wild-type", "Light": "on", "FeedingState": "Starved"})` |
| `Wild-type_Lights-on_Starved-without-water-2` | `("TNT", {"Genotype": "Wild-type", "Light": "on", "FeedingState": "Starved-without-water"})` |
| `TNTxEmptyGal4-Light-off` | `("TNT", {"Genotype": "TNTxEmptyGal4", "Light": "off"})` |
| `TNTxIR8a-Light-on` | `("TNT", {"Genotype": "TNTxIR8a", "Light": "on"})` |
| `Multi-trials-1` | `("Learning", {"Trial_duration": "1"})` |
| `LC6` | `("TNT", {"Genotype": "LC6"})` |
| `Wild-type_PR` | `("TNT", {"Genotype": "Wild-type_PR"})` |
| `MBON-gamma1pedc_to_alpha_beta` | `("TNT", {"Genotype": "MBON-gamma1pedc_to_alpha_beta"})` |
| `<any other single-token>` | `("TNT", {"Genotype": "<name>"})` |

Notes on the trickier patterns:

- **MagnetBlock** — exact match, case-insensitive (`MagnetBlock` /
  `magnetblock` both work).
- **Generalisation-** — splits the condition (last hyphen-segment)
  from the genotype (the rest), so genotypes containing hyphens
  (`Wild-type`, `TNTxLC10-2`, `TNTxLC16-1`) parse correctly.
- **Wild-type_Lights-…_…** — requires three or more underscore-
  separated tokens with the second starting with `Lights-`. This
  distinguishes feeding-state archives from the silencing-screen
  `Wild-type_PR` / `Wild-type_CS` archives (only two underscore
  tokens, fall through to the default `Genotype` branch).
  Trailing `-1` / `-2` replicate suffixes are stripped so two
  replicates of `Starved-without-water` collapse onto a single
  feather column value.
- **`<Genotype>-Light-{on,off}`** — anything ending in `-Light-on`
  or `-Light-off` is treated as olfaction-dark with the prefix as
  the genotype. The silencing-screen archives don't end this way,
  so there's no collision.
- **Default branch** — anything else is treated as a single-token
  silencing-screen genotype name (with hyphens and underscores
  preserved).

To inspect the recipe for a name without launching a build:

```python
from ballpushing_utils.dataverse import parse_archive_name
parse_archive_name("Generalisation-TNTxLC10-2-pretrained")
# ('F1', {'Pretraining': 'y', 'F1_condition': 'pretrained', 'Genotype': 'TNTxLC10-2'})
```

---

## Caveats

- **No `Metadata.json`, no `.mp4`, no `fps.npy`.** The Dataverse
  archives ship SLEAP HDF5 tracks only. The package handles this:
  per-arena metadata is synthesised from the archive name (this
  document); fps defaults to 29 (the canonical rig rate);
  ball-coordinate normalisation falls back to a thorax-anchored
  affine fit so no original-video size is needed (see
  `SkeletonMetrics._estimate_raw_to_template_transform`).
- **F1 has no skeleton tracks** — the package skips
  `SkeletonMetrics` entirely on F1 paradigm flies, so the affine fit
  is irrelevant there. F1 archives ship `*_ball.h5` + `*_fly.h5`
  only.
- **Cross-annotation panels**. Some ED panels filter on two columns
  at once (e.g. `Genotype × Light`, `Genotype × FeedingState`). The
  per-archive recipes only populate the columns relevant to *that
  archive*'s paradigm, so a panel that needs `Genotype × Light` from
  the Screen archive won't reproduce from a single-archive rerun —
  fall back to the bundled pooled feather, which was built from the
  on-rig data with full `Metadata.json`.
- **Region map / F1 template**. The brain-region registry CSV and
  the F1 heatmap background are bundled in the package itself
  (`src/ballpushing_utils/assets/`) — no Dataverse download required
  for figures that need them.
- **Trial durations / `Trial_duration` paradigm**. This is a Learning
  archive (different `experiment_type`) — make sure
  `EXPERIMENT_TYPE_FROM_PATH` recognises it, or pass
  `--experiment-type Learning` if the auto-detect picks the wrong
  one.

---

## Editable config: `Config.json` shipped with the feathers

Each archive ships the `ballpushing_utils.Config` snapshot used to
build the published feathers. To audit or replicate the exact
thresholds:

```python
import json
with open("Magnetblock_config.json") as f:
    cfg = json.load(f)
print(cfg["ballpushing_config"]["interaction_threshold"])
print(cfg["ballpushing_config"]["major_event_threshold"])
print(cfg["ballpushing_config"]["final_event_threshold"])
```

To rebuild against the exact same thresholds:

```python
from ballpushing_utils import Config, Fly
cfg = Config(**cfg["ballpushing_config"])
fly = Fly(fly_dir, custom_config=cfg, dataverse_condition={...})
```

If your local rebuild diverges from the published feathers, the
first thing to compare is this `Config` block.
