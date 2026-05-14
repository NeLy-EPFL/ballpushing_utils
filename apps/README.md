# Interactive dashboards

These Panel/HoloViews apps let you explore the paper datasets interactively —
select genotypes, metrics, and plot modes without writing any code.

## Prerequisites

Install the `[interactive]` extras (Panel, HoloViews, Bokeh) into your
environment before running any app here:

```bash
pip install -e ".[interactive]"
```

## Available apps

### `screen_explorer.py` — TNT silencing-screen metrics

Browse the ~225-genotype silencing screen: filter by brain region or genotype,
then compare metrics as boxplots with superimposed strip plots, or explore
pairwise correlations in scatter mode.

**Data requirement**: the screen summary feather must be present. If it is not,
the app opens a screen with download instructions. You can also pre-fetch it:

```bash
ballpushing-fetch --archive screen
```

**Run**:

```bash
panel serve apps/screen_explorer.py --show
```

Add `--autoreload` during development to pick up edits without restarting.

## Tips

- **Brain region → Genotype cascade**: selecting one or more brain regions
  narrows the Genotype dropdown to only the genotypes from those regions.
  Leave both empty to see all genotypes.
- **Controls toggle**: check/uncheck "Include controls" to add or hide
  Empty-Gal4, Empty-Split, and TNTxPR.
- **Boxplot columns**: the "Columns" slider adjusts how many metric panels
  fit side-by-side.
- **Scatter colour by Genotype**: with many genotypes selected, colours cycle
  through Category20 (20-colour Bokeh palette).
