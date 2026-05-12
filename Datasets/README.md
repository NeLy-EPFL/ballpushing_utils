# Datasets/

This folder is the default destination for feathers downloaded from the
Harvard Dataverse via:

```bash
ballpushing-fetch              # everything the paper figures need
ballpushing-fetch --help       # all options
```

The contents are git-ignored. To use a different location, set
`BALLPUSHING_DATA_ROOT` to point at it; the figure scripts and the
fetcher both honour the same variable.

After the fetch completes, `python run_all_figures.py` reproduces every
paper panel without any further configuration.

Don't commit feather files here — the published versions live on
Dataverse (see DOIs in the top-level README).
