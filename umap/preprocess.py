import numpy as np
import pandas as pd
from utils import c2xy
from pathlib import Path
from tqdm import tqdm

idx_col = ["fly", "event_id", "frame"]
data_cols = ['centre_preprocessed', 'Thorax', 'Head', 'Abdomen', 'Lfront', 'Lmid', 'Lhind', 'Rfront', 'Rmid', 'Rhind']
new_data_cols = ["b", "t", "h", "a", "lf", "lm", "lh", "rf", "rm", "rh"]
data_xy_cols = [f"{c}_{col}" for col in data_cols for c in "xy"]
root_dir = Path('/mnt/upramdya/data/MD/Ballpushing_TNTScreen/Datasets/')
dataset_name = "250809_02_standardized_contacts_TNT_screen_Data"
data_dir = root_dir / dataset_name / "standardized_contacts"
data_paths = sorted(data_dir.glob("2*.feather"))
save_dir = Path("cached")
save_dir.mkdir(exist_ok=True, parents=True)

for data_path in tqdm(data_paths):
    save_name = data_path.stem.removesuffix('_Videos_Tracked_standardized_contacts')

    df = pd.read_feather(data_path)
    df = df[df["event_type"].eq("contact")]
    assert len(df) ==  len(df[idx_col].drop_duplicates())
    df.set_index(idx_col, inplace=True)
    df.sort_index(inplace=True)
    df_flies = df.reset_index()[["fly", "Genotype", "flypath"]] \
        .drop_duplicates() \
        .rename({"Genotype": "line", "flypath": "path"}, axis=1) \
        .reset_index(drop=True)
    df = pd.DataFrame(
        df[data_xy_cols].values.reshape((len(df), -1, 2)) @ (1, 1j),
        columns=new_data_cols,
        index=df.index,
    )
    to_drop = []
    for key, df_ in df.groupby(["fly", "event_id"]):
        if (~np.isfinite(df_.values)).all():
            to_drop.append(key)
    size = df.groupby(["fly", "event_id"]).size()
    df.drop(size[size.ne(120)].index, inplace=True)
    for key, df_ in df.groupby(level=["fly", "event_id"]):
        df.loc[key] = df_.interpolate(method="linear", limit_direction="both", limit=30, axis=0, limit_area="inside")
    df.loc[df["lf"].isna(), "lf"] = df.loc[df["lf"].isna(), "rf"].values
    df.loc[df["rf"].isna(), "rf"] = df.loc[df["rf"].isna(), "lf"].values
    df.loc[df["lm"].isna(), "lm"] = df.loc[df["lm"].isna(), "rm"].values
    df.loc[df["rm"].isna(), "rm"] = df.loc[df["rm"].isna(), "lm"].values
    df.loc[df["lh"].isna(), "lh"] = df.loc[df["lh"].isna(), "rh"].values
    df.loc[df["rh"].isna(), "rh"] = df.loc[df["rh"].isna(), "lh"].values
    for key, df_ in df.groupby(level=["fly", "event_id"]):
        df.loc[key] = df_.interpolate(method="linear", limit=10, axis=0)
    df.dropna(inplace=True, axis=0)
    size = df.groupby(["fly", "event_id"]).size()
    df.drop(size[size.ne(120)].index, inplace=True)
    c2xy(df).to_feather(save_dir / f"{save_name}_contacts.feather")
    df_flies.to_feather(save_dir / f"{save_name}_flies.feather")
