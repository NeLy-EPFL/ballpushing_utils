from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml


CONTROL_BY_SPLIT = {"y": "Empty-Split", "n": "Empty-Gal4", "m": "TNTxPR"}


@dataclass
class NicknameControlInfo:
    nickname: str
    genotype: str
    split: str
    simplified_nickname: str
    brain_region: str
    control_nickname: str
    control_genotype: str


def load_experiment_directories_from_yaml(yaml_path: Path) -> list[Path]:
    with open(yaml_path, "r") as f:
        payload = yaml.safe_load(f) or {}
    directories = payload.get("directories", [])
    return [Path(p) for p in directories]


def _normalize_str(value) -> str:
    return str(value).strip()


def resolve_nickname_control(
    nickname: str,
    split_registry_path: Path,
    control_nickname: Optional[str] = None,
    force_control: Optional[str] = None,
) -> NicknameControlInfo:
    if not split_registry_path.exists():
        raise FileNotFoundError(f"Split registry not found: {split_registry_path}")

    reg = pd.read_csv(split_registry_path)
    required = ["Nickname", "Genotype", "Split"]
    missing = [c for c in required if c not in reg.columns]
    if missing:
        raise ValueError(f"Split registry missing columns: {missing}")

    reg["Nickname"] = reg["Nickname"].astype(str).str.strip()
    if "Simplified Nickname" in reg.columns:
        reg["Simplified Nickname"] = reg["Simplified Nickname"].astype(str).str.strip()
    reg["Genotype"] = reg["Genotype"].astype(str).str.strip()
    reg["Split"] = reg["Split"].astype(str).str.strip()

    nickname_in = _normalize_str(nickname)
    row = reg.loc[reg["Nickname"] == nickname_in]
    if row.empty and "Simplified Nickname" in reg.columns:
        row = reg.loc[reg["Simplified Nickname"] == nickname_in]
    if row.empty:
        raise ValueError(f"Nickname '{nickname}' not found in split registry: {split_registry_path}")

    canonical_nickname = row.iloc[0]["Nickname"]
    genotype = row.iloc[0]["Genotype"]
    split = row.iloc[0]["Split"]
    simplified_nickname = (
        row.iloc[0]["Simplified Nickname"] if "Simplified Nickname" in reg.columns else canonical_nickname
    )
    brain_region = row.iloc[0]["Simplified region"] if "Simplified region" in reg.columns else "Unknown"

    if control_nickname is None:
        if force_control is not None and split != "m":
            control_nickname = force_control
        else:
            control_nickname = CONTROL_BY_SPLIT.get(split)

    if control_nickname is None:
        raise ValueError(f"Could not resolve control nickname for split '{split}'")

    control_in = _normalize_str(control_nickname)
    control_row = reg.loc[reg["Nickname"] == control_in]
    if control_row.empty and "Simplified Nickname" in reg.columns:
        control_row = reg.loc[reg["Simplified Nickname"] == control_in]
    if control_row.empty:
        raise ValueError(f"Control nickname '{control_nickname}' not found in split registry")

    canonical_control_nickname = control_row.iloc[0]["Nickname"]
    control_genotype = control_row.iloc[0]["Genotype"]

    return NicknameControlInfo(
        nickname=canonical_nickname,
        genotype=genotype,
        split=split,
        simplified_nickname=str(simplified_nickname).strip(),
        brain_region=str(brain_region).strip(),
        control_nickname=canonical_control_nickname,
        control_genotype=control_genotype,
    )


def _load_metadata_variable_by_arena(experiment_dir: Path, variable_name: str) -> dict[str, str]:
    metadata_path = experiment_dir / "Metadata.json"
    if not metadata_path.exists():
        return {}

    with open(metadata_path, "r") as f:
        payload = json.load(f)

    variables = payload.get("Variable", [])
    if not variables:
        return {}

    var_lut = {_normalize_str(v).lower(): i for i, v in enumerate(variables)}
    var_idx = var_lut.get(_normalize_str(variable_name).lower())
    if var_idx is None:
        return {}

    out = {}
    for arena_i in range(1, 10):
        arena_key = f"Arena{arena_i}"
        values = payload.get(arena_key, [])
        if var_idx < len(values):
            out[arena_key.lower()] = _normalize_str(values[var_idx])
    return out


def _list_corridors_with_video(arena_dir: Path) -> list[Path]:
    if not arena_dir.exists() or not arena_dir.is_dir():
        return []

    fly_dirs = []
    for child in sorted(arena_dir.iterdir()):
        if not child.is_dir():
            continue
        if any(child.glob("*.mp4")):
            fly_dirs.append(child)
    return fly_dirs


def collect_fly_manifest_by_metadata(
    yaml_path: Path,
    metadata_key: str,
    metadata_values: list[str],
    max_experiments: Optional[int] = None,
    max_flies_per_value: Optional[int] = None,
) -> pd.DataFrame:
    experiments = load_experiment_directories_from_yaml(yaml_path)
    if max_experiments is not None:
        experiments = experiments[:max_experiments]

    wanted = {_normalize_str(v) for v in metadata_values}
    rows = []

    for exp_dir in experiments:
        if not exp_dir.exists():
            continue

        arena_value = _load_metadata_variable_by_arena(exp_dir, metadata_key)
        if not arena_value:
            continue

        for arena, value in arena_value.items():
            if value not in wanted:
                continue
            arena_dir = exp_dir / arena
            for fly_dir in _list_corridors_with_video(arena_dir):
                rows.append(
                    {
                        "experiment_dir": exp_dir.as_posix(),
                        "arena": arena,
                        "fly_dir": fly_dir.as_posix(),
                        "metadata_key": metadata_key,
                        "metadata_value": value,
                    }
                )

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        return manifest

    if max_flies_per_value is not None:
        manifest = (
            manifest.sort_values(["metadata_value", "experiment_dir", "arena", "fly_dir"])
            .groupby("metadata_value", as_index=False)
            .head(max_flies_per_value)
            .reset_index(drop=True)
        )

    return manifest


def load_flies_from_manifest(manifest: pd.DataFrame) -> list:
    from Ballpushing_utils import Fly

    flies = []
    for fly_dir in manifest["fly_dir"].tolist():
        try:
            fly = Fly(Path(fly_dir), as_individual=True)
            if fly.tracking_data is None or not fly.tracking_data.valid_data:
                continue
            flies.append(fly)
        except Exception as exc:
            print(f"Skipping fly {fly_dir}: {exc}")
            continue
    return flies


def build_dataset_from_yaml_by_metadata(
    yaml_path: Path,
    dataset_type: str,
    metadata_key: str,
    metadata_values: list[str],
    annotate_events: bool = True,
    annotate_contacts: bool = False,
    max_experiments: Optional[int] = None,
    max_flies_per_value: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from Ballpushing_utils import Dataset

    manifest = collect_fly_manifest_by_metadata(
        yaml_path=yaml_path,
        metadata_key=metadata_key,
        metadata_values=metadata_values,
        max_experiments=max_experiments,
        max_flies_per_value=max_flies_per_value,
    )
    if manifest.empty:
        return pd.DataFrame(), manifest

    flies = load_flies_from_manifest(manifest)
    if not flies:
        return pd.DataFrame(), manifest

    for fly in flies:
        if hasattr(fly, "config"):
            fly.config.annotate_contacts = bool(annotate_contacts)

    dataset_obj = Dataset(flies, dataset_type=dataset_type).data
    dataset = dataset_obj if isinstance(dataset_obj, pd.DataFrame) else pd.DataFrame()

    if isinstance(dataset, pd.DataFrame) and not dataset.empty and not annotate_events:
        for col in ["interaction_event", "interaction_event_onset"]:
            if col in dataset.columns:
                dataset = dataset.drop(columns=[col])
    return dataset, manifest


def build_dataset_for_nickname_and_control(
    yaml_path: Path,
    dataset_type: str,
    nickname: str,
    split_registry_path: Path,
    control_nickname: Optional[str] = None,
    force_control: Optional[str] = None,
    annotate_events: bool = True,
    annotate_contacts: bool = False,
    max_experiments: Optional[int] = None,
    max_flies_per_group: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, NicknameControlInfo]:
    info = resolve_nickname_control(
        nickname=nickname,
        split_registry_path=split_registry_path,
        control_nickname=control_nickname,
        force_control=force_control,
    )

    dataset, manifest = build_dataset_from_yaml_by_metadata(
        yaml_path=yaml_path,
        dataset_type=dataset_type,
        metadata_key="Genotype",
        metadata_values=[info.genotype, info.control_genotype],
        annotate_events=annotate_events,
        annotate_contacts=annotate_contacts,
        max_experiments=max_experiments,
        max_flies_per_value=max_flies_per_group,
    )

    if isinstance(dataset, pd.DataFrame) and not dataset.empty and "Nickname" in dataset.columns:
        dataset = dataset[dataset["Nickname"].astype(str).isin([info.nickname, info.control_nickname])].copy()
    elif not isinstance(dataset, pd.DataFrame):
        dataset = pd.DataFrame()

    return dataset, manifest, info
