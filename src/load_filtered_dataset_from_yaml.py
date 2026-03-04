#!/usr/bin/env python3
"""
Build a dataset from selected flies by filtering experiment metadata across a YAML list of experiments.

Examples
--------
By metadata:
python load_filtered_dataset_from_yaml.py \
  --yaml experiments_yaml/TNT_screen.yaml \
  --dataset-type summary \
  --metadata-key Genotype \
  --metadata-value MB247 \
  --metadata-value Empty-Split

By nickname + matched control:
python load_filtered_dataset_from_yaml.py \
  --yaml experiments_yaml/TNT_screen.yaml \
  --dataset-type coordinates \
  --nickname MB247

Fast filtering from an existing pooled dataset (no fly reload):
python load_filtered_dataset_from_yaml.py \
    --yaml experiments_yaml/F1_TNT_Full.yaml \
    --input-dataset /mnt/.../pooled_summary.feather \
    --metadata-key Genotype \
    --metadata-value TNTxMB247 \
    --metadata-value TNTxEmptySplit \
    --output-path /mnt/.../subset.feather
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).parent))

from Ballpushing_utils.filtered_dataset_loader import (
    build_dataset_for_nickname_and_control,
    build_dataset_from_yaml_by_metadata,
    load_experiment_directories_from_yaml,
    resolve_nickname_control,
)


def _load_tabular_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".feather":
        return pd.read_feather(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input dataset format: {path.suffix}. Use .feather, .csv, or .parquet")


def _normalize_datestr(value: object) -> str:
    s = str(value)
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits[:6] if len(digits) >= 6 else ""


def _extract_yaml_date_tokens(yaml_path: Path) -> set[str]:
    date_tokens: set[str] = set()
    for exp_dir in load_experiment_directories_from_yaml(yaml_path):
        name = exp_dir.name
        m = re.match(r"(\d{6})", name)
        if m:
            date_tokens.add(m.group(1))
    return date_tokens


def _restrict_to_yaml_dates_if_possible(dataset: pd.DataFrame, yaml_path: Path) -> tuple[pd.DataFrame, str]:
    date_tokens = _extract_yaml_date_tokens(yaml_path)
    if not date_tokens:
        return dataset, "No date tokens extracted from YAML directories; skipped YAML restriction."

    for date_col in ["Date", "date"]:
        if date_col in dataset.columns:
            date_norm = dataset[date_col].map(_normalize_datestr)
            restricted = dataset.loc[date_norm.isin(date_tokens), :].copy()
            return (
                restricted,
                f"Applied YAML date restriction on '{date_col}' ({len(restricted)}/{len(dataset)} rows kept).",
            )

    return dataset, "No Date/date column found in input dataset; skipped YAML restriction."


def _filter_existing_dataset_by_metadata(
    dataset: pd.DataFrame,
    metadata_key: str,
    metadata_values: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if metadata_key not in dataset.columns:
        raise ValueError(f"Column '{metadata_key}' not found in input dataset")

    wanted = {str(v).strip() for v in metadata_values}
    key_as_str = dataset[metadata_key].astype(str).str.strip()
    subset = dataset.loc[key_as_str.isin(wanted), :].copy()

    manifest = (
        subset[["fly", metadata_key]].drop_duplicates().rename(columns={metadata_key: "metadata_value"})
        if "fly" in subset.columns
        else pd.DataFrame({"metadata_value": sorted(subset[metadata_key].astype(str).unique())})
    )
    return subset, manifest


def _filter_existing_dataset_by_nickname(
    dataset: pd.DataFrame,
    nickname: str,
    split_registry: Path,
    control_nickname: str | None,
    force_control: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    info = resolve_nickname_control(
        nickname=nickname,
        split_registry_path=split_registry,
        control_nickname=control_nickname,
        force_control=force_control,
    )

    subset = pd.DataFrame()
    filter_mode = ""

    if "Nickname" in dataset.columns:
        allowed_nicknames = {str(info.nickname).strip(), str(info.control_nickname).strip()}
        subset = dataset.loc[dataset["Nickname"].astype(str).str.strip().isin(allowed_nicknames), :].copy()
        filter_mode = "Nickname"
    elif "Simplified Nickname" in dataset.columns:
        allowed_simplified = {str(info.simplified_nickname).strip()}
        allowed_control = {str(info.control_nickname).strip()}
        subset = dataset.loc[
            dataset["Simplified Nickname"].astype(str).str.strip().isin(allowed_simplified.union(allowed_control)),
            :,
        ].copy()
        filter_mode = "Simplified Nickname"
    elif "Genotype" in dataset.columns:
        allowed_genotypes = {str(info.genotype).strip(), str(info.control_genotype).strip()}
        subset = dataset.loc[dataset["Genotype"].astype(str).str.strip().isin(allowed_genotypes), :].copy()
        filter_mode = "Genotype"
    else:
        raise ValueError("Input dataset has none of required columns: Nickname, Simplified Nickname, Genotype")

    manifest = (
        subset[["fly", "Genotype"]].drop_duplicates()
        if all(c in subset.columns for c in ["fly", "Genotype"])
        else pd.DataFrame()
    )
    summary = f"Nickname mode: {info.nickname} ({info.genotype}) vs {info.control_nickname} ({info.control_genotype}) [filtered by {filter_mode}]"
    return subset, manifest, summary


def _print_available_filters_from_dataset(dataset: pd.DataFrame, max_values: int = 30) -> None:
    print("No metadata filter provided. Available fields and example values:")
    candidate_cols = [
        "Genotype",
        "Nickname",
        "Simplified Nickname",
        "Pretraining",
        "FeedingState",
        "Period",
        "Orientation",
        "Light",
        "Unlocked",
        "Date",
        "ball_identity",
        "ball_condition",
    ]

    found_any = False
    for col in candidate_cols:
        if col not in dataset.columns:
            continue
        found_any = True
        values = [str(v).strip() for v in dataset[col].dropna().tolist()]
        unique_vals = sorted({v for v in values if v != ""})
        preview = unique_vals[:max_values]
        suffix = "" if len(unique_vals) <= max_values else f" ... (+{len(unique_vals) - max_values} more)"
        print(f"- {col} ({len(unique_vals)} unique): {preview}{suffix}")

    if not found_any:
        non_numeric = [
            c for c in dataset.columns if not pd.api.types.is_numeric_dtype(dataset[c]) and dataset[c].notna().any()
        ]
        if not non_numeric:
            print("No obvious categorical metadata columns found in dataset.")
            return
        print("No standard metadata columns found; available non-numeric columns:")
        for c in non_numeric[:20]:
            unique_vals = sorted(dataset[c].dropna().astype(str).str.strip().unique().tolist())
            preview = unique_vals[:10]
            suffix = "" if len(unique_vals) <= 10 else f" ... (+{len(unique_vals) - 10} more)"
            print(f"- {c}: {preview}{suffix}")


def _print_available_metadata_keys_from_yaml(yaml_path: Path) -> None:
    experiments = load_experiment_directories_from_yaml(yaml_path)
    for exp_dir in experiments:
        meta_path = exp_dir / "Metadata.json"
        if not meta_path.exists():
            continue
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception:
            continue

        vars_found = meta.get("Variable", []) if isinstance(meta, dict) else []
        if vars_found:
            print("No metadata filter provided. Available metadata keys from YAML experiments:")
            print(f"- {vars_found}")
            return

    print("No metadata filter provided and could not infer metadata keys from YAML (missing/invalid Metadata.json).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build dataset from YAML experiments filtered by metadata or nickname+control"
    )
    parser.add_argument("--yaml", type=str, required=True, help="Path to YAML with experiment directories")
    parser.add_argument(
        "--dataset-type",
        type=str,
        required=False,
        default="summary",
        help="Dataset type for Ballpushing_utils.Dataset (e.g., coordinates, summary, event_metrics)",
    )
    parser.add_argument(
        "--input-dataset",
        type=str,
        default=None,
        help=(
            "Optional existing dataset path (.feather/.csv/.parquet). "
            "If provided, filters this table directly instead of rebuilding from raw flies."
        ),
    )
    parser.add_argument(
        "--restrict-to-yaml-dates",
        action="store_true",
        help="When --input-dataset is used, keep only rows matching date prefixes found in YAML experiment directories.",
    )

    parser.add_argument("--nickname", type=str, default=None, help="Nickname to load together with matched control")
    parser.add_argument(
        "--split-registry",
        type=str,
        default="/mnt/upramdya_data/MD/Region_map_250908.csv",
        help="Split registry CSV for nickname/genotype/control mapping",
    )
    parser.add_argument("--control-nickname", type=str, default=None, help="Optional control nickname override")
    parser.add_argument(
        "--force-control",
        type=str,
        default=None,
        choices=["Empty-Split", "Empty-Gal4", "TNTxPR"],
        help="Force control for split y/n lines (split m remains TNTxPR)",
    )

    parser.add_argument("--metadata-key", type=str, default=None, help="Metadata key to filter on (e.g., Genotype)")
    parser.add_argument(
        "--metadata-value",
        action="append",
        default=None,
        help="Metadata value(s) to keep (repeat flag for multiple values)",
    )

    parser.add_argument(
        "--max-experiments", type=int, default=None, help="Limit number of experiments for fast testing"
    )
    parser.add_argument("--max-flies-per-group", type=int, default=None, help="Limit flies per selected metadata value")

    parser.add_argument(
        "--output-path",
        type=str,
        required=False,
        default=None,
        help="Output path (.feather or .csv)",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help="Optional output path for fly manifest CSV",
    )

    args = parser.parse_args()

    yaml_path = Path(args.yaml)
    output_path = Path(args.output_path) if args.output_path else None

    if args.input_dataset:
        input_dataset_path = Path(args.input_dataset)
        if not input_dataset_path.exists():
            raise FileNotFoundError(f"Input dataset not found: {input_dataset_path}")

        dataset = _load_tabular_dataset(input_dataset_path)
        print(f"Loaded existing dataset: {input_dataset_path}")
        print(f"Initial shape: {dataset.shape}")

        if args.restrict_to_yaml_dates:
            dataset, msg = _restrict_to_yaml_dates_if_possible(dataset, yaml_path)
            print(msg)

        if args.nickname:
            dataset, manifest, summary = _filter_existing_dataset_by_nickname(
                dataset=dataset,
                nickname=args.nickname,
                split_registry=Path(args.split_registry),
                control_nickname=args.control_nickname,
                force_control=args.force_control,
            )
            print(summary)
        else:
            if not args.metadata_key or not args.metadata_value:
                _print_available_filters_from_dataset(dataset)
                return

            dataset, manifest = _filter_existing_dataset_by_metadata(
                dataset=dataset,
                metadata_key=args.metadata_key,
                metadata_values=args.metadata_value,
            )
            print(f"Metadata mode: {args.metadata_key} in {args.metadata_value}")

        if args.max_flies_per_group is not None and "fly" in dataset.columns:
            group_col = "Nickname" if "Nickname" in dataset.columns else args.metadata_key
            if group_col and group_col in dataset.columns:
                group_col_name = str(group_col)
                limited_flies = (
                    dataset[[group_col_name, "fly"]]
                    .drop_duplicates()
                    .groupby(group_col_name, as_index=False)
                    .head(args.max_flies_per_group)
                )
                dataset = dataset.merge(limited_flies, on=[group_col_name, "fly"], how="inner")
                print(f"Applied max flies per group in fast mode: {args.max_flies_per_group}")

    else:
        if args.nickname:
            dataset, manifest, info = build_dataset_for_nickname_and_control(
                yaml_path=yaml_path,
                dataset_type=args.dataset_type,
                nickname=args.nickname,
                split_registry_path=Path(args.split_registry),
                control_nickname=args.control_nickname,
                force_control=args.force_control,
                max_experiments=args.max_experiments,
                max_flies_per_group=args.max_flies_per_group,
            )
            print(
                f"Nickname mode: {info.nickname} ({info.genotype}) vs "
                f"{info.control_nickname} ({info.control_genotype})"
            )
        else:
            if not args.metadata_key or not args.metadata_value:
                _print_available_metadata_keys_from_yaml(yaml_path)
                return

            dataset, manifest = build_dataset_from_yaml_by_metadata(
                yaml_path=yaml_path,
                dataset_type=args.dataset_type,
                metadata_key=args.metadata_key,
                metadata_values=args.metadata_value,
                max_experiments=args.max_experiments,
                max_flies_per_value=args.max_flies_per_group,
            )

    if output_path is None:
        raise ValueError("--output-path is required when an actual filtered dataset is produced")

    if dataset.empty:
        print("No data produced for the requested filter.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        dataset.to_csv(output_path, index=False)
    else:
        dataset.to_feather(output_path)

    if args.manifest_path:
        manifest_path = Path(args.manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(manifest_path, index=False)
        print(f"Manifest saved: {manifest_path}")

    print(f"Dataset saved: {output_path}")
    print(f"Shape: {dataset.shape}")
    if "Nickname" in dataset.columns:
        print("Flies by nickname:")
        print(dataset.groupby(dataset["Nickname"].astype(str))["fly"].nunique().to_string())


if __name__ == "__main__":
    main()
