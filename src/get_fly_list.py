import argparse
from pathlib import Path
import yaml
import json

from Ballpushing_utils.fly_metadata import FlyMetadata
from Ballpushing_utils.experiment import Experiment


def load_yaml_dirs(yaml_path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return [Path(d) for d in data.get("directories", [])]


def find_corridors_by_metadata(directories, key, value):
    matching_corridors = []
    for recording in directories:
        if not recording.exists():
            continue
        try:
            exp = Experiment(recording, metadata_only=True)
        except Exception as e:
            continue
        for arena in recording.iterdir():
            if not arena.is_dir():
                continue
            arena_key = arena.name.lower()
            # Check if the key exists for this arena in the experiment metadata
            for corridor in arena.iterdir():
                if not corridor.is_dir():
                    continue
                try:
                    # exp.metadata[key][arena_key] gives the value for this arena
                    val = exp.metadata.get(key, {}).get(arena_key, None)
                    if val == value:
                        matching_corridors.append(str(corridor.resolve()))
                except Exception:
                    continue
    return matching_corridors


def main():
    parser = argparse.ArgumentParser(description="Find corridor folders by metadata key/value.")
    parser.add_argument("--yaml", required=True, help="YAML file listing experiment root folders")
    parser.add_argument("--key", required=True, help="Metadata key to match (e.g. Nickname)")
    parser.add_argument("--value", required=True, help="Metadata value to match (e.g. PR)")
    parser.add_argument("--output", default=None, help="Optional output file to save results")
    args = parser.parse_args()

    directories = load_yaml_dirs(args.yaml)
    results = find_corridors_by_metadata(directories, args.key, args.value)

    if args.output:
        # Only save as YAML
        yaml_output = args.output
        if not yaml_output.endswith(".yaml"):
            yaml_output = yaml_output + ".yaml"
        with open(yaml_output, "w") as yf:
            yaml.dump({"directories": results}, yf)
    else:
        for path in results:
            print(path)


if __name__ == "__main__":
    main()
