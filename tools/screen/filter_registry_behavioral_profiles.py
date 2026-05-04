"""Behavioral-profile registry for the silencing-screen Gal4 lines.

Lists each Gal4 line from the silencing screen by behavioral category
(Slower / Faster / Saturate / NoFinish) and, when run as a script, copies
the corresponding grid videos from the raw-grids folder into per-category
folders for downstream review.

The category lists are the curated outcome of the screen analysis — they
are paper-specific. The script logic just sorts videos by line name into
the matching category folder.
"""

import argparse
import os
import shutil
from pathlib import Path

from ballpushing_utils.paths import dataset

Slower = [
    "Empty-Gal4",
    "LC25",
    "LC4",
    "86666 (LH1139)",
    "86637 (LH2220)",
    "86705 (LH1668)",
    "86699 (LH123)",
    "SS52577-gal4 (PBG2‐9.s‐FBℓ3.b‐NO2V.b (PB))",
    "SS00078-gal4 (PBG2‐9.s‐FBℓ3.b‐NO2D.b (PB))",
    "SS02239-gal4 (P-F3LC patch line)",
    "MB312B (PAM-07)",
    "MB043B (PAM-11)",
    "MB315C (PAM-01)",
    "MB504B (All PPL1)",
    "MB063B (PAM-10)",
]

Faster = [
    "LC10-2",
    "PR",
    "R15B07-gal4 (R1/R3/R4d (EB))",
    "R78A01 (ExR1 (EB))",
    "R59B10-gal4 (R4m - medial (EB))",
    "R38H02-gal4 (R4 (EB))",
    "c316 (MB-DPM)",
    "51975 (Tk-GAL4 5Fa)",
    "51976 (Crz-GAL4 3M)",
    "51977 (Crz-GAL4 4M)",
    "51978 (AstA-GAL4 3M)",
    "51979 (AstA-GAL4 5)",
    "51970 (Capa-GAL4 5F)",
    "51988 (Dh31-GAL4 2M)",
    "51985 (Ms-GAL4 1M)",
    "51986 (Ms-GAL4 6Ma)",
    "25681 (NPF-GAL4 2)",
    "25682 (NPF-GAL4 1)",
    "51980 (Burs-GAL4 4M)",
]

Saturate = [
    "LC16-1",
    "LC6",
    "LC12",
    "LPLC2",
    "75823 (LH2446)",
    "75945 (LH1543)",
    "86676 (LH191)",
    "86630 (LH2385)",
    "86632 (LH2392)",
    "86681 (LH141)",
    "86674 (LH247)",
    "86667 (LH1000)",
    "86682 (LH85)",
    "86639 (LH1990)",
    "86685 (LH272)",
    "86707 (LH578)",
    "86671 (LH412)",
    "R19G02 (E-PG)",
    "SS32219-Gal4 (LAL-2)",
    "SS32230-Gal4 (LAL-1)",
    "MB113C (OA-VPM4)",
    "MB296B (PPL1-03)",
    "MB399B",
    "MB434B",
    "MB418B ",
    "MBON-13-GaL4 (MBON-α′2)",
    "MBON-01-GaL4 (MBON-γ5β′2a)",
]

NoFinish = [
    "34497 (MZ19-GAL4)",
    "DDC-gal4",
    "Ple-Gal4.F a.k.a TH-Gal4",
    "MB504B (All PPL1)",
    "VT43924 (MB-APL)",
    "41744 (IR8a mutant)",
    "50742 (MB247-GAL4)",
    "MB247-Gal4",
    "854 (OK107-Gal4)",
]

CATEGORIES = {
    "Slower": Slower,
    "Faster": Faster,
    "Saturate": Saturate,
    "NoFinish": NoFinish,
}

DEFAULT_VIDEOS_IN = dataset("TNT_Screen_RawGrids")
DEFAULT_VIDEOS_OUT = dataset("TNT_Hits_Grids")


def copy_by_category(videos_in: Path, videos_out: Path) -> None:
    for category_name in CATEGORIES:
        (videos_out / category_name).mkdir(parents=True, exist_ok=True)

    files = os.listdir(videos_in)
    for category_name, category_list in CATEGORIES.items():
        for name in category_list:
            for file in files:
                if name in file:
                    src_file = videos_in / file
                    dest_file = videos_out / category_name / file
                    if not dest_file.exists():
                        shutil.copy(src_file, dest_file)
                        print(f"Copied {file} to {dest_file}")
                    else:
                        print(f"File {file} already exists in {dest_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--videos-in",
        type=Path,
        default=DEFAULT_VIDEOS_IN,
        help=f"Directory of raw grid videos (default: {DEFAULT_VIDEOS_IN})",
    )
    parser.add_argument(
        "--videos-out",
        type=Path,
        default=DEFAULT_VIDEOS_OUT,
        help=f"Output root for per-category folders (default: {DEFAULT_VIDEOS_OUT})",
    )
    args = parser.parse_args()
    copy_by_category(args.videos_in, args.videos_out)


if __name__ == "__main__":
    main()
