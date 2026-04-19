"""Re-encode one fly's video(s) under ``tests/fixtures/sample_data/``.

Workflow on the workstation (where the raw recordings live)::

    python tools/compress_sample_fly.py \\
        --source /mnt/upramdya_data/MD/MultiMazeRecorder/Videos/231130_TNT_Fine_2_Videos_Tracked/arena2/corridor5 \\
        --data-root /mnt/upramdya_data/MD

This copies the fly corridor + its parent experiment's metadata into
``tests/fixtures/sample_data/<relative/path>/`` with:

* every ``*.mp4`` re-encoded with ``libx264 -crf 28 -preset veryslow`` and
  ``-an`` (audio dropped — SLEAP videos don't carry any). Resolution and
  frame rate are preserved so the SLEAP ``.h5`` pixel coordinates keep
  lining up with the video.
* every ``*.h5`` copied as-is (tracks are already small).
* ``Metadata.json`` and ``fps.npy`` copied from the experiment directory
  (one level up from the arena directory).

Idempotent: re-running against the same destination overwrites videos and
refreshes copies. Files that already look compressed (per-pixel bitrate
below a heuristic threshold) are skipped unless ``--force`` is passed.

This script is only useful inside the lab (or wherever the raw data
lives). External users pull the already-compressed fixtures via Git LFS.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURES_ROOT = REPO_ROOT / "tests" / "fixtures" / "sample_data"

# Heuristic: if the source video is already below this many bits per pixel
# per frame it's probably already compressed and re-encoding would only
# hurt quality. Skip unless --force. (Typical SLEAP recordings at CRF 23
# land around 0.15 bpp; CRF 28 around 0.08 bpp.)
SKIP_BPP_THRESHOLD = 0.05


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    """Run ``cmd`` with stderr/stdout inherited so the user sees ffmpeg progress."""
    print(f"$ {' '.join(map(str, cmd))}", flush=True)
    return subprocess.run(cmd, check=check)


def ffprobe_bpp(video: Path) -> float | None:
    """Return bits-per-pixel-per-frame for *video*, or ``None`` if unknown."""
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,bit_rate",
                "-of",
                "default=nw=1:nk=1",
                str(video),
            ],
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    parts = out.strip().splitlines()
    if len(parts) < 4:
        return None
    try:
        width = int(parts[0])
        height = int(parts[1])
        num, _, den = parts[2].partition("/")
        fps = float(num) / float(den) if den else float(parts[2])
        bit_rate = int(parts[3])
    except (ValueError, ZeroDivisionError):
        return None
    if not (width and height and fps and bit_rate):
        return None
    return bit_rate / (width * height * fps)


def compress_video(src: Path, dst: Path, *, crf: int, force: bool) -> None:
    """Re-encode ``src`` to ``dst`` at ``-crf`` quality."""
    if dst.exists() and not force:
        print(f"  - {dst.name}: already exists, skipping (use --force to overwrite)")
        return
    bpp = ffprobe_bpp(src)
    if bpp is not None and bpp < SKIP_BPP_THRESHOLD and not force:
        print(
            f"  - {src.name}: already ~{bpp:.3f} bits/pixel/frame "
            f"(< {SKIP_BPP_THRESHOLD}); copying verbatim instead of re-encoding"
        )
        shutil.copy2(src, dst)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-stats",
            "-i",
            str(src),
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-preset",
            "veryslow",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            "-fps_mode",
            "passthrough",
            str(dst),
        ]
    )
    src_mb = src.stat().st_size / 1_048_576
    dst_mb = dst.stat().st_size / 1_048_576
    print(f"  - {src.name}: {src_mb:.1f} MB -> {dst_mb:.1f} MB ({dst_mb / src_mb:.1%})")


def copy_verbatim(src: Path, dst: Path, *, force: bool) -> None:
    if dst.exists() and not force:
        print(f"  - {dst.name}: already exists, skipping")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  - {dst.name}: copied ({dst.stat().st_size / 1024:.1f} KB)")


def process_fly(
    source_fly: Path,
    fixtures_root: Path,
    data_root: Path,
    *,
    crf: int,
    force: bool,
) -> Path:
    """Mirror ``source_fly`` + its experiment metadata into ``fixtures_root``.

    Returns the absolute path of the destination fly directory.
    """
    source_fly = source_fly.resolve()
    data_root = data_root.resolve()
    try:
        relative = source_fly.relative_to(data_root)
    except ValueError as err:
        raise SystemExit(
            f"--source ({source_fly}) is not under --data-root ({data_root}); "
            "pass --data-root matching your data layout."
        ) from err

    dest_fly = fixtures_root / relative
    print(f"Destination: {dest_fly}")
    dest_fly.mkdir(parents=True, exist_ok=True)

    # Compress videos
    mp4s = sorted(source_fly.glob("*.mp4"))
    if not mp4s:
        raise SystemExit(f"No *.mp4 files under {source_fly}; wrong path?")
    print(f"Videos ({len(mp4s)}):")
    for mp4 in mp4s:
        compress_video(mp4, dest_fly / mp4.name, crf=crf, force=force)

    # Copy H5 tracks verbatim
    h5s = sorted(source_fly.glob("*.h5"))
    print(f"Tracks ({len(h5s)}):")
    for h5 in h5s:
        copy_verbatim(h5, dest_fly / h5.name, force=force)

    # Copy any JSON the fly might have (some pipelines ship per-corridor configs)
    for extra in sorted(source_fly.glob("*.json")):
        copy_verbatim(extra, dest_fly / extra.name, force=force)

    # Experiment-level metadata: Metadata.json + fps.npy live at the
    # experiment-directory level, which is arena/.parent.parent relative
    # to the corridor directory. Walk up until we find Metadata.json.
    experiment_dir = None
    for candidate in (source_fly.parent.parent, source_fly.parent):
        if (candidate / "Metadata.json").exists():
            experiment_dir = candidate
            break
    if experiment_dir is None:
        raise SystemExit(
            f"Could not find Metadata.json walking up from {source_fly}. "
            "Is this a ballpushing-style fly directory?"
        )
    dest_exp = fixtures_root / experiment_dir.relative_to(data_root)
    print(f"Experiment metadata ({experiment_dir.name}):")
    for name in ("Metadata.json", "fps.npy"):
        src = experiment_dir / name
        if src.exists():
            copy_verbatim(src, dest_exp / name, force=force)
        elif name == "Metadata.json":
            raise SystemExit(f"Missing required file: {src}")
        else:
            print(f"  - {name}: not present in source (optional), skipping")

    return dest_fly


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Path to the fly corridor directory to bundle (e.g. .../arena2/corridor5).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/upramdya_data/MD"),
        help="Absolute root of the raw data tree; --source must live under this. "
        "Default: /mnt/upramdya_data/MD (the NeLy lab share).",
    )
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=DEFAULT_FIXTURES_ROOT,
        help="Destination root under which the fly's relative path is mirrored. "
        f"Default: {DEFAULT_FIXTURES_ROOT.relative_to(REPO_ROOT)}",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=28,
        help="x264 CRF. Higher = smaller + lossier. Default: 28 (aggressive).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite files that already exist under --fixtures-root.",
    )
    args = parser.parse_args(argv)

    if not args.source.is_dir():
        print(f"error: --source {args.source} is not a directory", file=sys.stderr)
        return 2
    if shutil.which("ffmpeg") is None:
        print("error: ffmpeg not found on PATH; install it and retry", file=sys.stderr)
        return 2

    dest = process_fly(
        args.source,
        args.fixtures_root,
        args.data_root,
        crf=args.crf,
        force=args.force,
    )
    print()
    print(f"Done. Fixture fly written to: {dest}")
    print(
        "Next: `git add tests/fixtures .gitattributes && git commit`. "
        "Git LFS will catch the .mp4 / .h5 / .npy files automatically."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
