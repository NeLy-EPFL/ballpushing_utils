"""Download Dataverse feathers required to reproduce the paper figures.

Default behaviour: fetch the minimum set of feathers needed by every
panel in ``figures/**/*.py`` and place them under
``$BALLPUSHING_DATA_ROOT`` (or ``<repo>/Datasets/`` if that env var is
unset). After this completes, ``python run_all_figures.py`` reproduces
every panel with no further configuration.

Usage
-----
    # Default: fetch everything the figure scripts need.
    ballpushing-fetch
    # equivalent: ``python -m ballpushing_utils.dataverse_download``

    # Fetch only one archive
    ballpushing-fetch --archive affordance

    # Fetch into a specific directory (overrides BALLPUSHING_DATA_ROOT)
    ballpushing-fetch --dest ~/dataverse_feathers

    # List what would be fetched, don't download
    ballpushing-fetch --dry-run

    # Re-download files even if a same-sized copy exists
    ballpushing-fetch --force

    # Fetch every public file in the archives, not just the feathers
    # the figures need (includes the raw HDF5 track tars + grid videos)
    ballpushing-fetch --include-raw

The fetcher works against unauthenticated Dataverse REST APIs only, so
no API key is required for the published datasets.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Iterable
from pathlib import Path

from .dataverse_naming import (
    BASENAME_TO_ARCHIVE,
    DATAVERSE_DOIS,
    SERVER_TO_DATAVERSE,
    dataverse_candidates,
)

__all__ = [
    "DEFAULT_API_BASE",
    "collect_required_basenames",
    "list_dataset_files",
    "main",
    "resolve_destination",
]


DEFAULT_API_BASE = "https://dataverse.harvard.edu"

# Dataverse returns 403 to bare urllib defaults. Use a recognisable UA
# so requests succeed (and admins know who's hitting their endpoints).
_USER_AGENT = "ballpushing-fetch/0.1 (+https://github.com/NeLy-EPFL/ballpushing_utils)"


def _http_get(url: str, *, timeout: float = 60.0) -> "urllib.request.addinfourl":
    req = urllib.request.Request(
        url, headers={"Accept": "*/*", "User-Agent": _USER_AGENT}
    )
    return urllib.request.urlopen(req, timeout=timeout)


# ---------------------------------------------------------------------------
# Helpers — destination, file list, AST walk over figures/.
# ---------------------------------------------------------------------------


def resolve_destination(arg_dest: str | os.PathLike[str] | None) -> Path:
    """Resolve the download destination directory.

    Precedence: ``--dest`` argument → ``$BALLPUSHING_DATA_ROOT`` →
    ``<repo>/Datasets/``. The directory is created if missing.
    """
    if arg_dest:
        dest = Path(arg_dest).expanduser()
    else:
        env = os.environ.get("BALLPUSHING_DATA_ROOT")
        if env:
            dest = Path(env).expanduser()
        else:
            # ``<repo>/Datasets/`` — same constant used by
            # ``paths.data_root()``'s fallback.
            dest = Path(__file__).resolve().parents[2] / "Datasets"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _figures_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "figures"


def _dataset_literals_in(py_path: Path) -> list[str]:
    """Return every string literal passed to ``dataset(...)`` in the file.

    Uses an AST walk to keep the source of truth in the figure scripts.
    """
    try:
        tree = ast.parse(py_path.read_text())
    except SyntaxError:
        return []
    out: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "dataset"
            and node.args
        ):
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                out.append(arg.value)
    return out


def collect_required_basenames() -> dict[str, list[str]]:
    """Walk ``figures/**/*.py`` and group the required basenames by archive.

    Returns ``{archive_label: sorted([basenames])}``. Skips scripts under
    ``old/`` and ``run_all_panels.py`` wrappers. Literals that don't map
    via :func:`dataverse_candidates` are silently ignored (they're
    handled separately — see ``KNOWN_UNMAPPED`` in
    ``tests/unit/test_dataverse_naming.py``).
    """
    figures = _figures_dir()
    required: dict[str, set[str]] = {}
    for py in sorted(figures.rglob("*.py")):
        if "/old/" in str(py) or py.name == "run_all_panels.py":
            continue
        for literal in _dataset_literals_in(py):
            candidates = dataverse_candidates(literal)
            if not candidates:
                continue
            for basename in candidates:
                archive = BASENAME_TO_ARCHIVE.get(basename)
                if archive is None:
                    continue
                required.setdefault(archive, set()).add(basename)
    return {archive: sorted(basenames) for archive, basenames in required.items()}


def _required_split_aware(required_basenames: Iterable[str], remote_files: dict[str, dict]) -> list[str]:
    """Expand requested basenames into the actual filenames published.

    A basename ``foo.feather`` that doesn't exist on the server but has
    matching ``foo-1.feather``, ``foo-2.feather``, … parts is expanded
    into the part filenames. Order is preserved.
    """
    out: list[str] = []
    for basename in required_basenames:
        if basename in remote_files:
            out.append(basename)
            continue
        stem = Path(basename).stem
        suffix = Path(basename).suffix
        parts: list[tuple[int, str]] = []
        for name in remote_files:
            if not name.startswith(f"{stem}-") or not name.endswith(suffix):
                continue
            tail = name[len(stem) + 1 : -len(suffix)] if suffix else name[len(stem) + 1 :]
            if tail.isdigit():
                parts.append((int(tail), name))
        if parts:
            parts.sort()
            out.extend(name for _, name in parts)
        else:
            # Neither the single file nor the split parts are on the
            # server. The verification step at the end will catch this.
            out.append(basename)
    return out


# ---------------------------------------------------------------------------
# Dataverse API helpers.
# ---------------------------------------------------------------------------


def list_dataset_files(
    doi: str, server: str = DEFAULT_API_BASE, *, timeout: float = 60.0
) -> dict[str, dict]:
    """Return the published files of a Dataverse dataset, keyed by filename.

    Each value is a dict with at least ``id`` (numeric ``dataFile.id``,
    used for the download endpoint), ``size``, and ``md5`` when the
    server publishes a checksum.
    """
    import json

    url = f"{server.rstrip('/')}/api/datasets/:persistentId/"
    full = f"{url}?{urllib.parse.urlencode({'persistentId': doi})}"
    with _http_get(full, timeout=timeout) as resp:
        payload = json.load(resp)
    latest = payload["data"]["latestVersion"]
    files: dict[str, dict] = {}
    for entry in latest.get("files", []):
        df = entry.get("dataFile", {}) or {}
        name = df.get("filename") or entry.get("label")
        if not name:
            continue
        files[name] = {
            "id": df.get("id"),
            "size": df.get("filesize"),
            "md5": df.get("md5"),
            "contentType": df.get("contentType"),
        }
    return files


def _human_size(n: float | None) -> str:
    if n is None:
        return "?"
    nf = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if nf < 1024:
            return f"{nf:.2f} {unit}"
        nf /= 1024
    return f"{nf:.2f} PiB"


def _md5_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _download_one(
    file_id: int,
    target: Path,
    *,
    server: str,
    expected_size: int | None,
    expected_md5: str | None,
    verify_md5: bool,
) -> None:
    """Stream a Dataverse data file to ``target`` (atomic-by-rename)."""
    url = f"{server.rstrip('/')}/api/access/datafile/{file_id}"
    tmpdir = target.parent
    tmpdir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".part", dir=tmpdir)
    tmp_path = Path(tmp_name)
    try:
        os.close(fd)
        with _http_get(url, timeout=120.0) as resp, tmp_path.open("wb") as out:
            shutil.copyfileobj(resp, out, length=1 << 20)
        if expected_size is not None and tmp_path.stat().st_size != expected_size:
            raise IOError(
                f"size mismatch for {target.name}: "
                f"got {tmp_path.stat().st_size}, expected {expected_size}"
            )
        if verify_md5 and expected_md5:
            actual = _md5_of(tmp_path)
            if actual != expected_md5:
                raise IOError(
                    f"md5 mismatch for {target.name}: got {actual}, expected {expected_md5}"
                )
        tmp_path.replace(target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ballpushing-fetch",
        description=(
            "Download Dataverse feathers required to reproduce the "
            "paper figures. Reads the file list from figures/**/*.py "
            "and fetches the matching basenames from the three "
            "published Harvard Dataverse archives (Affordance, Screen, "
            "Exploration)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help=(
            "Destination directory. Order: --dest → $BALLPUSHING_DATA_ROOT "
            "→ <repo>/Datasets/. Created if missing."
        ),
    )
    parser.add_argument(
        "--archive",
        choices=sorted(DATAVERSE_DOIS),
        action="append",
        default=None,
        help="Fetch only this archive (repeatable). Default: all three.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the manifest without downloading.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even when a same-sized local copy exists.",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help=(
            "Fetch every public file in the archives, not just the "
            "feathers needed by the figure scripts. Adds raw HDF5 track "
            "tars + grid videos (~tens of GB). Only useful for the "
            "rerun-from-tracks path via dataset_builder.py."
        ),
    )
    parser.add_argument(
        "--verify-md5",
        action="store_true",
        help=(
            "MD5-verify each downloaded file against the server-published "
            "checksum. Slower but catches corrupted downloads."
        ),
    )
    parser.add_argument(
        "--server",
        default=DEFAULT_API_BASE,
        help="Dataverse base URL (only useful if testing against a mirror).",
    )
    return parser.parse_args(argv)


def _archive_label_for(filename: str, remote_files_by_archive: dict[str, dict[str, dict]]) -> str | None:
    for label, files in remote_files_by_archive.items():
        if filename in files:
            return label
    return None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    dest = resolve_destination(args.dest)
    selected_archives = args.archive or list(DATAVERSE_DOIS)

    print(f"Destination : {dest}")
    print(f"Archives    : {', '.join(selected_archives)}")
    print()

    # Discover what's actually published on each archive.
    print("Querying Dataverse for file listings...")
    remote_by_archive: dict[str, dict[str, dict]] = {}
    for archive in selected_archives:
        doi = DATAVERSE_DOIS[archive]
        try:
            remote_by_archive[archive] = list_dataset_files(doi, server=args.server)
        except Exception as exc:
            print(f"  ERROR listing {archive} ({doi}): {exc}", file=sys.stderr)
            return 2
        print(f"  {archive:<12} {doi:<30} {len(remote_by_archive[archive])} files")
    print()

    # Decide what to fetch.
    if args.include_raw:
        wanted_by_archive: dict[str, list[str]] = {
            archive: sorted(remote_by_archive[archive].keys())
            for archive in selected_archives
        }
    else:
        required = collect_required_basenames()
        wanted_by_archive = {}
        for archive in selected_archives:
            basenames = required.get(archive, [])
            wanted_by_archive[archive] = _required_split_aware(basenames, remote_by_archive[archive])

    # Build the manifest.
    plan: list[tuple[str, str, int | None, str | None, Path]] = []  # (archive, name, size, md5, target)
    missing: list[tuple[str, str]] = []
    for archive in selected_archives:
        for name in wanted_by_archive[archive]:
            entry = remote_by_archive[archive].get(name)
            if entry is None:
                missing.append((archive, name))
                continue
            target = dest / name
            plan.append((archive, name, entry["size"], entry["md5"], target))

    if missing:
        print("WARNING: the following requested files are not present on Dataverse:")
        for archive, name in missing:
            print(f"  [{archive}] {name}")
        print(
            "  Either the mapping in src/ballpushing_utils/dataverse_naming.py "
            "is outdated,\n  or the file hasn't been uploaded yet."
        )
        print()

    # Print the manifest.
    total_bytes = sum((size or 0) for _, _, size, _, _ in plan)
    print(f"Files in plan: {len(plan)}  ({_human_size(total_bytes)} on-disk total)")
    for archive, name, size, _md5, target in plan:
        status = ""
        if target.exists():
            local = target.stat().st_size
            if size is None:
                status = "  [exists locally]"
            elif local == size and not args.force:
                status = "  [exists locally, size match — skipping]"
            elif local == size and args.force:
                status = "  [exists locally — re-downloading (--force)]"
            else:
                status = f"  [size mismatch: local={local}, remote={size} — re-downloading]"
        print(f"  [{archive}] {name}  ({_human_size(size)}){status}")
    print()

    if args.dry_run:
        print("--dry-run: exiting without downloading.")
        return 0

    # Download.
    failures: list[tuple[str, Exception]] = []
    skipped = 0
    for archive, name, size, md5, target in plan:
        if (
            target.exists()
            and (size is None or target.stat().st_size == size)
            and not args.force
        ):
            skipped += 1
            continue
        entry = remote_by_archive[archive][name]
        file_id = entry["id"]
        print(f"Downloading {name}  ({_human_size(size)})  → {target.name}")
        try:
            _download_one(
                file_id,
                target,
                server=args.server,
                expected_size=size,
                expected_md5=md5,
                verify_md5=args.verify_md5,
            )
            print("  ok")
        except Exception as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            failures.append((name, exc))

    print()
    print("=" * 60)
    print(f"Done. Downloaded: {len(plan) - skipped - len(failures)}   "
          f"Skipped (already on disk): {skipped}   "
          f"Failed: {len(failures)}")

    if failures:
        for name, exc in failures:
            print(f"  FAILED: {name} — {exc}", file=sys.stderr)
        return 2

    # Final verification: every figure-required path resolves under dest.
    if not args.include_raw and not missing:
        _verify_resolves(dest, selected_archives)

    return 2 if missing else 0


def _verify_resolves(dest: Path, selected_archives: list[str]) -> None:
    """Sanity-check that every figure-required path resolves under ``dest``.

    Uses the same resolver the figure scripts use at runtime, after
    pointing it at ``dest``. Surfaces missing basenames so a typo in the
    mapping or an upload omission is caught at download time rather than
    at figure-execution time.
    """
    from .dataverse_naming import expand_split_parts

    print()
    print("Verifying every figure-required feather resolves...")
    bad: list[tuple[str, str]] = []
    server_paths_by_archive: dict[str, list[str]] = {a: [] for a in selected_archives}
    for server_path, basenames in SERVER_TO_DATAVERSE.items():
        archives_for_pattern = {BASENAME_TO_ARCHIVE.get(b) for b in basenames}
        if not archives_for_pattern & set(selected_archives):
            continue
        for archive in archives_for_pattern & set(selected_archives):
            if archive is None:
                continue
            server_paths_by_archive[archive].append(server_path)

    seen: set[str] = set()
    for archive in selected_archives:
        for server_path in server_paths_by_archive[archive]:
            if server_path in seen:
                continue
            seen.add(server_path)
            basenames = SERVER_TO_DATAVERSE[server_path]
            resolved_any = False
            for basename in basenames:
                hits = expand_split_parts(basename, dest)
                if hits:
                    resolved_any = True
                    break
            if not resolved_any:
                bad.append((archive, server_path))

    if bad:
        print(f"  {len(bad)} server path(s) NOT resolvable from {dest}:")
        for archive, sp in bad:
            print(f"  [{archive}] {sp}")
        print(
            "  Fix: re-run ballpushing-fetch, or check that the basenames in "
            "BASENAME_TO_ARCHIVE\n  match the ones published on the Dataverse."
        )
    else:
        print(f"  All {len(seen)} figure-required server paths resolve. ✓")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
