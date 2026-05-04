#!/usr/bin/env python3
"""Utilities for creating split tar archives when folders are too large."""

from __future__ import annotations

import tarfile
from pathlib import Path


def _path_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _partition_children(root_dir: Path, max_bytes: int) -> list[list[Path]]:
    """Partition immediate children of root_dir into sequential size-bounded parts."""
    children = sorted([p for p in root_dir.iterdir()], key=lambda p: p.name)
    if not children:
        return []

    parts: list[list[Path]] = []
    current: list[Path] = []
    current_size = 0

    for child in children:
        size = _path_size_bytes(child)
        if current and current_size + size > max_bytes:
            parts.append(current)
            current = [child]
            current_size = size
        else:
            current.append(child)
            current_size += size

    if current:
        parts.append(current)

    return parts


def _add_subset_to_tar(
    members: list[Path],
    archive_path: Path,
    arc_root: str,
) -> int:
    """Create tar archive containing only selected immediate children.

    Returns number of archived regular files.
    """
    with tarfile.open(archive_path, "w:") as tar:
        for member in members:
            tar.add(member, arcname=f"{arc_root}/{member.name}")

    with tarfile.open(archive_path, "r:") as tar:
        return sum(1 for m in tar.getmembers() if m.isfile())


def archive_directory_with_split(
    source_dir: Path,
    output_root: Path,
    max_archive_size_bytes: int,
) -> bool:
    """Archive source_dir into one or more .tar files, splitting by size.

    If source_dir total size is <= max_archive_size_bytes, creates <name>.tar.
    Otherwise creates <name>-1.tar, <name>-2.tar, ... by partitioning immediate
    children of source_dir while preserving deterministic ordering.

    Returns True when archive(s) are created and source_dir is removed.
    Returns False when skipped due to existing archives or integrity mismatch.
    """
    base = source_dir.name
    existing = [
        p
        for p in output_root.glob(f"{base}*.tar")
        if p.name == f"{base}.tar" or p.name.startswith(f"{base}-")
    ]
    if existing:
        print(
            "  [SKIP] Existing archive(s) detected for "
            f"{base}: {[p.name for p in sorted(existing)]}"
        )
        return False

    source_file_count = sum(1 for p in source_dir.rglob("*") if p.is_file())
    source_size = _path_size_bytes(source_dir)

    if source_size <= max_archive_size_bytes:
        archive_path = output_root / f"{base}.tar"
        print(f"  Archiving {base} -> {archive_path.name} ...", end="", flush=True)
        archived_files = _add_subset_to_tar(
            [p for p in source_dir.iterdir()],
            archive_path,
            base,
        )
        size_gb = archive_path.stat().st_size / (1024**3)
        print(f" done ({size_gb:.2f} GB)")

        if archived_files != source_file_count:
            print(
                f"  [ERROR] Archive has {archived_files} files but source has "
                f"{source_file_count} - NOT removing source directory"
            )
            return False

        return _remove_source(source_dir)

    parts = _partition_children(source_dir, max_archive_size_bytes)
    print(
        f"  Splitting {base}: source size {source_size / (1024**3):.2f} GB "
        f"into {len(parts)} archive part(s)"
    )

    total_archived_files = 0
    created_archives: list[Path] = []
    for i, members in enumerate(parts, 1):
        part_name = f"{base}-{i}"
        archive_path = output_root / f"{part_name}.tar"
        print(f"  Archiving {part_name} -> {archive_path.name} ...", end="", flush=True)
        archived_files = _add_subset_to_tar(members, archive_path, part_name)
        total_archived_files += archived_files
        created_archives.append(archive_path)
        size_gb = archive_path.stat().st_size / (1024**3)
        print(f" done ({size_gb:.2f} GB)")

    if total_archived_files != source_file_count:
        print(
            f"  [ERROR] Split archives contain {total_archived_files} files but source has "
            f"{source_file_count} - NOT removing source directory"
        )
        for p in created_archives:
            p.unlink(missing_ok=True)
        return False

    return _remove_source(source_dir)


def _remove_source(source_dir: Path) -> bool:
    for p in sorted(source_dir.rglob("*"), reverse=True):
        if p.is_file() or p.is_symlink():
            p.unlink(missing_ok=True)
        elif p.is_dir():
            p.rmdir()
    source_dir.rmdir()
    print(f"  Removed source directory: {source_dir.name}")
    return True
