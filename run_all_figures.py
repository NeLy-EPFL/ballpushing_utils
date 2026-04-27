#!/usr/bin/env python3
"""
Run every figure script found under figures/ in one shot.

• Auto-discovers all *.py files recursively under figures/ (sorted by path).
• Skips this script itself if it somehow ends up in the search tree.
• Runs each script in its own subprocess using the current Python interpreter,
  so whichever conda/venv is active will be used.
• Continues after failures and prints a summary at the end.

Usage:
    conda run -n tracking_analysis python run_all_figures.py
    # or, with the env already active:
    python run_all_figures.py
"""

import subprocess
import sys
import time
from pathlib import Path

FIGURES_DIR = Path(__file__).parent / "figures"
THIS_SCRIPT = Path(__file__).resolve()

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BOLD = "\033[1m"
RESET = "\033[0m"


def discover_scripts(root: Path) -> list[Path]:
    """Find every panel script under ``root``.

    Excludes ``run_all_panels.py`` files: those are per-figure orchestrators
    that subprocess their sibling panels, so leaving them in the discovery
    list would double-execute every panel they wrap.

    Excludes anything under an ``old/`` subdir — those are archived legacy
    versions kept for reference, not meant to ship.
    """
    scripts = sorted(root.rglob("*.py"))
    return [
        s
        for s in scripts
        if s.resolve() != THIS_SCRIPT
        and s.name != "run_all_panels.py"
        and "old" not in s.parts
    ]


def run_script(script: Path) -> tuple[bool, float, str]:
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=script.parent,  # run from the script's own directory
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0
    output = result.stdout + result.stderr
    return result.returncode == 0, elapsed, output


def main() -> None:
    scripts = discover_scripts(FIGURES_DIR)
    if not scripts:
        print(f"{YELLOW}No scripts found under {FIGURES_DIR}{RESET}")
        sys.exit(0)

    n = len(scripts)
    print(f"{BOLD}Found {n} figure script(s) under {FIGURES_DIR}{RESET}\n")

    passed, failed = [], []

    for i, script in enumerate(scripts, 1):
        rel = script.relative_to(Path(__file__).parent)
        print(f"[{i}/{n}] {rel} ", end="", flush=True)
        ok, elapsed, output = run_script(script)
        if ok:
            print(f"{GREEN}OK{RESET} ({elapsed:.1f}s)")
            passed.append(script)
        else:
            print(f"{RED}FAILED{RESET} ({elapsed:.1f}s)")
            # indent every output line for readability
            for line in output.strip().splitlines():
                print(f"       {line}")
            failed.append(script)

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}Summary: {GREEN}{len(passed)} passed{RESET}{BOLD}, " f"{RED}{len(failed)} failed{RESET}")
    if failed:
        print(f"\n{RED}Failed scripts:{RESET}")
        for s in failed:
            print(f"  • {s.relative_to(Path(__file__).parent)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
