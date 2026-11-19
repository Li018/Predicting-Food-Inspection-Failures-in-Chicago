#!/usr/bin/env python3
"""Reproduce Topic 1 from the committed source hashes; keep detailed logs local.

Use --dry-run to check source hashes, script syntax and the scheduled CLI arguments
without fitting models or regenerating outputs. --download may fill missing local
snapshots only after a staged download matches the original committed hashes.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = ROOT / "data/snapshots"
STAGED = ROOT / ".local/download_snapshot"
MANIFEST = "data/snapshots/manifest.json"
SOURCE_NAMES = ("inspections.parquet", "licenses.parquet")
STEPS = (
    ("profile_data.py", ()),
    ("run_model.py", ("--family", "prevalence")),
    ("run_model.py", ("--family", "logistic_regression")),
    ("run_model.py", ("--family", "lightgbm")),
    ("compare_models.py", ()),
    ("plot_data_drift.py", ()),
    ("plot_model_performance.py", ()),
    ("plot_calibration.py", ()),
    ("explain_model.py", ()),
    ("analyze_errors.py", ()),
    ("analyze_feature_context.py", ()),
    ("analyze_lag_sensitivity.py", ()),
    ("qa_delivery.py", ()),
)


def source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_manifest() -> tuple[bytes, dict]:
    """Read HEAD before any download; never adopt newly downloaded metadata."""
    result = subprocess.run(
        ["git", "show", f"HEAD:{MANIFEST}"], cwd=ROOT, capture_output=True, check=False,
    )
    if result.returncode:
        raise RuntimeError("A Git checkout with the committed source manifest is required.")
    original = result.stdout
    expected = json.loads(original)
    if set(expected["files"]) != set(SOURCE_NAMES):
        raise RuntimeError("The committed manifest must identify exactly the two source Parquet files.")
    if (ROOT / MANIFEST).read_bytes() != original:
        raise RuntimeError("The working source manifest differs from HEAD; resolve it before reproducing.")
    config = json.loads((ROOT / "config/topic1.json").read_text())
    if config["data_end_date"] != expected["data_end_date"]:
        raise RuntimeError("The configured data end date differs from the committed snapshot.")
    return original, expected


def check_sources(directory: Path, expected: dict) -> list[str]:
    missing = []
    for name in SOURCE_NAMES:
        path = directory / name
        if not path.is_file():
            missing.append(name)
            continue
        record = expected["files"][name]
        if path.stat().st_size != record["bytes"] or source_hash(path) != record["sha256"]:
            raise RuntimeError(
                f"Source hash mismatch: {path.relative_to(ROOT)}. This is not the frozen study "
                "snapshot; no model has been run and the committed manifest is unchanged."
            )
    return missing


def command(script: str, arguments: tuple[str, ...]) -> list[str]:
    return [sys.executable, str(ROOT / "scripts" / script), *arguments]


def validate_plan(plan: tuple) -> None:
    """Static validation only: do not import scripts with executable entry points."""
    for script, arguments in plan:
        path = ROOT / "scripts" / script
        if not path.is_file():
            raise RuntimeError(f"Missing reproduction script: scripts/{script}")
        tree = ast.parse(path.read_text(), filename=str(path))
        declarations = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ]
        flags = {
            arg.value for node in declarations for arg in node.args
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
        }
        provided = {arg for arg in arguments if arg.startswith("--")}
        if not provided.issubset(flags):
            raise RuntimeError(f"Scheduled CLI options are not declared by scripts/{script}.")
        for declaration in declarations:
            required = any(
                item.arg == "required" and isinstance(item.value, ast.Constant)
                and item.value.value is True for item in declaration.keywords
            )
            aliases = {
                arg.value for arg in declaration.args if isinstance(arg, ast.Constant)
                and isinstance(arg.value, str)
            }
            if required and not aliases.intersection(provided):
                raise RuntimeError(f"A required CLI option is missing for scripts/{script}.")
        if script == "run_model.py":
            grids = next(
                node.value for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "GRIDS"
                        for target in node.targets)
            )
            families = {key.value for key in grids.keys if isinstance(key, ast.Constant)}
            if arguments != ("--family", arguments[-1]) or arguments[-1] not in families:
                raise RuntimeError("The scheduled model family is not implemented by run_model.py.")


def run_logged(script: str, arguments: tuple[str, ...], log_dir: Path, label: str) -> None:
    suffix = f"_{arguments[-1]}" if script == "run_model.py" else ""
    log = log_dir / f"{Path(script).stem}{suffix}.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["PYTHONUNBUFFERED"] = "1"
    started = time.monotonic()
    print(f"[{label}] {script}{suffix}: running; log={log.relative_to(ROOT)}", flush=True)
    with log.open("w") as stream:
        result = subprocess.run(
            command(script, arguments), cwd=ROOT, env=env,
            stdout=stream, stderr=subprocess.STDOUT, check=False,
        )
    if result.returncode:
        raise RuntimeError(
            f"{script} exited {result.returncode}; inspect {log.relative_to(ROOT)} locally."
        )
    print(f"[{label}] OK ({time.monotonic() - started:.1f}s)", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Validate and show the plan; no analysis runs.")
    parser.add_argument(
        "--download", action="store_true",
        help="Download missing snapshots to local staging, then require the committed hashes.",
    )
    args = parser.parse_args()
    original, expected = frozen_manifest()
    download_args = (
        "--end-date", expected["data_end_date"],
        "--output-dir", str(STAGED), "--raw-dir", str(ROOT / "data/raw"),
    )
    validate_plan(STEPS + (("download_data.py", download_args),))
    missing = check_sources(SNAPSHOTS, expected)
    if missing and not args.download:
        raise RuntimeError(
            "Missing local sources: " + ", ".join(missing)
            + ". Run this entry point with --download, or stage a download with:\n"
            + shlex.join(command("download_data.py", download_args))
            + "\nThen rerun with --download to verify and install only matching files. "
            "City backfills or different Parquet serialization may prevent an exact hash match."
        )
    if args.dry_run:
        print("DRY RUN: script syntax, scheduled options and model families checked; no analysis executed.")
        print("Source verification: " + (
            "existing files match; missing files require a future staged download."
            if missing else "both local snapshots match the committed manifest."
        ))
        plan = (("download_data.py", download_args),) + STEPS if missing else STEPS
        for index, (script, arguments) in enumerate(plan, 1):
            print(f"{index:02d}. {shlex.join([sys.executable, 'scripts/' + script, *arguments])}")
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log_dir = ROOT / ".local/logs" / f"reproduce-{stamp}"
    log_dir.mkdir(parents=True)
    if missing:
        run_logged("download_data.py", download_args, log_dir, "download")
        if check_sources(STAGED, expected):
            raise RuntimeError("The staged download did not produce both required snapshots.")
        if (ROOT / MANIFEST).read_bytes() != original:
            raise RuntimeError("The source manifest changed during download; stopping before analysis.")
        for name in missing:
            destination = SNAPSHOTS / name
            with (STAGED / name).open("rb") as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target)
        if check_sources(SNAPSHOTS, expected):
            raise RuntimeError("Local snapshot installation is incomplete.")
    print("Source hashes verified. Analysis outputs will be regenerated; full logs remain local.")
    for index, (script, arguments) in enumerate(STEPS, 1):
        run_logged(script, arguments, log_dir, f"{index}/{len(STEPS)}")
    print(f"Completed all {len(STEPS)} stages, including delivery QA. Logs: {log_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, SyntaxError) as error:
        print(f"STOP: {error}", file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        print("Stopped by user; detailed logs remain under .local/logs.", file=sys.stderr)
        raise SystemExit(130) from None
