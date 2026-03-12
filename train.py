#!/usr/bin/env python3
"""
Cppfort research loop harness.

One invocation equals one bounded evaluation run:
- configure/build if needed
- run the authoritative conveyor surface when possible
- snapshot key artifacts
- emit structured machine-readable output

This is the seed. The agent loop mutates repo code; this file measures the result.
"""

from __future__ import annotations

import argparse
import tempfile
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BUILD_DIR = ROOT / "build"
RUNS_DIR = ROOT / "runs"
CONVEYOR_ROOT = BUILD_DIR / "conveyor"
CPPFRONT_REPO = ROOT / "third_party" / "cppfront"
CPPFRONT_SOURCE = CPPFRONT_REPO / "source" / "cppfront.cpp"
CPPFRONT_INCLUDE = CPPFRONT_REPO / "include"
CPPFRONT_REGRESSION = CPPFRONT_REPO / "regression-tests"
CPPFRONT_UPSTREAM = "https://github.com/hsutter/cppfront.git"


def now_ts() -> float:
    return time.time()


RUN_ID = os.environ.get("AUTORESEARCH_RUN_ID") or f"{int(now_ts())}-{os.getpid()}"
PARENT_RUN_ID = os.environ.get("AUTORESEARCH_PARENT_RUN_ID", "")
HYPOTHESIS = os.environ.get("AUTORESEARCH_HYPOTHESIS", "")
MUTATION_TAG = os.environ.get("AUTORESEARCH_MUTATION_TAG", "")
OBJECTIVE = os.environ.get(
    "AUTORESEARCH_OBJECTIVE",
    "grow cppfort by one honest bounded slice",
)
RUN_DIR = Path(os.environ.get("AUTORESEARCH_RUN_DIR", RUNS_DIR / RUN_ID))
EVENT_LOG_INTERVAL = int(os.environ.get("AUTORESEARCH_EVENT_LOG_INTERVAL", "1"))

RESULT_PATH = RUN_DIR / "result.json"
COMMAND_LOG = RUN_DIR / "command.log"
SUMMARY_SNAPSHOT = RUN_DIR / "CONVEYOR_SUMMARY.md"
CPPFRONT_FAILURES_SNAPSHOT = RUN_DIR / "cppfront_failures.txt"
CPPFORT_FAILURES_SNAPSHOT = RUN_DIR / "cppfort_failures.txt"


def ensure_run_dir() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)


def emit_event(kind: str, **payload: object) -> None:
    event = {
        "kind": kind,
        "run_id": RUN_ID,
        "timestamp": now_ts(),
        "objective": OBJECTIVE,
        **payload,
    }
    print(f"event_json: {json.dumps(event, sort_keys=True)}", flush=True)


def write_result(status: str, **payload: object) -> dict[str, object]:
    ensure_run_dir()
    result = {
        "status": status,
        "run_id": RUN_ID,
        "parent_run_id": PARENT_RUN_ID,
        "hypothesis": HYPOTHESIS,
        "mutation_tag": MUTATION_TAG,
        "objective": OBJECTIVE,
        "repo_root": str(ROOT),
        "run_dir": str(RUN_DIR),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "timestamp": now_ts(),
        **payload,
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"run_result_json: {json.dumps(result, sort_keys=True)}", flush=True)
    return result


def run_logged(cmd: list[str], *, cwd: Path | None = None) -> int:
    ensure_run_dir()
    with COMMAND_LOG.open("a", encoding="utf-8") as log:
        log.write(f"$ {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=cwd or ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log.write(f"[exit={proc.returncode}]\n")
        return proc.returncode


def configure_if_needed() -> None:
    if (BUILD_DIR / "build.ninja").exists():
        return
    emit_event("configure_start", build_dir=str(BUILD_DIR))
    rc = run_logged(["cmake", "-S", str(ROOT), "-B", str(BUILD_DIR), "-G", "Ninja"])
    if rc != 0:
        write_result(
            "failed",
            failure_kind="cmake_configure",
            exit_code=rc,
            command_log=str(COMMAND_LOG),
        )
        raise SystemExit(rc)
    emit_event("configure_done", build_dir=str(BUILD_DIR))


def check_preconditions() -> list[str]:
    missing: list[str] = []
    required_paths = [
        CPPFRONT_REPO,
        CPPFRONT_SOURCE,
        CPPFRONT_INCLUDE,
        CPPFRONT_REGRESSION,
    ]
    for path in required_paths:
        if not path.exists():
            missing.append(str(path))
    return missing


def bootstrap_cppfront_oracle() -> list[str]:
    """
    Materialize only the missing oracle surfaces needed by conveyor.
    Leaves repo-owned source intact and fills in upstream include/tests.
    """
    materialized: list[str] = []
    wanted = []
    if not CPPFRONT_INCLUDE.exists():
        wanted.append("include")
    if not CPPFRONT_REGRESSION.exists():
        wanted.append("regression-tests")
    if not wanted:
        return materialized

    emit_event("bootstrap_start", wanted=wanted, upstream=CPPFRONT_UPSTREAM)
    with tempfile.TemporaryDirectory(prefix="cppfront-bootstrap-") as tmp:
        tmp_path = Path(tmp)
        clone_dir = tmp_path / "cppfront"
        rc = run_logged(
            [
                "git",
                "clone",
                "--depth=1",
                "--filter=blob:none",
                "--sparse",
                CPPFRONT_UPSTREAM,
                str(clone_dir),
            ]
        )
        if rc != 0:
            write_result(
                "failed",
                failure_kind="bootstrap_clone",
                exit_code=rc,
                command_log=str(COMMAND_LOG),
            )
            raise SystemExit(rc)
        rc = run_logged(
            ["git", "sparse-checkout", "set", "include", "regression-tests"],
            cwd=clone_dir,
        )
        if rc != 0:
            write_result(
                "failed",
                failure_kind="bootstrap_sparse_checkout",
                exit_code=rc,
                command_log=str(COMMAND_LOG),
            )
            raise SystemExit(rc)

        if "include" in wanted:
            shutil.copytree(clone_dir / "include", CPPFRONT_INCLUDE, dirs_exist_ok=True)
            materialized.append(str(CPPFRONT_INCLUDE))
        if "regression-tests" in wanted:
            shutil.copytree(
                clone_dir / "regression-tests",
                CPPFRONT_REGRESSION,
                dirs_exist_ok=True,
            )
            materialized.append(str(CPPFRONT_REGRESSION))

    emit_event("bootstrap_done", materialized=materialized)
    return materialized


def read_targets() -> set[str]:
    proc = subprocess.run(
        ["ninja", "-C", str(BUILD_DIR), "-t", "targets", "all"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    ensure_run_dir()
    with COMMAND_LOG.open("a", encoding="utf-8") as log:
        log.write("$ ninja -C build -t targets all\n")
        log.write(proc.stdout)
        log.write(f"[exit={proc.returncode}]\n")
    if proc.returncode != 0:
        return set()
    targets: set[str] = set()
    for line in proc.stdout.splitlines():
        name = line.split(":", 1)[0].strip()
        if name:
            targets.add(name)
    return targets


def build_target(target: str) -> int:
    emit_event("build_start", target=target)
    rc = run_logged(["ninja", "-C", str(BUILD_DIR), target])
    emit_event("build_done", target=target, exit_code=rc)
    return rc


def run_conveyor_binary(args: argparse.Namespace, *, force_allow_dirty: bool = False) -> int:
    conveyor_bin = BUILD_DIR / "bin" / "cppfront_conveyor"
    if not conveyor_bin.exists():
        write_result(
            "failed",
            failure_kind="missing_conveyor_binary",
            path=str(conveyor_bin),
            command_log=str(COMMAND_LOG),
        )
        return 2
    cmd = [str(conveyor_bin)]
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.skip_ctest:
        cmd.append("--skip-ctest")
    if args.skip_scoring:
        cmd.append("--skip-scoring")
    if args.skip_mappings:
        cmd.append("--skip-mappings")
    if args.allow_dirty_cppfront or force_allow_dirty:
        cmd.append("--allow-dirty-cppfront")
    if args.dry_run:
        cmd.append("--dry-run")
    emit_event("conveyor_start", command=cmd)
    rc = run_logged(cmd)
    emit_event("conveyor_done", exit_code=rc)
    return rc


def snapshot_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


SUMMARY_LINE_RE = re.compile(r"^- (?P<key>[^:]+): (?P<value>.+)$")


def parse_value(value: str) -> object:
    value = value.strip()
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    if value.startswith("`") and value.endswith("`"):
        return value[1:-1]
    return value


def parse_summary(summary_path: Path) -> dict[str, object]:
    metrics: dict[str, object] = {}
    if not summary_path.exists():
        return metrics
    for line in summary_path.read_text(encoding="utf-8").splitlines():
        match = SUMMARY_LINE_RE.match(line.strip())
        if not match:
            continue
        key = (
            match.group("key")
            .lower()
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
            .replace("-", "_")
            .replace(" ", "_")
        )
        metrics[key] = parse_value(match.group("value"))
    return metrics


def count_lines(path: Path) -> int | None:
    if not path.exists():
        return None
    return len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one bounded cppfort research evaluation.")
    parser.add_argument("--limit", type=int, default=None, help="Limit corpus inputs.")
    parser.add_argument("--skip-ctest", action="store_true", help="Pass through to cppfront_conveyor.")
    parser.add_argument("--skip-scoring", action="store_true", help="Pass through to cppfront_conveyor.")
    parser.add_argument("--skip-mappings", action="store_true", help="Pass through to cppfront_conveyor.")
    parser.add_argument("--allow-dirty-cppfront", action="store_true", help="Pass through to cppfront_conveyor.")
    parser.add_argument("--dry-run", action="store_true", help="Pass through to cppfront_conveyor.")
    args = parser.parse_args()

    ensure_run_dir()
    emit_event(
        "run_start",
        controller={
            "parent_run_id": PARENT_RUN_ID,
            "hypothesis": HYPOTHESIS,
            "mutation_tag": MUTATION_TAG,
        },
        run_dir=str(RUN_DIR),
    )

    materialized = bootstrap_cppfront_oracle()
    missing_preconditions = check_preconditions()
    if missing_preconditions:
        write_result(
            "failed",
            failure_kind="missing_precondition",
            missing_paths=missing_preconditions,
            expected_preconditions=[
                str(CPPFRONT_REPO),
                str(CPPFRONT_REGRESSION),
            ],
            materialized=materialized,
        )
        return 2

    configure_if_needed()
    targets = read_targets()
    has_conveyor_target = "conveyor" in targets
    has_conveyor_binary_target = "cppfront_conveyor" in targets
    emit_event(
        "target_scan",
        has_conveyor_target=has_conveyor_target,
        has_conveyor_binary_target=has_conveyor_binary_target,
    )

    if has_conveyor_target:
        rc = build_target("conveyor")
    else:
        if not has_conveyor_binary_target:
            write_result(
                "failed",
                failure_kind="missing_targets",
                has_conveyor_target=has_conveyor_target,
                has_conveyor_binary_target=has_conveyor_binary_target,
                command_log=str(COMMAND_LOG),
            )
            return 2
        rc = build_target("cppfront_conveyor")
        if rc == 0:
            rc = run_conveyor_binary(args, force_allow_dirty=bool(materialized))

    summary_path = CONVEYOR_ROOT / "CONVEYOR_SUMMARY.md"
    cppfront_failures = CONVEYOR_ROOT / "cppfront_failures.txt"
    cppfort_failures = CONVEYOR_ROOT / "cppfort_failures.txt"
    snapshot_if_exists(summary_path, SUMMARY_SNAPSHOT)
    snapshot_if_exists(cppfront_failures, CPPFRONT_FAILURES_SNAPSHOT)
    snapshot_if_exists(cppfort_failures, CPPFORT_FAILURES_SNAPSHOT)

    metrics = parse_summary(summary_path)
    metrics["cppfront_failure_list_count"] = count_lines(cppfront_failures)
    metrics["cppfort_failure_list_count"] = count_lines(cppfort_failures)
    metrics["has_conveyor_target"] = has_conveyor_target
    metrics["has_conveyor_binary_target"] = has_conveyor_binary_target

    status = "ok" if rc == 0 else "failed"
    payload = {
        "exit_code": rc,
        "command_log": str(COMMAND_LOG),
        "artifacts": {
            "summary": str(SUMMARY_SNAPSHOT if SUMMARY_SNAPSHOT.exists() else summary_path),
            "cppfront_failures": str(
                CPPFRONT_FAILURES_SNAPSHOT if CPPFRONT_FAILURES_SNAPSHOT.exists() else cppfront_failures
            ),
            "cppfort_failures": str(
                CPPFORT_FAILURES_SNAPSHOT if CPPFORT_FAILURES_SNAPSHOT.exists() else cppfort_failures
            ),
        },
        "metrics": metrics,
        "controller": {
            "parent_run_id": PARENT_RUN_ID,
            "hypothesis": HYPOTHESIS,
            "mutation_tag": MUTATION_TAG,
        },
        "materialized_preconditions": materialized,
    }
    if rc != 0:
        payload["failure_kind"] = "conveyor" if summary_path.exists() else "build_or_target"

    write_result(status, **payload)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
