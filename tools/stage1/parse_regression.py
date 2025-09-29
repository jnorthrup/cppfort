#!/usr/bin/env python3
"""
Simple parser for `regression-tests/regression_log.txt` to extract per-test
results and error messages. Produces JSON and CSV outputs that Stage1 can
consume as training / analysis data for inductive learning.

Usage:
  ./parse_regression.py [path/to/regression_log.txt]

Outputs:
  - regression_summary.json
  - regression_summary.csv

This is intentionally small and dependency-free so it can be extended by
Stage1 components (pattern extraction, AST linking, auto-suggested fixes).
"""
from __future__ import annotations

import json
import csv
import re
import sys
from pathlib import Path
from typing import List, Dict, Any


TEST_HEADER_RE = re.compile(r"^Testing\s+(?P<file>.+\.cpp2)\s*$")
ERROR_LINE_RE = re.compile(r"^(?P<file>[^\(]+)\((?P<line>\d+),(?P<col>\d+)\):\s*error:\s*(?P<msg>.+)$")


def parse_log(text: str) -> List[Dict[str, Any]]:
    lines = text.splitlines()
    entries: List[Dict[str, Any]] = []
    cur = None

    i = 0
    while i < len(lines):
        m = TEST_HEADER_RE.match(lines[i])
        if m:
            if cur:
                entries.append(cur)
            cur = {
                "test_file": m.group("file").strip(),
                "raw": [],
                "transpile": None,
                "compile": None,
                "run": None,
                "errors": [],
            }
            i += 1
            continue

        if cur is None:
            i += 1
            continue

        line = lines[i]
        cur["raw"].append(line)

        # Status lines
        if line.strip().startswith("Transpile OK"):
            cur["transpile"] = "OK"
        elif line.strip().startswith("Transpile FAILED"):
            cur["transpile"] = "FAILED"
        elif line.strip().startswith("Compile OK"):
            cur["compile"] = "OK"
        elif line.strip().startswith("Compile FAILED"):
            cur["compile"] = "FAILED"
        elif line.strip().startswith("Run OK"):
            cur["run"] = "OK"
        elif line.strip().startswith("Run FAILED"):
            cur["run"] = "FAILED"

        # Error lines that match file(line,col): error: msg
        em = ERROR_LINE_RE.match(line.strip())
        if em:
            cur["errors"].append({
                "file": em.group("file").strip(),
                "line": int(em.group("line")),
                "col": int(em.group("col")),
                "message": em.group("msg").strip(),
            })

        i += 1

    if cur:
        entries.append(cur)

    return entries


def summarize(entries: List[Dict[str, Any]]) -> Dict[str, int]:
    total = len(entries)
    fails = 0
    by_type = {"transpile_failed": 0, "compile_failed": 0, "run_failed": 0}
    for e in entries:
        if e.get("transpile") == "FAILED":
            by_type["transpile_failed"] += 1
            fails += 1
        if e.get("compile") == "FAILED":
            by_type["compile_failed"] += 1
            fails += 1
        if e.get("run") == "FAILED":
            by_type["run_failed"] += 1
            fails += 1
    return {"total_tests": total, "total_failures_reported": fails, **by_type}


def write_outputs(entries: List[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "regression_summary.json"
    csv_path = out_dir / "regression_summary.csv"

    with json_path.open("w", encoding="utf-8") as jf:
        json.dump({"entries": entries, "summary": summarize(entries)}, jf, indent=2)

    # CSV: test_file,transpile,compile,run,num_errors
    with csv_path.open("w", encoding="utf-8", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["test_file", "transpile", "compile", "run", "num_errors"])
        for e in entries:
            writer.writerow([
                e.get("test_file"),
                e.get("transpile"),
                e.get("compile"),
                e.get("run"),
                len(e.get("errors", [])),
            ])

    print(f"Wrote: {json_path}\nWrote: {csv_path}")


def main(argv: List[str]) -> int:
    if len(argv) >= 2:
        p = Path(argv[1])
    else:
        p = Path(__file__).resolve().parents[2] / "regression-tests" / "regression_log.txt"

    if not p.exists():
        print(f"Regression log not found: {p}")
        return 2

    text = p.read_text(encoding="utf-8")
    entries = parse_log(text)
    out_dir = Path(__file__).resolve().parent
    write_outputs(entries, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
