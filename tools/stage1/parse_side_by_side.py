#!/usr/bin/env python3
"""
Parser for side-by-side regression logs to extract comparative metrics and graph-node signals.

Usage:
  ./parse_side_by_side.py path/to/side_by_side_log.txt

Outputs:
  - side_by_side_summary.json (extended regression data)
  - graph_nodes.json (inductive model priming data)

Graph nodes capture:
- Stage equivalence signals (transpile, compile, run, output)
- Binary metrics (size, disassembly features)
- Error patterns and contextual features
"""
from __future__ import annotations

import json
import csv
import re
import sys
from pathlib import Path
from typing import List, Dict, Any

TEST_HEADER_RE = re.compile(r"^Testing\s+(?P<file>.+\.cpp2)\s*$")
METRIC_RE = re.compile(r"^\s+(?P<stage>Stage[01])\s+(?P<metric>.+?):\s*(?P<value>.+)$")

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
                "stage0": {
                    "transpile": None,
                    "compile": None,
                    "run": None,
                    "binary_size": 0,
                    "disasm_lines": 0,
                    "output": "",
                },
                "stage1": {
                    "transpile": None,
                    "compile": None,
                    "run": None,
                    "binary_size": 0,
                    "disasm_lines": 0,
                    "output": "",
                },
                "comparison": {
                    "cpp_diff": None,
                    "output_diff": None,
                    "binary_size_diff": 0,
                    "disasm_lines_diff": 0,
                },
                "errors": [],
                "raw": [],
            }
            i += 1
            continue

        if cur is None:
            i += 1
            continue

        line = lines[i]
        cur["raw"].append(line)

        # Status lines
        if "Stage0 transpile OK" in line:
            cur["stage0"]["transpile"] = "OK"
        elif "Stage0 transpile FAILED" in line:
            cur["stage0"]["transpile"] = "FAILED"
        elif "Stage1 transpile OK" in line:
            cur["stage1"]["transpile"] = "OK"
        elif "Stage1 transpile FAILED" in line:
            cur["stage1"]["transpile"] = "FAILED"
        elif "Stage0 compile OK" in line:
            cur["stage0"]["compile"] = "OK"
        elif "Stage0 compile FAILED" in line:
            cur["stage0"]["compile"] = "FAILED"
        elif "Stage1 compile OK" in line:
            cur["stage1"]["compile"] = "OK"
        elif "Stage1 compile FAILED" in line:
            cur["stage1"]["compile"] = "FAILED"
        elif "Stage0 run OK" in line:
            cur["stage0"]["run"] = "OK"
        elif "Stage0 run FAILED" in line:
            cur["stage0"]["run"] = "FAILED"
        elif "Stage1 run OK" in line:
            cur["stage1"]["run"] = "OK"
        elif "Stage1 run FAILED" in line:
            cur["stage1"]["run"] = "FAILED"

        # Metric lines
        mm = METRIC_RE.match(line)
        if mm:
            stage = mm.group("stage").lower()
            metric = mm.group("metric").replace(" ", "_").lower()
            value_str = mm.group("value")
            try:
                if "size" in metric:
                    value = int(value_str.split()[0])
                elif "lines" in metric:
                    value = int(value_str)
                else:
                    value = value_str
                cur[stage][metric] = value
            except ValueError:
                pass

        # Comparison signals
        if "C++ outputs identical" in line:
            cur["comparison"]["cpp_diff"] = "identical"
        elif "C++ outputs differ" in line:
            cur["comparison"]["cpp_diff"] = "differ"
        elif "Outputs identical" in line:
            cur["comparison"]["output_diff"] = "identical"
        elif "Outputs differ" in line:
            cur["comparison"]["output_diff"] = "differ"

        i += 1

    if cur:
        entries.append(cur)

    # Compute diffs
    for e in entries:
        e["comparison"]["binary_size_diff"] = e["stage1"]["binary_size"] - e["stage0"]["binary_size"]
        e["comparison"]["disasm_lines_diff"] = e["stage1"]["disasm_lines"] - e["stage0"]["disasm_lines"]

    return entries

def extract_graph_nodes(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract graph-node signals for inductive model priming."""
    nodes = []
    for e in entries:
        tf = e["test_file"]
        s0 = e["stage0"]
        s1 = e["stage1"]
        cmp = e["comparison"]

        # Equivalence signals
        transpile_equiv = (s0["transpile"] == s1["transpile"] == "OK")
        compile_equiv = (s0["compile"] == s1["compile"] == "OK")
        run_equiv = (s0["run"] == s1["run"] == "OK")
        output_equiv = (cmp["output_diff"] == "identical")

        # Binary metrics
        size_ratio = s1["binary_size"] / max(s0["binary_size"], 1)
        disasm_ratio = s1["disasm_lines"] / max(s0["disasm_lines"], 1)

        node = {
            "test_file": tf,
            "signals": {
                "transpile_equiv": transpile_equiv,
                "compile_equiv": compile_equiv,
                "run_equiv": run_equiv,
                "output_equiv": output_equiv,
                "semantic_equiv": transpile_equiv and compile_equiv and run_equiv and output_equiv,
            },
            "metrics": {
                "stage0_binary_size": s0["binary_size"],
                "stage1_binary_size": s1["binary_size"],
                "binary_size_ratio": size_ratio,
                "stage0_disasm_lines": s0["disasm_lines"],
                "stage1_disasm_lines": s1["disasm_lines"],
                "disasm_ratio": disasm_ratio,
            },
            "features": {
                "cpp_diff_type": cmp["cpp_diff"],
                "output_diff_type": cmp["output_diff"],
                "stage0_success": s0["transpile"] == "OK" and s0["compile"] == "OK" and s0["run"] == "OK",
                "stage1_success": s1["transpile"] == "OK" and s1["compile"] == "OK" and s1["run"] == "OK",
            }
        }
        nodes.append(node)

    return nodes

def write_outputs(entries: List[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Summary JSON
    summary = {
        "entries": entries,
        "summary": {
            "total_tests": len(entries),
            "stage0_success": sum(1 for e in entries if e["stage0"]["transpile"] == "OK" and e["stage0"]["compile"] == "OK"),
            "stage1_success": sum(1 for e in entries if e["stage1"]["transpile"] == "OK" and e["stage1"]["compile"] == "OK"),
            "both_success": sum(1 for e in entries if (
                e["stage0"]["transpile"] == "OK" and e["stage0"]["compile"] == "OK" and
                e["stage1"]["transpile"] == "OK" and e["stage1"]["compile"] == "OK"
            )),
            "semantic_equiv": sum(1 for e in entries if (
                e["comparison"]["cpp_diff"] == "identical" and
                e["comparison"]["output_diff"] == "identical"
            )),
        }
    }

    json_path = out_dir / "side_by_side_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Graph nodes JSON
    nodes = extract_graph_nodes(entries)
    nodes_path = out_dir / "graph_nodes.json"
    nodes_path.write_text(json.dumps({"nodes": nodes}, indent=2), encoding="utf-8")

    # CSV for analysis
    csv_path = out_dir / "side_by_side_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow([
            "test_file",
            "stage0_transpile", "stage0_compile", "stage0_run", "stage0_binary_size", "stage0_disasm_lines",
            "stage1_transpile", "stage1_compile", "stage1_run", "stage1_binary_size", "stage1_disasm_lines",
            "cpp_diff", "output_diff", "binary_size_diff", "disasm_lines_diff"
        ])
        for e in entries:
            writer.writerow([
                e["test_file"],
                e["stage0"]["transpile"], e["stage0"]["compile"], e["stage0"]["run"], e["stage0"]["binary_size"], e["stage0"]["disasm_lines"],
                e["stage1"]["transpile"], e["stage1"]["compile"], e["stage1"]["run"], e["stage1"]["binary_size"], e["stage1"]["disasm_lines"],
                e["comparison"]["cpp_diff"], e["comparison"]["output_diff"], e["comparison"]["binary_size_diff"], e["comparison"]["disasm_lines_diff"]
            ])

def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print(f"Usage: {argv[0]} path/to/side_by_side_log.txt", file=sys.stderr)
        return 1

    log_path = Path(argv[1])
    if not log_path.exists():
        print(f"Log file not found: {log_path}", file=sys.stderr)
        return 1

    text = log_path.read_text(encoding="utf-8")
    entries = parse_log(text)

    out_dir = Path("tools/stage1")
    write_outputs(entries, out_dir)

    print(f"Parsed {len(entries)} test entries")
    print(f"Outputs written to {out_dir}")

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))