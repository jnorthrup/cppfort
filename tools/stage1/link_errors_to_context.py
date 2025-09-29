#!/usr/bin/env python3
"""
Link errors from regression_summary.json to source snippets and simple token context.

Usage:
  ./link_errors_to_context.py path/to/regression_summary.json

Outputs:
  - tools/stage1/linked_errors.json
  - tools/stage1/error_analysis.txt (top messages and failing tests)

This script is intentionally lightweight and uses a simple tokenizer to
provide immediate contextual features Stage1 can use for inductive
analysis. Later iterations can call into the actual C++ lexer/AST.
"""
from __future__ import annotations

import json
import sys
import re
from pathlib import Path
from typing import Any, Dict, List


TOKEN_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)|(\d+\.\d+|\d+)|([{}()\[\];,<>.+\-*/%&|^~=!:?])")


def simple_tokenize(line: str) -> List[str]:
    tokens = []
    for m in TOKEN_RE.finditer(line):
        tokens.append(m.group(0))
    return tokens


def load_summary(p: Path) -> List[Dict[str, Any]]:
    j = json.loads(p.read_text(encoding="utf-8"))
    return j.get("entries", [])


def link_errors(summary_path: Path, out_dir: Path) -> None:
    entries = load_summary(summary_path)
    linked = []
    fail_tests = set()
    msg_counts = {}

    for e in entries:
        tf = e.get("test_file")
        # test files are located in regression-tests/ relative to repo root
        candidate = Path("regression-tests") / tf
        if not candidate.exists():
            # sometimes the error file is included with a path; try basename
            candidate = Path("regression-tests") / Path(tf).name

        src_lines = []
        if candidate.exists():
            src_lines = candidate.read_text(encoding="utf-8").splitlines()

        for err in e.get("errors", []):
            msg = err.get("message", "")
            msg_counts[msg] = msg_counts.get(msg, 0) + 1
            fail_tests.add(tf)
            line_no = err.get("line", 1)
            # 1-indexed lines
            idx = max(0, line_no - 1)
            start = max(0, idx - 3)
            end = min(len(src_lines), idx + 4)
            snippet = src_lines[start:end] if src_lines else []
            tokens = simple_tokenize(src_lines[idx]) if src_lines and 0 <= idx < len(src_lines) else []
            linked.append({
                "test_file": tf,
                "error": err,
                "context": {
                    "line_number": line_no,
                    "snippet_start_line": start + 1,
                    "snippet": snippet,
                    "tokens_on_line": tokens,
                },
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    linked_path = out_dir / "linked_errors.json"
    linked_path.write_text(json.dumps({"linked": linked}, indent=2), encoding="utf-8")

    # write simple analysis
    analysis = []
    analysis.append("Top error messages:")
    for msg, cnt in sorted(msg_counts.items(), key=lambda x: -x[1])[:10]:
        analysis.append(f"{cnt:4d}  {msg}")
    analysis.append("")
    analysis.append("Tests with failures:")
    for t in sorted(fail_tests):
        analysis.append(f" - {t}")

    analysis_path = out_dir / "error_analysis.txt"
    analysis_path.write_text("\n".join(analysis), encoding="utf-8")

    print(f"Wrote: {linked_path}\nWrote: {analysis_path}")


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("Usage: link_errors_to_context.py path/to/regression_summary.json")
        return 2
    summary = Path(argv[1])
    if not summary.exists():
        print(f"Summary not found: {summary}")
        return 2
    out_dir = summary.resolve().parent
    link_errors(summary, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
