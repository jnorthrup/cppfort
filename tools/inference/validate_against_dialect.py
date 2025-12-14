#!/usr/bin/env python3
"""Validate mapping candidates against Cpp2Dialect.td definitions.

Usage:
  python validate_against_dialect.py -m mappings.json -d ../../include/Cpp2Dialect.td
"""

import sys
import json
import re
from pathlib import Path
from typing import Dict, Set, List
from collections import defaultdict
import argparse


def parse_dialect_ops(td_file: Path) -> Dict[str, Dict]:
    """Extract op definitions from TableGen dialect file."""
    ops = {}
    content = td_file.read_text()
    
    # Pattern: def OpName : Cpp2_Op<"mnemonic", ...>
    op_pattern = re.compile(
        r'def\s+(\w+)\s*:\s*Cpp2_Op<"([^"]+)"[^{]*\{([^}]*)\}',
        re.DOTALL
    )
    
    for match in op_pattern.finditer(content):
        op_name, mnemonic, body = match.groups()
        
        # Extract summary
        summary_match = re.search(r'let\s+summary\s*=\s*"([^"]+)"', body)
        summary = summary_match.group(1) if summary_match else ""
        
        ops[mnemonic] = {
            "op_name": op_name,
            "mnemonic": mnemonic,
            "summary": summary,
            "definition": match.group(0)
        }
    
    return ops


def extract_mlir_op_from_template(template: str) -> str:
    """Extract MLIR op mnemonic from template string.
    
    Examples:
      "cpp2.func @{name}" -> "func"
      "cpp2.if %cond { ... }" -> "if"
      "cpp2.return %value" -> "return"
    """
    match = re.match(r'cpp2\.(\w+)', template)
    return match.group(1) if match else None


def validate_mappings(mappings: List[Dict], dialect_ops: Dict[str, Dict]) -> Dict:
    """Validate mappings against dialect ops."""
    results = {
        "total_mappings": len(mappings),
        "valid": [],
        "missing_ops": [],
        "unmatched_mappings": [],
        "coverage": {},
    }
    
    # Track which dialect ops are covered
    dialect_coverage = {op: 0 for op in dialect_ops.keys()}
    
    for m in mappings:
        mlir_template = m.get("mlir_template", "")
        op_mnemonic = extract_mlir_op_from_template(mlir_template)
        
        if not op_mnemonic:
            results["unmatched_mappings"].append({
                "id": m.get("id"),
                "template": mlir_template,
                "reason": "Could not extract op mnemonic"
            })
            continue
        
        if op_mnemonic in dialect_ops:
            results["valid"].append({
                "mapping_id": m.get("id"),
                "ast_kind": m.get("ast_kind"),
                "op_mnemonic": op_mnemonic,
                "dialect_op": dialect_ops[op_mnemonic]["op_name"],
                "confidence": m.get("confidence")
            })
            dialect_coverage[op_mnemonic] += 1
        else:
            results["missing_ops"].append({
                "mapping_id": m.get("id"),
                "ast_kind": m.get("ast_kind"),
                "op_mnemonic": op_mnemonic,
                "template": mlir_template
            })
    
    results["coverage"] = {
        op: {"count": count, "summary": info["summary"]}
        for op, count in dialect_coverage.items()
        for info in [dialect_ops[op]]
    }
    
    return results


def print_validation_report(results: Dict):
    """Pretty-print validation results."""
    print("=" * 70)
    print("MAPPING VALIDATION REPORT")
    print("=" * 70)
    
    print(f"\nTotal Mappings: {results['total_mappings']}")
    print(f"Valid Mappings: {len(results['valid'])}")
    print(f"Missing Ops: {len(results['missing_ops'])}")
    print(f"Unmatched: {len(results['unmatched_mappings'])}")
    
    print("\n" + "=" * 70)
    print("DIALECT COVERAGE")
    print("=" * 70)
    for op, data in sorted(results["coverage"].items(), key=lambda x: -x[1]["count"]):
        status = "✓" if data["count"] > 0 else "✗"
        print(f"{status} {op:20} {data['count']:4} mappings | {data['summary']}")
    
    if results["missing_ops"]:
        print("\n" + "=" * 70)
        print("MISSING DIALECT OPS (need to be added to Cpp2Dialect.td)")
        print("=" * 70)
        missing_by_op = defaultdict(list)
        for m in results["missing_ops"]:
            missing_by_op[m["op_mnemonic"]].append(m)
        
        for op, mappings in sorted(missing_by_op.items()):
            print(f"\n{op} (referenced by {len(mappings)} mappings):")
            ast_kinds = {m["ast_kind"] for m in mappings}
            print(f"  AST kinds: {', '.join(sorted(ast_kinds))}")
            print(f"  Example template: {mappings[0]['template']}")
    
    if results["unmatched_mappings"]:
        print("\n" + "=" * 70)
        print("UNMATCHED MAPPINGS (invalid templates)")
        print("=" * 70)
        for m in results["unmatched_mappings"][:10]:
            print(f"  {m['id']}: {m['template']}")
            print(f"    Reason: {m['reason']}")


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mappings", required=True, help="Mappings JSON file")
    parser.add_argument("-d", "--dialect", required=True, help="Cpp2Dialect.td file")
    parser.add_argument("-o", "--output", help="Output JSON file for results")
    args = parser.parse_args(argv)
    
    mappings_file = Path(args.mappings)
    dialect_file = Path(args.dialect)
    
    if not mappings_file.exists():
        print(f"ERROR: Mappings file not found: {mappings_file}")
        return 1
    
    if not dialect_file.exists():
        print(f"ERROR: Dialect file not found: {dialect_file}")
        return 1
    
    # Load mappings
    with open(mappings_file) as f:
        data = json.load(f)
        mappings = data.get("mappings", [])
    
    # Parse dialect
    dialect_ops = parse_dialect_ops(dialect_file)
    print(f"Found {len(dialect_ops)} ops in {dialect_file.name}")
    
    # Validate
    results = validate_mappings(mappings, dialect_ops)
    
    # Report
    print_validation_report(results)
    
    # Save results if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved detailed results to {args.output}")
    
    # Exit code: 0 if all valid, 1 if any missing
    return 0 if not results["missing_ops"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
