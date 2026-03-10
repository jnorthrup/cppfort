#!/usr/bin/env python3
"""Batch process corpus files to emit mapping candidates.

Usage:
  python batch_emit_mappings.py -i corpus/inputs -o mappings_output --limit 10
"""

import sys
import os
import json
import subprocess
from pathlib import Path
import argparse


def find_cpp_files(input_dir: Path, limit: int = None):
    """Find all .cpp and .cpp2 files."""
    cpp_files = list(input_dir.glob("*.cpp")) + list(input_dir.glob("*.cpp2"))
    cpp_files = sorted(cpp_files)
    if limit:
        cpp_files = cpp_files[:limit]
    return cpp_files


def process_file(input_file: Path, output_dir: Path, script_dir: Path):
    """Process a single file to emit mappings."""
    output_file = output_dir / f"{input_file.stem}_mappings.json"
    
    cmd = [
        str(script_dir / "run_inference.sh"),
        str(script_dir / "emit_mappings.py"),
        "-i", str(input_file),
        "-o", str(output_file),
        "--", "-std=c++20"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return True, output_file, None
        else:
            return False, None, result.stderr
    except subprocess.TimeoutExpired:
        return False, None, "Timeout"
    except Exception as e:
        return False, None, str(e)


def aggregate_mappings(output_dir: Path):
    """Aggregate all mappings into a single file."""
    all_mappings = []
    mapping_files = list(output_dir.glob("*_mappings.json"))
    
    for mf in mapping_files:
        with open(mf, "r") as f:
            data = json.load(f)
            all_mappings.extend(data.get("mappings", []))
    
    # Deduplicate by stable semantic + grammar signatures and accumulate support.
    unique_mappings = []
    seen = {}
    for m in all_mappings:
        key = (
            m.get("ast_kind"),
            m.get("semantic_signature"),
            m.get("grammar_fingerprint"),
        )
        source_file = m.get("source_sample", {}).get("file")
        if key not in seen:
            merged = dict(m)
            merged["support_count"] = 1
            merged["supporting_files"] = [source_file] if source_file else []
            unique_mappings.append(merged)
            seen[key] = merged
            continue

        merged = seen[key]
        merged["support_count"] += 1
        if source_file and source_file not in merged["supporting_files"]:
            merged["supporting_files"].append(source_file)
        merged["confidence"] = max(merged.get("confidence", 0.0), m.get("confidence", 0.0))
        for example in m.get("examples", []):
            if example not in merged["examples"]:
                merged["examples"].append(example)

    unique_mappings.sort(
        key=lambda m: (
            m.get("ast_kind", ""),
            m.get("semantic_signature", ""),
            m.get("grammar_fingerprint", ""),
        )
    )
    return unique_mappings


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", required=True, help="Input directory with .cpp/.cpp2 files")
    parser.add_argument("-o", "--output-dir", required=True, help="Output directory for mapping files")
    parser.add_argument("--limit", type=int, help="Limit number of files to process")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate all mappings into one file")
    args = parser.parse_args(argv)
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    script_dir = Path(__file__).parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cpp_files = find_cpp_files(input_dir, args.limit)
    print(f"Found {len(cpp_files)} files to process")
    
    success_count = 0
    failed = []
    
    for i, cf in enumerate(cpp_files, 1):
        print(f"[{i}/{len(cpp_files)}] Processing {cf.name}...", end=" ")
        success, output_file, error = process_file(cf, output_dir, script_dir)
        if success:
            print(f"✓ -> {output_file.name}")
            success_count += 1
        else:
            print(f"✗ ({error})")
            failed.append((cf.name, error))
    
    print(f"\nProcessed {success_count}/{len(cpp_files)} files successfully")
    
    if args.aggregate:
        print("\nAggregating mappings...")
        unique = aggregate_mappings(output_dir)
        agg_file = output_dir / "aggregated_mappings.json"
        with open(agg_file, "w") as f:
            json.dump({
                "schema_version": "1.0",
                "spec": "docs/MAPPING_SPEC.md",
                "total_mappings": len(unique),
                "mappings": unique
            }, f, indent=2)
        print(f"Wrote {len(unique)} unique mappings to {agg_file}")
    
    if failed:
        print(f"\nFailed files ({len(failed)}):")
        for fname, err in failed[:10]:
            print(f"  - {fname}: {err}")


if __name__ == "__main__":
    main(sys.argv[1:])
