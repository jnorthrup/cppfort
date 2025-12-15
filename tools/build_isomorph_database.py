#!/usr/bin/env python3
"""
Build corpus-wide isomorph database with deduplication.

Aggregates all tagged isomorphs from the corpus, deduplicates by signature,
tracks occurrence counts, and builds a comprehensive database.

Usage:
    ./build_isomorph_database.py --tagged-dir corpus/tagged --output database.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class IsomorphEntry:
    """Corpus-wide isomorph entry with aggregated data."""
    id: str
    signature: str
    occurrence_count: int
    files: List[str]
    ast_pattern: Dict
    mlir_region: Dict
    confidence: float
    examples: List[Dict]

    def to_dict(self):
        return asdict(self)


class IsomorphDatabaseBuilder:
    """Builds deduplicated corpus-wide isomorph database."""

    def __init__(self):
        self.signatures: Dict[str, IsomorphEntry] = {}
        self.total_files = 0
        self.total_isomorphs = 0

    def add_file(self, tagged_file: Path):
        """Add all isomorphs from a tagged file to the database."""
        try:
            data = json.loads(tagged_file.read_text())
        except Exception as e:
            print(f"Warning: Failed to read {tagged_file.name}: {e}")
            return

        self.total_files += 1
        source_file = tagged_file.stem.replace(".tagged", "")

        tagged_isomorphs = data.get("tagged_isomorphs", [])
        self.total_isomorphs += len(tagged_isomorphs)

        for isomorph in tagged_isomorphs:
            self._add_isomorph(isomorph, source_file)

    def _add_isomorph(self, isomorph: Dict, source_file: str):
        """Add or merge an isomorph into the database."""
        ast_pattern = isomorph.get("ast_pattern", {})
        signature = ast_pattern.get("signature", "")

        if not signature:
            return

        if signature in self.signatures:
            # Merge with existing entry
            entry = self.signatures[signature]
            entry.occurrence_count += 1
            if source_file not in entry.files:
                entry.files.append(source_file)
            entry.examples.extend(isomorph.get("examples", []))
        else:
            # Create new entry
            self.signatures[signature] = IsomorphEntry(
                id=isomorph.get("isomorph_id", "unknown"),
                signature=signature,
                occurrence_count=1,
                files=[source_file],
                ast_pattern=ast_pattern,
                mlir_region=isomorph.get("mlir_region", {}),
                confidence=isomorph.get("confidence", 0.0),
                examples=isomorph.get("examples", [])
            )

    def build(self) -> Dict:
        """Build the final database structure."""
        # Sort by occurrence count (most common first)
        sorted_entries = sorted(
            self.signatures.values(),
            key=lambda e: e.occurrence_count,
            reverse=True
        )

        # Assign IDs based on frequency
        for i, entry in enumerate(sorted_entries):
            entry.id = f"iso_{i:06d}"

        return {
            "version": "1.0",
            "corpus_size": self.total_files,
            "total_isomorphs": self.total_isomorphs,
            "unique_patterns": len(self.signatures),
            "isomorphs": [e.to_dict() for e in sorted_entries]
        }

    def generate_summary(self) -> str:
        """Generate human-readable summary statistics."""
        summary = []
        summary.append("=== Isomorph Database Summary ===\n")
        summary.append(f"Total files processed: {self.total_files}")
        summary.append(f"Total isomorphs extracted: {self.total_isomorphs}")
        summary.append(f"Unique patterns (deduplicated): {len(self.signatures)}")
        summary.append(f"Deduplication ratio: {len(self.signatures) / self.total_isomorphs * 100:.1f}%\n")

        # Top 10 most common patterns
        top_patterns = sorted(
            self.signatures.values(),
            key=lambda e: e.occurrence_count,
            reverse=True
        )[:10]

        summary.append("Top 10 most common patterns:")
        for i, entry in enumerate(top_patterns, 1):
            root_kind = entry.ast_pattern.get("root_kind", "Unknown")
            region_type = entry.mlir_region.get("region_type", "unknown")
            summary.append(
                f"  {i}. {root_kind} → {region_type}: "
                f"{entry.occurrence_count} occurrences across {len(entry.files)} files"
            )

        summary.append("\nBy MLIR region type:")
        by_region = defaultdict(int)
        for entry in self.signatures.values():
            region_type = entry.mlir_region.get("region_type", "unknown")
            by_region[region_type] += entry.occurrence_count

        for region_type, count in sorted(by_region.items(), key=lambda x: -x[1]):
            summary.append(f"  {region_type}: {count}")

        summary.append("\nBy AST root kind:")
        by_kind = defaultdict(int)
        for entry in self.signatures.values():
            root_kind = entry.ast_pattern.get("root_kind", "Unknown")
            by_kind[root_kind] += entry.occurrence_count

        for kind, count in sorted(by_kind.items(), key=lambda x: -x[1])[:15]:
            summary.append(f"  {kind}: {count}")

        return "\n".join(summary)


def main():
    parser = argparse.ArgumentParser(
        description="Build corpus-wide isomorph database with deduplication"
    )
    parser.add_argument(
        "--tagged-dir",
        type=Path,
        required=True,
        help="Directory containing tagged isomorph JSON files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output database JSON file"
    )
    parser.add_argument(
        "--summary",
        type=Path,
        help="Optional output file for summary statistics"
    )

    args = parser.parse_args()

    # Build database
    print(f"Scanning tagged isomorphs in: {args.tagged_dir}")
    builder = IsomorphDatabaseBuilder()

    tagged_files = sorted(args.tagged_dir.glob("*.tagged.json"))
    print(f"Found {len(tagged_files)} tagged files")

    for tagged_file in tagged_files:
        builder.add_file(tagged_file)
        if builder.total_files % 10 == 0:
            print(f"  Processed {builder.total_files}/{len(tagged_files)} files...", end='\r')

    print(f"\n  Processed {builder.total_files}/{len(tagged_files)} files")

    # Generate database
    print("Building deduplicated database...")
    database = builder.build()

    # Write database
    print(f"Writing database to {args.output}")
    args.output.write_text(json.dumps(database, indent=2))

    # Generate and display summary
    summary = builder.generate_summary()
    print(f"\n{summary}")

    # Write summary if requested
    if args.summary:
        print(f"\nWriting summary to {args.summary}")
        args.summary.write_text(summary)


if __name__ == "__main__":
    main()
