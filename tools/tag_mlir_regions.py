#!/usr/bin/env python3
"""
Tag AST isomorphs with MLIR base-dialect region patterns.

Maps canonical AST patterns to normalized MLIR Cpp2Dialect operations
based on structural matching rules.

Usage:
    ./tag_mlir_regions.py --isomorphs input.isomorph.json --output tagged.json
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict


@dataclass
class MLIROperation:
    """Represents an MLIR dialect operation."""
    op: str
    role: str  # entry, data, control, terminator


@dataclass
class MLIRRegion:
    """MLIR region template for an AST pattern."""
    dialect: str
    operations: List[Dict[str, str]]
    region_type: str
    ssa_form: bool


@dataclass
class TaggedIsomorph:
    """AST isomorph tagged with MLIR region mapping."""
    isomorph_id: str
    ast_pattern: Dict
    mlir_region: Dict
    confidence: float
    examples: List[Dict]

    def to_dict(self):
        return asdict(self)


class MLIRRegionTagger:
    """Tags AST isomorphs with MLIR region patterns."""

    # Tagging rules from AST_ISOMORPH_MAPPING_SPEC.md
    TAGGING_RULES = {
        # Function patterns
        "FunctionDecl": {
            "operations": [
                {"op": "cpp2.func", "role": "entry"},
                {"op": "cpp2.return", "role": "terminator"}
            ],
            "region_type": "function_body",
            "confidence": 1.0
        },

        # Control flow patterns
        "IfStmt": {
            "operations": [
                {"op": "cpp2.if", "role": "control"},
                {"op": "cpp2.region", "role": "data"}
            ],
            "region_type": "conditional",
            "confidence": 1.0
        },
        "WhileStmt": {
            "operations": [
                {"op": "cpp2.loop", "role": "control"},
                {"op": "cpp2.if", "role": "control"},
                {"op": "cpp2.break", "role": "terminator"}
            ],
            "region_type": "loop",
            "confidence": 0.9
        },
        "ForStmt": {
            "operations": [
                {"op": "cpp2.loop", "role": "control"}
            ],
            "region_type": "loop",
            "confidence": 0.85
        },
        "DoStmt": {
            "operations": [
                {"op": "cpp2.loop", "role": "control"}
            ],
            "region_type": "loop",
            "confidence": 0.9
        },

        # Statement patterns
        "ReturnStmt": {
            "operations": [
                {"op": "cpp2.return", "role": "terminator"}
            ],
            "region_type": "terminator",
            "confidence": 1.0
        },
        "VarDecl": {
            "operations": [
                {"op": "cpp2.var", "role": "data"}
            ],
            "region_type": "declaration",
            "confidence": 1.0
        },

        # Expression patterns
        "CallExpr": {
            "operations": [
                {"op": "cpp2.call", "role": "data"}
            ],
            "region_type": "expression",
            "confidence": 0.9
        },
        "CXXOperatorCallExpr": {
            "operations": [
                {"op": "cpp2.ufcs_call", "role": "data"}
            ],
            "region_type": "expression",
            "confidence": 0.9
        },
        "BinaryOperator": {
            "operations": [
                {"op": "cpp2.binop", "role": "data"}
            ],
            "region_type": "expression",
            "confidence": 1.0
        },

        # Memory patterns
        "CXXNewExpr": {
            "operations": [
                {"op": "cpp2.new", "role": "data"}
            ],
            "region_type": "allocation",
            "confidence": 1.0
        },
        "CXXDeleteExpr": {
            "operations": [
                {"op": "cpp2.delete", "role": "data"}
            ],
            "region_type": "deallocation",
            "confidence": 1.0
        }
    }

    def __init__(self, isomorphs_data: Dict):
        self.isomorphs = isomorphs_data.get("isomorphs", [])
        self.source_file = isomorphs_data.get("source_file", "unknown")
        self.tagged = []

    def tag_all(self) -> List[TaggedIsomorph]:
        """Tag all isomorphs with MLIR patterns."""
        for isomorph in self.isomorphs:
            tagged = self._tag_isomorph(isomorph)
            if tagged:
                self.tagged.append(tagged)

        return self.tagged

    def _tag_isomorph(self, isomorph: Dict) -> Optional[TaggedIsomorph]:
        """Tag a single isomorph with MLIR region pattern."""
        root_kind = isomorph.get("root_kind")

        if root_kind not in self.TAGGING_RULES:
            # No tagging rule for this pattern
            return None

        rule = self.TAGGING_RULES[root_kind]

        mlir_region = MLIRRegion(
            dialect="cpp2",
            operations=rule["operations"],
            region_type=rule["region_type"],
            ssa_form=True
        )

        ast_pattern = {
            "root_kind": root_kind,
            "structure": isomorph.get("structure", []),
            "signature": isomorph.get("signature", "")
        }

        return TaggedIsomorph(
            isomorph_id=isomorph.get("id", "unknown"),
            ast_pattern=ast_pattern,
            mlir_region=asdict(mlir_region),
            confidence=rule["confidence"],
            examples=isomorph.get("examples", [])
        )

    def export(self) -> Dict:
        """Export tagged isomorphs as JSON-serializable dict."""
        return {
            "version": "1.0",
            "source_file": self.source_file,
            "total_tagged": len(self.tagged),
            "tagged_isomorphs": [t.to_dict() for t in self.tagged]
        }


def main():
    parser = argparse.ArgumentParser(
        description="Tag AST isomorphs with MLIR base-dialect region patterns"
    )
    parser.add_argument(
        "--isomorphs",
        type=Path,
        required=True,
        help="Input isomorph JSON file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output tagged JSON file"
    )

    args = parser.parse_args()

    # Read isomorphs
    print(f"Reading isomorphs: {args.isomorphs}")
    isomorphs_data = json.loads(args.isomorphs.read_text())

    # Tag with MLIR patterns
    print("Tagging with MLIR region patterns...")
    tagger = MLIRRegionTagger(isomorphs_data)
    tagged = tagger.tag_all()

    # Write output
    output_data = tagger.export()
    print(f"Writing {len(tagged)} tagged isomorphs to {args.output}")
    args.output.write_text(json.dumps(output_data, indent=2))

    # Print summary
    print(f"\n=== Tagging Summary ===")
    print(f"Total isomorphs: {isomorphs_data.get('total_isomorphs', 0)}")
    print(f"Tagged patterns: {len(tagged)}")
    print(f"Coverage: {len(tagged) / isomorphs_data.get('total_isomorphs', 1) * 100:.1f}%")

    # Breakdown by region type
    by_region_type = {}
    for t in tagged:
        region_type = t.mlir_region["region_type"]
        by_region_type[region_type] = by_region_type.get(region_type, 0) + 1

    print(f"\nBy region type:")
    for region_type, count in sorted(by_region_type.items(), key=lambda x: -x[1]):
        print(f"  {region_type}: {count}")


if __name__ == "__main__":
    main()
