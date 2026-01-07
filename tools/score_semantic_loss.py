#!/usr/bin/env python3
"""
Score semantic loss between reference and candidate transpiler outputs.

Compares cppfront reference AST against cppfort candidate AST to measure
semantic divergence using structural, type, and operation distances.

Usage:
    ./score_semantic_loss.py --reference ref.tagged.json --candidate cand.isomorph.json --output loss.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class LossMetrics:
    """Semantic loss metrics for a single file."""
    file: str
    structural_distance: float
    type_distance: float
    operation_distance: float
    combined_loss: float
    reference_isomorphs: int
    candidate_isomorphs: int
    matched_patterns: int
    missing_patterns: int
    extra_patterns: int

    def to_dict(self):
        return asdict(self)


class SemanticLossScorer:
    """Computes semantic loss between reference and candidate ASTs."""

    # Weighted combination: 50% structural + 30% type + 20% operation
    WEIGHT_STRUCTURAL = 0.5
    WEIGHT_TYPE = 0.3
    WEIGHT_OPERATION = 0.2

    def __init__(self, reference_data: Dict, candidate_data: Dict):
        self.reference = reference_data
        self.candidate = candidate_data

    def score(self) -> LossMetrics:
        """Compute all loss metrics."""
        # Build signature maps for comparison
        ref_signatures = self._build_signature_map(
            self.reference.get("tagged_isomorphs", [])
        )
        cand_signatures = self._build_signature_map(
            self.candidate.get("tagged_isomorphs", [])
        )

        # Compute distances
        structural = self._structural_distance(ref_signatures, cand_signatures)
        type_dist = self._type_distance(ref_signatures, cand_signatures)
        operation = self._operation_distance(ref_signatures, cand_signatures)

        # Combined weighted loss
        combined = (
            self.WEIGHT_STRUCTURAL * structural +
            self.WEIGHT_TYPE * type_dist +
            self.WEIGHT_OPERATION * operation
        )

        # Pattern matching statistics
        matched = len(set(ref_signatures.keys()) & set(cand_signatures.keys()))
        missing = len(set(ref_signatures.keys()) - set(cand_signatures.keys()))
        extra = len(set(cand_signatures.keys()) - set(ref_signatures.keys()))

        return LossMetrics(
            file=self.reference.get("source_file", "unknown"),
            structural_distance=structural,
            type_distance=type_dist,
            operation_distance=operation,
            combined_loss=combined,
            reference_isomorphs=len(ref_signatures),
            candidate_isomorphs=len(cand_signatures),
            matched_patterns=matched,
            missing_patterns=missing,
            extra_patterns=extra
        )

    def _build_signature_map(self, isomorphs: List[Dict]) -> Dict[str, Dict]:
        """Build map of signature -> isomorph data."""
        signature_map = {}
        for isomorph in isomorphs:
            ast_pattern = isomorph.get("ast_pattern", {})
            signature = ast_pattern.get("signature", "")
            if signature:
                signature_map[signature] = isomorph
        return signature_map

    def _structural_distance(
        self,
        ref_signatures: Dict[str, Dict],
        cand_signatures: Dict[str, Dict]
    ) -> float:
        """
        Compute structural distance (normalized graph edit distance).

        Edit operations: insert/delete node, relabel node
        Distance = number of edits / max(ref_nodes, cand_nodes)
        """
        if not ref_signatures and not cand_signatures:
            return 0.0

        # Count matched vs unmatched signatures
        ref_set = set(ref_signatures.keys())
        cand_set = set(cand_signatures.keys())

        matched = len(ref_set & cand_set)
        missing = len(ref_set - cand_set)  # In ref but not in candidate
        extra = len(cand_set - ref_set)    # In candidate but not in ref

        # Edit distance approximation
        edits = missing + extra
        max_patterns = max(len(ref_signatures), len(cand_signatures))

        return edits / max_patterns if max_patterns > 0 else 0.0

    def _type_distance(
        self,
        ref_signatures: Dict[str, Dict],
        cand_signatures: Dict[str, Dict]
    ) -> float:
        """
        Compute type distance (type inference mismatches).

        For matched patterns, compare if types/attributes match.
        """
        if not ref_signatures:
            return 1.0 if cand_signatures else 0.0

        matched_signatures = set(ref_signatures.keys()) & set(cand_signatures.keys())
        if not matched_signatures:
            return 1.0

        mismatches = 0
        for sig in matched_signatures:
            ref_pattern = ref_signatures[sig].get("ast_pattern", {})
            cand_pattern = cand_signatures[sig].get("ast_pattern", {})

            # Compare structure (simplified type comparison)
            ref_structure = ref_pattern.get("structure", [])
            cand_structure = cand_pattern.get("structure", [])

            if ref_structure != cand_structure:
                mismatches += 1

        return mismatches / len(matched_signatures)

    def _operation_distance(
        self,
        ref_signatures: Dict[str, Dict],
        cand_signatures: Dict[str, Dict]
    ) -> float:
        """
        Compute operation distance (MLIR operation differences).

        For matched patterns, compare MLIR region operations.
        """
        if not ref_signatures:
            return 1.0 if cand_signatures else 0.0

        matched_signatures = set(ref_signatures.keys()) & set(cand_signatures.keys())
        if not matched_signatures:
            return 1.0

        # For reference data, we have MLIR tagging
        # For candidate data (from cppfort), we would need MLIR generation
        # Currently we'll compare at root kind level as proxy

        mismatches = 0
        for sig in matched_signatures:
            ref_pattern = ref_signatures[sig].get("ast_pattern", {})
            cand_pattern = cand_signatures[sig].get("ast_pattern", {})

            ref_kind = ref_pattern.get("root_kind", "")
            cand_kind = cand_pattern.get("root_kind", "")

            if ref_kind != cand_kind:
                mismatches += 1

        return mismatches / len(matched_signatures)


def main():
    parser = argparse.ArgumentParser(
        description="Score semantic loss between reference and candidate ASTs"
    )
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Reference tagged isomorph JSON (from cppfront)"
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="Candidate isomorph JSON (from cppfort)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output loss metrics JSON"
    )

    args = parser.parse_args()

    # Read data
    print(f"Reading reference: {args.reference}")
    reference_data = json.loads(args.reference.read_text())

    print(f"Reading candidate: {args.candidate}")
    candidate_data = json.loads(args.candidate.read_text())

    # Score loss
    print("Computing semantic loss...")
    scorer = SemanticLossScorer(reference_data, candidate_data)
    metrics = scorer.score()

    # Write output
    output_data = {
        "version": "1.0",
        "metrics": metrics.to_dict()
    }

    print(f"Writing metrics to {args.output}")
    args.output.write_text(json.dumps(output_data, indent=2))

    # Print summary
    print(f"\n=== Semantic Loss Summary ===")
    print(f"File: {metrics.file}")
    print(f"Combined loss: {metrics.combined_loss:.3f}")
    print(f"  Structural: {metrics.structural_distance:.3f}")
    print(f"  Type:       {metrics.type_distance:.3f}")
    print(f"  Operation:  {metrics.operation_distance:.3f}")
    print(f"\nPattern matching:")
    print(f"  Reference patterns: {metrics.reference_isomorphs}")
    print(f"  Candidate patterns: {metrics.candidate_isomorphs}")
    print(f"  Matched: {metrics.matched_patterns}")
    print(f"  Missing: {metrics.missing_patterns}")
    print(f"  Extra: {metrics.extra_patterns}")

    # Loss category
    if metrics.combined_loss == 0.0:
        category = "ZERO LOSS (perfect)"
    elif metrics.combined_loss < 0.1:
        category = "LOW LOSS (minor differences)"
    elif metrics.combined_loss < 0.5:
        category = "MEDIUM LOSS (different strategies)"
    else:
        category = "HIGH LOSS (semantic differences)"

    print(f"\nLoss category: {category}")


if __name__ == "__main__":
    main()
