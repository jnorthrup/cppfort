#!/usr/bin/env python3
"""
Direct side-by-side comparison of clang AST dumps from cppfront and cppfort.

This tool performs back-mapping of clang AST semantics by:
1. Generating clang AST dumps from both transpilers for the same cpp2 source
2. Comparing AST structures directly (node-by-node)
3. Reporting semantic loss metrics based on structural differences

This is more direct than the isomorph-based approach, as it compares
the actual AST dumps rather than extracted patterns.

Usage:
    ./compare_ast_dumps.py --cpp2 file.cpp2 --ref-ast ref.ast.txt --output report.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class ASTNode:
    """Represents a node in a clang AST dump."""
    kind: str
    address: str
    location: str
    attributes: str
    depth: int
    children: List['ASTNode'] = field(default_factory=list)

    def __hash__(self):
        return hash((self.kind, self.address))

    def to_dict(self):
        return {
            "kind": self.kind,
            "address": self.address,
            "location": self.location,
            "attributes": self.attributes,
            "depth": self.depth,
            "children": [c.to_dict() for c in self.children]
        }


@dataclass
class ASTDiff:
    """Represents a difference between two AST nodes."""
    diff_type: str  # "missing", "extra", "kind_mismatch", "attr_mismatch"
    ref_node: Optional[ASTNode]
    cand_node: Optional[ASTNode]
    path: str  # Path to the node (e.g., "FunctionDecl:main:CompoundStmt:IfStmt")

    def to_dict(self):
        return {
            "type": self.diff_type,
            "path": self.path,
            "ref": {
                "kind": self.ref_node.kind if self.ref_node else None,
                "location": self.ref_node.location if self.ref_node else None
            },
            "cand": {
                "kind": self.cand_node.kind if self.cand_node else None,
                "location": self.cand_node.location if self.cand_node else None
            }
        }


@dataclass
class ComparisonMetrics:
    """Metrics for AST comparison."""
    file: str
    ref_total_nodes: int
    cand_total_nodes: int
    matched_nodes: int
    missing_nodes: int
    extra_nodes: int
    kind_mismatches: int
    structural_similarity: float  # 0.0 to 1.0
    semantic_loss: float  # 0.0 to 1.0

    def to_dict(self):
        return {
            "file": self.file,
            "ref_total_nodes": self.ref_total_nodes,
            "cand_total_nodes": self.cand_total_nodes,
            "matched_nodes": self.matched_nodes,
            "missing_nodes": self.missing_nodes,
            "extra_nodes": self.extra_nodes,
            "kind_mismatches": self.kind_mismatches,
            "structural_similarity": self.structural_similarity,
            "semantic_loss": self.semantic_loss
        }


class ASTDumpParser:
    """Parses clang AST dump text format."""

    def __init__(self, ast_text: str):
        self.lines = ast_text.split('\n')
        self.current_line = 0

    def parse(self) -> ASTNode:
        """Parse full AST dump into tree."""
        root = ASTNode(
            kind="TranslationUnitDecl",
            address="",
            location="",
            attributes="",
            depth=0
        )

        while self.current_line < len(self.lines):
            node = self._parse_node(0)
            if node:
                root.children.append(node)

        return root

    def _parse_node(self, expected_depth: int) -> Optional[ASTNode]:
        """Parse a single AST node and its children."""
        while self.current_line < len(self.lines):
            line = self.lines[self.current_line].strip()
            if not line:
                self.current_line += 1
                continue

            # Calculate depth from indentation
            full_line = self.lines[self.current_line]
            depth = 0
            for char in full_line:
                if char == ' ':
                    depth += 1
                elif char in ('|', '`', '-'):
                    depth += 1
                else:
                    break

            depth = depth // 2  # Two spaces per level

            if depth < expected_depth:
                return None

            # Parse node line format:
            # |-FunctionDecl 0x12345 <file:line:col, line:col> <implicit> attributes
            match = re.match(r'[|\s`-]*(\w+)\s+(0x[0-9a-f]+)\s*(<[^>]+>)?\s*(.*)', line)
            if not match:
                self.current_line += 1
                continue

            kind = match.group(1)
            address = match.group(2)
            location = match.group(3) or ""
            attributes = match.group(4) or ""

            node = ASTNode(
                kind=kind,
                address=address,
                location=location,
                attributes=attributes,
                depth=depth
            )

            self.current_line += 1

            # Parse children
            while self.current_line < len(self.lines):
                child = self._parse_node(depth + 1)
                if child is None:
                    break
                node.children.append(child)

            return node

        return None


class ASTComparator:
    """Compares two AST dumps for semantic differences."""

    def __init__(self, ref_ast: ASTNode, cand_ast: ASTNode, file_name: str = ""):
        self.ref_ast = ref_ast
        self.cand_ast = cand_ast
        self.file_name = file_name
        self.diffs: List[ASTDiff] = []

    def compare(self) -> ComparisonMetrics:
        """Perform comparison and return metrics."""
        self.diffs = []
        self._compare_nodes(self.ref_ast, self.cand_ast, "")

        ref_count = self._count_nodes(self.ref_ast)
        cand_count = self._count_nodes(self.cand_ast)

        missing = sum(1 for d in self.diffs if d.diff_type == "missing")
        extra = sum(1 for d in self.diffs if d.diff_type == "extra")
        kind_mismatch = sum(1 for d in self.diffs if d.diff_type == "kind_mismatch")
        matched = ref_count - missing

        # Structural similarity: matched / max(ref, cand)
        max_nodes = max(ref_count, cand_count)
        similarity = matched / max_nodes if max_nodes > 0 else 1.0

        # Semantic loss: normalized by node count and differences
        loss = (missing + extra + kind_mismatch * 2) / max_nodes if max_nodes > 0 else 0.0
        loss = min(loss, 1.0)

        return ComparisonMetrics(
            file=self.file_name,
            ref_total_nodes=ref_count,
            cand_total_nodes=cand_count,
            matched_nodes=matched,
            missing_nodes=missing,
            extra_nodes=extra,
            kind_mismatches=kind_mismatch,
            structural_similarity=similarity,
            semantic_loss=loss
        )

    def _compare_nodes(self, ref: ASTNode, cand: ASTNode, path: str):
        """Recursively compare AST nodes."""
        # Compare current nodes
        if ref.kind != cand.kind:
            self.diffs.append(ASTDiff(
                diff_type="kind_mismatch",
                ref_node=ref,
                cand_node=cand,
                path=path
            ))

        # Compare children
        ref_children = ref.children
        cand_children = cand.children

        # Match children by kind (simplified matching)
        ref_by_kind = defaultdict(list)
        cand_by_kind = defaultdict(list)

        for i, child in enumerate(ref_children):
            ref_by_kind[child.kind].append((i, child))

        for i, child in enumerate(cand_children):
            cand_by_kind[child.kind].append((i, child))

        # Find matches
        matched_ref = set()
        matched_cand = set()

        for kind in ref_by_kind:
            if kind in cand_by_kind:
                # Match as many as possible
                ref_list = ref_by_kind[kind]
                cand_list = cand_by_kind[kind]
                min_count = min(len(ref_list), len(cand_list))

                for i in range(min_count):
                    ref_idx, ref_child = ref_list[i]
                    cand_idx, cand_child = cand_list[i]
                    matched_ref.add(ref_idx)
                    matched_cand.add(cand_idx)

                    new_path = f"{path}:{kind}" if path else kind
                    self._compare_nodes(ref_child, cand_child, new_path)

        # Find missing in candidate
        for i, child in enumerate(ref_children):
            if i not in matched_ref:
                new_path = f"{path}:{child.kind}" if path else child.kind
                self.diffs.append(ASTDiff(
                    diff_type="missing",
                    ref_node=child,
                    cand_node=None,
                    path=new_path
                ))

        # Find extra in candidate
        for i, child in enumerate(cand_children):
            if i not in matched_cand:
                new_path = f"{path}:{child.kind}" if path else child.kind
                self.diffs.append(ASTDiff(
                    diff_type="extra",
                    ref_node=None,
                    cand_node=child,
                    path=new_path
                ))

    def _count_nodes(self, node: ASTNode) -> int:
        """Count total nodes in AST."""
        return 1 + sum(self._count_nodes(child) for child in node.children)


def main():
    parser = argparse.ArgumentParser(
        description="Direct side-by-side comparison of clang AST dumps"
    )
    parser.add_argument(
        "--cpp2",
        type=Path,
        required=True,
        help="Source cpp2 file"
    )
    parser.add_argument(
        "--ref-ast",
        type=Path,
        required=True,
        help="Reference AST dump (from cppfront)"
    )
    parser.add_argument(
        "--cand-cpp",
        type=Path,
        help="Candidate C++ file (pre-transpiled by cppfort). If not provided, will transpile from cpp2."
    )
    parser.add_argument(
        "--cppfort-bin",
        type=Path,
        default=Path("./build/src/cppfort"),
        help="Path to cppfort binary"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file for comparison results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed differences"
    )

    args = parser.parse_args()

    basename = args.cpp2.stem

    # 1. Parse reference AST
    print(f"Parsing reference AST: {args.ref_ast}")
    ref_text = args.ref_ast.read_text()
    ref_parser = ASTDumpParser(ref_text)
    ref_ast = ref_parser.parse()

    # 2. Transpile with cppfort if needed
    if args.cand_cpp:
        cand_cpp_path = args.cand_cpp
    else:
        print(f"Transpiling {args.cpp2} with cppfort...")
        cand_cpp_path = Path(f"/tmp/{basename}.cppfort.cpp")
        import subprocess
        result = subprocess.run(
            [str(args.cppfort_bin), str(args.cpp2), str(cand_cpp_path)],
            capture_output=True
        )
        if result.returncode != 0:
            print(f"Error: cppfort transpilation failed")
            print(result.stderr.decode())
            return 1

    # 3. Generate candidate AST dump
    print(f"Generating candidate AST dump...")
    import subprocess
    result = subprocess.run(
        ["clang++", "-std=c++20", "-Xclang", "-ast-dump", "-fsyntax-only", str(cand_cpp_path)],
        capture_output=True,
        text=True
    )
    cand_ast_text = result.stdout

    # 4. Parse candidate AST
    print("Parsing candidate AST...")
    cand_parser = ASTDumpParser(cand_ast_text)
    cand_ast = cand_parser.parse()

    # 5. Compare
    print("Comparing ASTs...")
    comparator = ASTComparator(ref_ast, cand_ast, basename)
    metrics = comparator.compare()

    # 6. Write output
    output = {
        "metrics": metrics.to_dict(),
        "differences": [d.to_dict() for d in comparator.diffs]
    }

    print(f"Writing results to {args.output}")
    args.output.write_text(json.dumps(output, indent=2))

    # 7. Print summary
    print(f"\n=== AST Comparison Summary ===")
    print(f"File: {metrics.file}")
    print(f"Reference nodes: {metrics.ref_total_nodes}")
    print(f"Candidate nodes: {metrics.cand_total_nodes}")
    print(f"Matched: {metrics.matched_nodes}")
    print(f"Missing: {metrics.missing_nodes}")
    print(f"Extra: {metrics.extra_nodes}")
    print(f"Kind mismatches: {metrics.kind_mismatches}")
    print(f"Structural similarity: {metrics.structural_similarity:.3f}")
    print(f"Semantic loss: {metrics.semantic_loss:.3f}")

    if args.verbose and comparator.diffs:
        print(f"\n=== Top 20 Differences ===")
        for diff in comparator.diffs[:20]:
            if diff.diff_type == "missing":
                print(f"  MISSING: {diff.path} (ref: {diff.ref_node.kind})")
            elif diff.diff_type == "extra":
                print(f"  EXTRA: {diff.path} (cand: {diff.cand_node.kind})")
            else:
                print(f"  MISMATCH: {diff.path} (ref: {diff.ref_node.kind}, cand: {diff.cand_node.kind})")

    return 0


if __name__ == "__main__":
    exit(main())
