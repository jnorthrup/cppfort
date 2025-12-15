#!/usr/bin/env python3
"""
Extract AST graph isomorphs from Clang AST dumps.

An AST isomorph is a canonical structural pattern that can be mapped to
an MLIR region. This script identifies subgraphs in the Clang AST and
computes their normalized signatures for isomorphism detection.

Usage:
    ./extract_ast_isomorphs.py --ast reference.ast.txt --output isomorphs.json
"""

import argparse
import json
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class ASTNode:
    """Represents a node in the Clang AST."""
    kind: str
    location: str
    attributes: str
    children: List['ASTNode']
    depth: int

    def to_dict(self):
        return {
            "kind": self.kind,
            "location": self.location,
            "attributes": self.attributes,
            "children": [c.to_dict() for c in self.children]
        }


@dataclass
class Isomorph:
    """Represents a canonical AST subgraph pattern."""
    id: str
    signature: str
    root_kind: str
    structure: List[str]
    depth: int
    node_count: int
    examples: List[Dict]

    def to_dict(self):
        return asdict(self)


class ASTParser:
    """Parses Clang AST text dump into tree structure."""

    def __init__(self, ast_text: str):
        self.lines = ast_text.split('\n')
        self.current_line = 0

    def parse(self) -> ASTNode:
        """Parse full AST dump."""
        root = ASTNode(
            kind="TranslationUnit",
            location="",
            attributes="",
            children=[],
            depth=0
        )

        while self.current_line < len(self.lines):
            node = self._parse_node(0)
            if node:
                root.children.append(node)

        return root

    def _parse_node(self, expected_depth: int) -> Optional[ASTNode]:
        """Parse a single AST node and its children."""
        if self.current_line >= len(self.lines):
            return None

        line = self.lines[self.current_line]

        # Skip empty lines
        if not line.strip():
            self.current_line += 1
            return None

        # Calculate depth from indentation
        depth = 0
        for char in line:
            if char == ' ':
                depth += 1
            elif char in ('|', '`', '-'):
                depth += 1
            else:
                break

        # If depth doesn't match expected, this node belongs to parent's sibling
        if depth < expected_depth:
            return None

        # Parse node line: "|-Kind <location> attributes"
        match = re.match(r'[|\s`-]*(\w+)\s+(<[^>]+>)?\s*(.*)', line)
        if not match:
            self.current_line += 1
            return None

        kind = match.group(1)
        location = match.group(2) or ""
        attributes = match.group(3) or ""

        node = ASTNode(
            kind=kind,
            location=location,
            attributes=attributes,
            children=[],
            depth=depth
        )

        self.current_line += 1

        # Parse children
        while self.current_line < len(self.lines):
            child = self._parse_node(depth + 2)  # Children are indented by 2
            if child is None:
                break
            node.children.append(child)

        return node


class IsomorphExtractor:
    """Extracts canonical graph isomorphs from AST."""

    def __init__(self, ast: ASTNode):
        self.ast = ast
        self.isomorphs: Dict[str, Isomorph] = {}

    def extract(self) -> List[Isomorph]:
        """Extract all isomorphs from AST."""
        self._extract_function_isomorphs(self.ast)
        self._extract_control_flow_isomorphs(self.ast)
        self._extract_expression_isomorphs(self.ast)

        return list(self.isomorphs.values())

    def _extract_function_isomorphs(self, node: ASTNode):
        """Extract function definition patterns."""
        if node.kind == "FunctionDecl":
            isomorph = self._create_isomorph(node, "function")
            if isomorph:
                self._add_isomorph(isomorph)

        for child in node.children:
            self._extract_function_isomorphs(child)

    def _extract_control_flow_isomorphs(self, node: ASTNode):
        """Extract control flow patterns (if, while, for, etc.)."""
        if node.kind in ["IfStmt", "WhileStmt", "ForStmt", "DoStmt"]:
            isomorph = self._create_isomorph(node, "control_flow")
            if isomorph:
                self._add_isomorph(isomorph)

        for child in node.children:
            self._extract_control_flow_isomorphs(child)

    def _extract_expression_isomorphs(self, node: ASTNode):
        """Extract expression patterns (calls, operators, etc.)."""
        if node.kind in ["CallExpr", "CXXOperatorCallExpr", "BinaryOperator"]:
            isomorph = self._create_isomorph(node, "expression")
            if isomorph:
                self._add_isomorph(isomorph)

        for child in node.children:
            self._extract_expression_isomorphs(child)

    def _create_isomorph(self, node: ASTNode, category: str) -> Optional[Isomorph]:
        """Create an isomorph from an AST subgraph."""
        structure = self._extract_structure(node)
        signature = self._compute_signature(structure)

        return Isomorph(
            id=f"{category}_{signature[:16]}",
            signature=signature,
            root_kind=node.kind,
            structure=structure,
            depth=self._compute_depth(node),
            node_count=self._count_nodes(node),
            examples=[{
                "location": node.location,
                "attributes": node.attributes
            }]
        )

    def _extract_structure(self, node: ASTNode, max_depth: int = 3) -> List[str]:
        """Extract structural pattern from node (kind sequence)."""
        structure = [node.kind]

        if max_depth > 0:
            for child in node.children:
                child_structure = self._extract_structure(child, max_depth - 1)
                structure.extend(child_structure)

        return structure

    def _compute_signature(self, structure: List[str]) -> str:
        """Compute canonical signature for structure."""
        signature_str = ":".join(structure)
        return hashlib.sha256(signature_str.encode()).hexdigest()

    def _compute_depth(self, node: ASTNode) -> int:
        """Compute maximum depth of subtree."""
        if not node.children:
            return 1
        return 1 + max(self._compute_depth(child) for child in node.children)

    def _count_nodes(self, node: ASTNode) -> int:
        """Count total nodes in subtree."""
        return 1 + sum(self._count_nodes(child) for child in node.children)

    def _add_isomorph(self, isomorph: Isomorph):
        """Add isomorph to database, merging if already exists."""
        if isomorph.signature in self.isomorphs:
            # Merge examples
            existing = self.isomorphs[isomorph.signature]
            existing.examples.extend(isomorph.examples)
        else:
            self.isomorphs[isomorph.signature] = isomorph


def main():
    parser = argparse.ArgumentParser(
        description="Extract AST graph isomorphs from Clang AST dumps"
    )
    parser.add_argument(
        "--ast",
        type=Path,
        required=True,
        help="Input Clang AST dump (text format)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file for isomorphs"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum depth for structure extraction"
    )

    args = parser.parse_args()

    # Read AST dump
    print(f"Reading AST dump: {args.ast}")
    ast_text = args.ast.read_text()

    # Parse AST
    print("Parsing AST...")
    ast_parser = ASTParser(ast_text)
    ast = ast_parser.parse()

    # Extract isomorphs
    print("Extracting isomorphs...")
    extractor = IsomorphExtractor(ast)
    isomorphs = extractor.extract()

    # Write output
    output_data = {
        "source_file": str(args.ast),
        "total_isomorphs": len(isomorphs),
        "isomorphs": [iso.to_dict() for iso in isomorphs]
    }

    print(f"Writing {len(isomorphs)} isomorphs to {args.output}")
    args.output.write_text(json.dumps(output_data, indent=2))

    # Print summary
    print(f"\n=== Isomorph Summary ===")
    print(f"Total unique patterns: {len(isomorphs)}")

    by_kind = defaultdict(int)
    for iso in isomorphs:
        by_kind[iso.root_kind] += 1

    print(f"\nBy node kind:")
    for kind, count in sorted(by_kind.items(), key=lambda x: -x[1]):
        print(f"  {kind}: {count}")


if __name__ == "__main__":
    main()
