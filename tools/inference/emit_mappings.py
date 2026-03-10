#!/usr/bin/env python3
"""Emit mapping candidates from Clang AST → MLIR region templates.

Usage:
  python emit_mappings.py -i sample.cpp -o mappings.json -- -std=c++20

Extends parse_and_infer.py to emit mapping candidates per docs/MAPPING_SPEC.md.
"""

import sys
import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

try:
    from clang import cindex
    try:
        from .clang_support import configure_libclang
    except ImportError:
        from clang_support import configure_libclang
    configure_libclang(cindex)
except Exception:
    print("ERROR: clang python bindings not found. Install with `pip install clang`.")
    sys.exit(2)

try:
    from .parse_and_infer import (
        parse_file,
        Node,
        semantic_signature,
        grammar_fingerprint,
        semantic_sections,
    )
except ImportError:
    from parse_and_infer import (
        parse_file,
        Node,
        semantic_signature,
        grammar_fingerprint,
        semantic_sections,
    )


@dataclass
class Mapping:
    """Schema per docs/MAPPING_SPEC.md"""
    id: str
    source_sample: Dict[str, Any]
    ast_kind: str
    son_node: Optional[str]
    mlir_template: str
    pattern: str
    confidence: float
    examples: List[Dict[str, str]]
    notes: str


def emit_mapping_candidates(root: Node, filename: str) -> List[Dict]:
    """Extract mapping candidates from AST."""
    mappings = []
    counter = {"id": 0}

    def next_id(prefix: str) -> str:
        counter["id"] += 1
        return f"{prefix}_{counter['id']}"

    def make_mapping(prefix: str,
                     node: Node,
                     ast_kind: str,
                     son_node: Optional[str],
                     mlir_template: str,
                     confidence: float,
                     examples: List[Dict[str, str]],
                     notes: str) -> Dict[str, Any]:
        signature = semantic_signature(node)
        fingerprint = grammar_fingerprint(node)
        return {
            "id": next_id(prefix),
            "source_sample": {
                "file": filename,
                "span": {"start": node.start, "end": node.end},
            },
            "ast_kind": ast_kind,
            "son_node": son_node,
            "mlir_template": mlir_template,
            "pattern": f"{signature} @ {fingerprint}",
            "semantic_signature": signature,
            "grammar_fingerprint": fingerprint,
            "semantic_sections": semantic_sections(node),
            "semantics_source": "clang",
            "grammar_source": "normalized_ast_shape",
            "confidence": confidence,
            "examples": examples,
            "notes": notes,
        }

    def visit(n: Node, parent=None):
        # FunctionDecl → cpp2.region
        if n.kind == "FUNCTION_DECL":
            mappings.append(make_mapping(
                "func_decl_to_region",
                n,
                "FunctionDecl",
                "RegionNode",
                "cpp2.func @{name}(%args) -> %results { %body }",
                0.9,
                [{"input": f"function {n.spelling}(...)", "output": f"cpp2.func @{n.spelling}"}],
                "Map function signature to region with entry block."
            ))

        # IfStmt → cpp2.if
        elif n.kind == "IF_STMT":
            mappings.append(make_mapping(
                "ifstmt_to_cpp2_if",
                n,
                "IfStmt",
                "IfNode",
                "cpp2.if %cond { %then_region } else { %else_region }",
                0.95,
                [{"input": "if (x > 0) {...} else {...}", "output": "cpp2.if %cond { ... }"}],
                "Split branches into regions and preserve branch semantics."
            ))

        # ForStmt → cpp2.for
        elif n.kind == "FOR_STMT":
            mappings.append(make_mapping(
                "forstmt_to_cpp2_for",
                n,
                "ForStmt",
                "LoopNode",
                "cpp2.for %init, %cond, %inc { %body }",
                0.85,
                [{"input": "for (int i = 0; i < n; ++i)", "output": "cpp2.for ..."}],
                "Map loop init, condition, increment, and body as separate semantic parts."
            ))

        # WhileStmt → cpp2.while
        elif n.kind == "WHILE_STMT":
            mappings.append(make_mapping(
                "whilestmt_to_cpp2_while",
                n,
                "WhileStmt",
                "LoopNode",
                "cpp2.while %cond { %body }",
                0.9,
                [{"input": "while (x > 0) {...}", "output": "cpp2.while %cond { ... }"}],
                "Map the loop condition and body while preserving typed control semantics."
            ))

        # ReturnStmt → cpp2.return
        elif n.kind == "RETURN_STMT":
            mappings.append(make_mapping(
                "returnstmt_to_cpp2_return",
                n,
                "ReturnStmt",
                "ReturnNode",
                "cpp2.return %value : %type",
                0.95,
                [{"input": "return x;", "output": "cpp2.return %x"}],
                "Handle value and void returns as the same semantic operation."
            ))

        # VarDecl → cpp2.var
        elif n.kind == "VAR_DECL":
            mappings.append(make_mapping(
                "vardecl_to_cpp2_var",
                n,
                "VarDecl",
                "VarDeclNode",
                "cpp2.var @{name} : %type = %init",
                0.88,
                [{"input": f"{n.spelling}: type = value", "output": f"cpp2.var @{n.spelling}"}],
                "Map typed storage and initializer semantics."
            ))

        # CallExpr → cpp2.call
        elif n.kind in {"CALL_EXPR", "CXX_MEMBER_CALL_EXPR"}:
            mappings.append(make_mapping(
                "callexpr_to_cpp2_call",
                n,
                "CallExpr",
                "CallNode",
                "cpp2.call @{callee}(%args) : %result_type",
                0.92,
                [{"input": "func(a, b)", "output": "cpp2.call @func(%a, %b)"}],
                "Handle free calls and member-shaped calls through one semantic call path."
            ))

        # BinaryOperator → cpp2.binop
        elif n.kind == "BINARY_OPERATOR":
            mappings.append(make_mapping(
                "binop_to_cpp2_binop",
                n,
                "BinaryOperator",
                "BinOpNode",
                "cpp2.binop {op} %lhs, %rhs : %type",
                0.93,
                [{"input": "a + b", "output": "cpp2.binop add %a, %b"}],
                "Map typed binary data flow independent of surface spelling."
            ))

        for c in n.children:
            visit(c, n)

    visit(root)
    return mappings


def main(argv):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--clang-args", nargs="*", default=[])
    args, unknown = parser.parse_known_args(argv)

    clang_args = args.clang_args or unknown
    if clang_args and clang_args[0] == "--":
        clang_args = clang_args[1:]

    root = parse_file(args.input, clang_args)
    mappings = emit_mapping_candidates(root, args.input)

    out = {
        "file": args.input,
        "mappings": mappings,
        "schema_version": "1.0",
        "spec": "docs/MAPPING_SPEC.md"
    }

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Emitted {len(mappings)} mapping candidates to {args.output}")


if __name__ == "__main__":
    main(sys.argv[1:])
