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
    # Configure libclang path if provided
    if "LIBCLANG_PATH" in os.environ:
        cindex.Config.set_library_file(os.environ["LIBCLANG_PATH"])
except Exception:
    print("ERROR: clang python bindings not found. Install with `pip install clang`.")
    sys.exit(2)

from parse_and_infer import parse_file, Node


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

    def visit(n: Node, parent=None):
        # FunctionDecl → cpp2.region
        if n.kind == "FUNCTION_DECL":
            mappings.append({
                "id": next_id("func_decl_to_region"),
                "source_sample": {
                    "file": filename,
                    "span": {"start": n.start, "end": n.end}
                },
                "ast_kind": "FunctionDecl",
                "son_node": "RegionNode",
                "mlir_template": "cpp2.func @{name}(%args) -> %results { %body }",
                "pattern": "FunctionDecl with optional CompoundStmt body",
                "confidence": 0.9,
                "examples": [{"input": f"function {n.spelling}(...)", "output": f"cpp2.func @{n.spelling}"}],
                "notes": "Map function signature to region with entry block"
            })

        # IfStmt → cpp2.if
        elif n.kind == "IF_STMT":
            mappings.append({
                "id": next_id("ifstmt_to_cpp2_if"),
                "source_sample": {
                    "file": filename,
                    "span": {"start": n.start, "end": n.end}
                },
                "ast_kind": "IfStmt",
                "son_node": "IfNode",
                "mlir_template": "cpp2.if %cond { %then_region } else { %else_region }",
                "pattern": "IfStmt with then/else branches",
                "confidence": 0.95,
                "examples": [{"input": "if (x>0) {...} else {...}", "output": "cpp2.if %cond { ... }"}],
                "notes": "Split branches into regions; handle missing else"
            })

        # ForStmt → cpp2.for
        elif n.kind == "FOR_STMT":
            mappings.append({
                "id": next_id("forstmt_to_cpp2_for"),
                "source_sample": {
                    "file": filename,
                    "span": {"start": n.start, "end": n.end}
                },
                "ast_kind": "ForStmt",
                "son_node": "LoopNode",
                "mlir_template": "cpp2.for %init, %cond, %inc { %body }",
                "pattern": "ForStmt with init/cond/inc/body",
                "confidence": 0.85,
                "examples": [{"input": "for (int i=0; i<n; i++)", "output": "cpp2.for ..."}],
                "notes": "Map loop components to region parameters"
            })

        # WhileStmt → cpp2.while
        elif n.kind == "WHILE_STMT":
            mappings.append({
                "id": next_id("whilestmt_to_cpp2_while"),
                "source_sample": {
                    "file": filename,
                    "span": {"start": n.start, "end": n.end}
                },
                "ast_kind": "WhileStmt",
                "son_node": "LoopNode",
                "mlir_template": "cpp2.while %cond { %body }",
                "pattern": "WhileStmt with condition and body",
                "confidence": 0.9,
                "examples": [{"input": "while (x>0) {...}", "output": "cpp2.while %cond { ... }"}],
                "notes": "Map condition to block argument"
            })

        # ReturnStmt → cpp2.return
        elif n.kind == "RETURN_STMT":
            mappings.append({
                "id": next_id("returnstmt_to_cpp2_return"),
                "source_sample": {
                    "file": filename,
                    "span": {"start": n.start, "end": n.end}
                },
                "ast_kind": "ReturnStmt",
                "son_node": "ReturnNode",
                "mlir_template": "cpp2.return %value : %type",
                "pattern": "ReturnStmt with optional value",
                "confidence": 0.95,
                "examples": [{"input": "return x;", "output": "cpp2.return %x"}],
                "notes": "Handle void returns"
            })

        # VarDecl → cpp2.var
        elif n.kind == "VAR_DECL":
            mappings.append({
                "id": next_id("vardecl_to_cpp2_var"),
                "source_sample": {
                    "file": filename,
                    "span": {"start": n.start, "end": n.end}
                },
                "ast_kind": "VarDecl",
                "son_node": "VarDeclNode",
                "mlir_template": "cpp2.var @{name} : %type = %init",
                "pattern": "VarDecl with optional initializer",
                "confidence": 0.88,
                "examples": [{"input": f"{n.spelling}: type = value", "output": f"cpp2.var @{n.spelling}"}],
                "notes": "Map type and init expr"
            })

        # CallExpr → cpp2.call
        elif n.kind == "CALL_EXPR":
            mappings.append({
                "id": next_id("callexpr_to_cpp2_call"),
                "source_sample": {
                    "file": filename,
                    "span": {"start": n.start, "end": n.end}
                },
                "ast_kind": "CallExpr",
                "son_node": "CallNode",
                "mlir_template": "cpp2.call @{callee}(%args) : %result_type",
                "pattern": "CallExpr with callee and arguments",
                "confidence": 0.92,
                "examples": [{"input": "func(a, b)", "output": "cpp2.call @func(%a, %b)"}],
                "notes": "Handle UFCS and method calls"
            })

        # BinaryOperator → cpp2.binop
        elif n.kind == "BINARY_OPERATOR":
            mappings.append({
                "id": next_id("binop_to_cpp2_binop"),
                "source_sample": {
                    "file": filename,
                    "span": {"start": n.start, "end": n.end}
                },
                "ast_kind": "BinaryOperator",
                "son_node": "BinOpNode",
                "mlir_template": "cpp2.binop {op} %lhs, %rhs : %type",
                "pattern": "BinaryOperator with operator and operands",
                "confidence": 0.93,
                "examples": [{"input": "a + b", "output": "cpp2.binop add %a, %b"}],
                "notes": "Map operator tokens to MLIR ops"
            })

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

    root = parse_file(args.input, args.clang_args or unknown)
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
