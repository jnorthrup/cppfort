#!/usr/bin/env python3
"""Prototype: parse a C/C++ file with libclang and infer MLIR-like regions/blocks.

Usage:
  python parse_and_infer.py -i sample.cpp -o mapping.json -- -std=c++20

This is a small prototype that:
 - parses the source with clang.cindex
 - builds a tree of extents (start/end positions)
 - runs simple heuristics to infer 'regions' and 'blocks' (functions, compound stmts, control flow)
 - writes JSON output describing inferred regions

This now separates two inference layers:
 - semantics inferred from Clang's typed AST
 - normalized grammar shape inferred from AST structure

This keeps the emitted definitions stronger than raw cursor kinds while staying
stable across many corpus samples.
"""

import sys
import os
import json
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Any, Iterable, Optional

try:
    from .clang_support import configure_libclang
except ImportError:
    from clang_support import configure_libclang

try:
    from clang import cindex
    configure_libclang(cindex)
except Exception:
    print("ERROR: clang python bindings not found. Install with `pip install clang` and ensure libclang is available.")
    sys.exit(2)


@dataclass
class Node:
    kind: str
    spelling: str
    displayname: str
    type_spelling: str
    is_definition: bool
    start: Dict[str, int]
    end: Dict[str, int]
    children: List[Any]


KIND_LABELS = {
    "FUNCTION_DECL": "function",
    "PARM_DECL": "param",
    "COMPOUND_STMT": "compound",
    "DECL_STMT": "decl",
    "VAR_DECL": "var",
    "IF_STMT": "if",
    "FOR_STMT": "for",
    "WHILE_STMT": "while",
    "CXX_FOR_RANGE_STMT": "for_range",
    "RETURN_STMT": "return",
    "CALL_EXPR": "call",
    "CXX_MEMBER_CALL_EXPR": "member_call",
    "BINARY_OPERATOR": "binary",
    "UNARY_OPERATOR": "unary",
    "DECL_REF_EXPR": "name",
    "MEMBER_REF_EXPR": "member",
    "INTEGER_LITERAL": "int",
    "STRING_LITERAL": "string",
    "FLOATING_LITERAL": "float",
    "CXX_BOOL_LITERAL_EXPR": "bool",
    "UNEXPOSED_EXPR": "group",
    "PAREN_EXPR": "group",
}

WRAPPER_KINDS = {
    "UNEXPOSED_EXPR",
    "PAREN_EXPR",
    "IMPLICIT_CAST_EXPR",
    "CSTYLE_CAST_EXPR",
    "CXX_STATIC_CAST_EXPR",
}

CONTROL_KINDS = {
    "IF_STMT",
    "FOR_STMT",
    "WHILE_STMT",
    "CXX_FOR_RANGE_STMT",
}


def detect_resource_dir() -> Optional[str]:
    clang_candidates = []
    libclang_path = os.environ.get("LIBCLANG_PATH")
    if libclang_path:
        prefix = os.path.realpath(os.path.join(os.path.dirname(libclang_path), ".."))
        clang_candidates.append(os.path.join(prefix, "bin", "clang"))
    clang_candidates.append("clang")

    for clang_bin in clang_candidates:
        try:
            resource_dir = subprocess.check_output(
                [clang_bin, "-print-resource-dir"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if resource_dir:
                return resource_dir
        except Exception:
            continue
    return None


def cursor_extent(cursor):
    start = cursor.extent.start
    end = cursor.extent.end
    return {
        "start": {"line": start.line, "col": start.column},
        "end": {"line": end.line, "col": end.column},
    }


def build_tree(cursor, main_file: Optional[str] = None):
    type_spelling = ""
    try:
        if cursor.type is not None:
            type_spelling = cursor.type.spelling or ""
    except Exception:
        type_spelling = ""

    is_definition = False
    try:
        is_definition = bool(cursor.is_definition())
    except Exception:
        is_definition = False

    node = Node(
        kind=str(cursor.kind).split('.')[-1],
        spelling=cursor.spelling or cursor.displayname or "",
        displayname=cursor.displayname or cursor.spelling or "",
        type_spelling=type_spelling,
        is_definition=is_definition,
        start=cursor_extent(cursor)["start"],
        end=cursor_extent(cursor)["end"],
        children=[],
    )
    for c in cursor.get_children():
        # skip implicit or builtin nodes to reduce noise
        if c.location.file is None:
            continue
        child_file = os.path.realpath(str(c.location.file))
        if main_file and child_file != main_file:
            continue
        node.children.append(build_tree(c, main_file))
    return node


def node_to_dict(node: Node):
    return {
        "kind": node.kind,
        "spelling": node.spelling,
        "displayname": node.displayname,
        "type_spelling": node.type_spelling,
        "is_definition": node.is_definition,
        "start": node.start,
        "end": node.end,
        "children": [node_to_dict(c) for c in node.children],
    }


def kind_label(kind: str) -> str:
    return KIND_LABELS.get(kind, kind.lower())


def normalize_type(type_spelling: str) -> str:
    if not type_spelling:
        return "auto"
    normalized = " ".join(type_spelling.split())
    normalized = normalized.replace(" &", "&").replace("& ", "&")
    normalized = normalized.replace(" *", "*").replace("* ", "*")
    return normalized


def iter_descendants(node: Node):
    for child in node.children:
        yield child
        yield from iter_descendants(child)


def ordered_unique(values: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def first_child(node: Node, *kinds: str):
    for child in node.children:
        if child.kind in kinds:
            return child
    return None


def unwrap_kind(node: Optional[Node]) -> str:
    current = node
    while current and current.kind in WRAPPER_KINDS and current.children:
        current = current.children[0]
    if not current:
        return "none"
    return kind_label(current.kind)


def count_descendants(node: Node, kind: str) -> int:
    return sum(1 for child in iter_descendants(node) if child.kind == kind)


def semantic_sections(node: Node) -> List[str]:
    if node.kind == "FUNCTION_DECL":
        sections = ["declaration", "callable"]
        if count_descendants(node, "RETURN_STMT"):
            sections.append("return")
        if any(child.kind in CONTROL_KINDS for child in iter_descendants(node)):
            sections.append("control-flow")
        return sections
    if node.kind == "IF_STMT":
        return ["control-flow", "branching"]
    if node.kind in {"FOR_STMT", "WHILE_STMT", "CXX_FOR_RANGE_STMT"}:
        return ["control-flow", "loop"]
    if node.kind == "RETURN_STMT":
        return ["control-flow", "return"]
    if node.kind == "VAR_DECL":
        sections = ["binding"]
        if node.children:
            sections.append("initialization")
        return sections
    if node.kind in {"CALL_EXPR", "CXX_MEMBER_CALL_EXPR"}:
        return ["call", "data-flow"]
    if node.kind == "BINARY_OPERATOR":
        return ["expression", "data-flow"]
    return ["syntax"]


def semantic_signature(node: Node) -> str:
    if node.kind == "FUNCTION_DECL":
        param_count = sum(1 for child in node.children if child.kind == "PARM_DECL")
        body = first_child(node, "COMPOUND_STMT")
        descendants = iter_descendants(body) if body else []
        controls = ordered_unique(
            kind_label(child.kind)
            for child in descendants if child.kind in CONTROL_KINDS
        )
        return (
            f"function$arity={param_count}"
            f"|body={'compound' if body else 'declaration'}"
            f"|controls={'+'.join(controls) if controls else 'none'}"
            f"|returns={count_descendants(node, 'RETURN_STMT')}"
        )
    if node.kind == "IF_STMT":
        branches = [child for child in node.children if child.kind.endswith("_STMT")]
        return (
            f"if|condition={unwrap_kind(node.children[0] if node.children else None)}"
            f"|else={'yes' if len(branches) > 1 else 'no'}"
        )
    if node.kind == "FOR_STMT":
        labels = [kind_label(child.kind) for child in node.children]
        return (
            f"for|init={labels[0] if len(labels) > 0 else 'none'}"
            f"|cond={labels[1] if len(labels) > 1 else 'none'}"
            f"|step={labels[2] if len(labels) > 2 else 'none'}"
            f"|body={labels[-1] if labels else 'none'}"
        )
    if node.kind == "WHILE_STMT":
        labels = [kind_label(child.kind) for child in node.children]
        return (
            f"while|condition={labels[0] if len(labels) > 0 else 'none'}"
            f"|body={labels[-1] if labels else 'none'}"
        )
    if node.kind == "CXX_FOR_RANGE_STMT":
        labels = [kind_label(child.kind) for child in node.children]
        return (
            f"for_range|range={labels[0] if len(labels) > 0 else 'none'}"
            f"|body={labels[-1] if labels else 'none'}"
        )
    if node.kind == "RETURN_STMT":
        return f"return|value={unwrap_kind(node.children[0] if node.children else None)}"
    if node.kind == "VAR_DECL":
        return (
            f"var|type={normalize_type(node.type_spelling)}"
            f"|init={'yes' if node.children else 'no'}"
        )
    if node.kind in {"CALL_EXPR", "CXX_MEMBER_CALL_EXPR"}:
        arg_count = max(len(node.children) - 1, 0)
        callee = unwrap_kind(node.children[0] if node.children else None)
        return (
            f"call|callee={callee}"
            f"|args={arg_count}"
            f"|result={normalize_type(node.type_spelling)}"
        )
    if node.kind == "BINARY_OPERATOR":
        lhs = unwrap_kind(node.children[0] if node.children else None)
        rhs = unwrap_kind(node.children[1] if len(node.children) > 1 else None)
        return (
            f"binary|lhs={lhs}"
            f"|rhs={rhs}"
            f"|result={normalize_type(node.type_spelling)}"
        )
    return f"{kind_label(node.kind)}|children={len(node.children)}"


def grammar_fingerprint(node: Node) -> str:
    if node.kind == "FUNCTION_DECL":
        params = ",".join("param" for child in node.children if child.kind == "PARM_DECL") or "none"
        body = first_child(node, "COMPOUND_STMT")
        body_children = ",".join(kind_label(child.kind) for child in body.children) if body else "none"
        return f"function({params})->{'compound' if body else 'declaration'}[{body_children}]"
    if node.kind == "IF_STMT":
        condition = unwrap_kind(node.children[0] if node.children else None)
        branches = [kind_label(child.kind) for child in node.children if child.kind.endswith("_STMT")]
        return f"if({condition})->{'|'.join(branches) if branches else 'none'}"
    if node.kind in {"FOR_STMT", "WHILE_STMT", "CXX_FOR_RANGE_STMT"}:
        parts = ",".join(kind_label(child.kind) for child in node.children) or "empty"
        return f"{kind_label(node.kind)}({parts})"
    if node.kind == "RETURN_STMT":
        return f"return({unwrap_kind(node.children[0] if node.children else None)})"
    if node.kind == "VAR_DECL":
        init = unwrap_kind(node.children[0] if node.children else None)
        return f"var({init})"
    if node.kind in {"CALL_EXPR", "CXX_MEMBER_CALL_EXPR"}:
        parts = ",".join(unwrap_kind(child) for child in node.children) or "empty"
        return f"call({parts})"
    if node.kind == "BINARY_OPERATOR":
        lhs = unwrap_kind(node.children[0] if node.children else None)
        rhs = unwrap_kind(node.children[1] if len(node.children) > 1 else None)
        return f"binary({lhs},{rhs})"
    parts = ",".join(kind_label(child.kind) for child in node.children) or "empty"
    return f"{kind_label(node.kind)}({parts})"


def infer_regions(root: Node):
    """Infer regions with both semantic and grammar fingerprints."""
    regions = []

    def visit(n: Node, parent_kind=None):
        if n.kind == "FUNCTION_DECL":
            region = {
                "type": "function",
                "name": n.spelling,
                "span": {"start": n.start, "end": n.end},
                "semantic_signature": semantic_signature(n),
                "grammar_fingerprint": grammar_fingerprint(n),
                "semantic_sections": semantic_sections(n),
                "blocks": [],
            }
            # entry block from children CompoundStmt
            for c in n.children:
                if c.kind == "COMPOUND_STMT":
                    region["blocks"].append(infer_block(c))
            regions.append(region)
        else:
            for c in n.children:
                visit(c, n.kind)

    def infer_block(n: Node):
        block = {
            "kind": n.kind,
            "span": {"start": n.start, "end": n.end},
            "semantic_signature": semantic_signature(n),
            "grammar_fingerprint": grammar_fingerprint(n),
            "semantic_sections": semantic_sections(n),
            "children": [],
        }
        for c in n.children:
            if c.kind in ("IF_STMT", "FOR_STMT", "WHILE_STMT"):
                block["children"].append({
                    "control": c.kind,
                    "span": {"start": c.start, "end": c.end},
                    "semantic_signature": semantic_signature(c),
                    "grammar_fingerprint": grammar_fingerprint(c),
                    "semantic_sections": semantic_sections(c),
                })
            elif c.kind == "COMPOUND_STMT":
                block["children"].append(infer_block(c))
            else:
                block["children"].append({
                    "kind": c.kind,
                    "spelling": c.spelling,
                    "semantic_signature": semantic_signature(c),
                    "grammar_fingerprint": grammar_fingerprint(c),
                })
        return block

    visit(root)
    return regions


def parse_file(path: str, clang_args: List[str]):
    resolved_args = ["-x", "c++", *clang_args]
    if "-resource-dir" not in resolved_args:
        resource_dir = detect_resource_dir()
        if resource_dir:
            resolved_args.extend(["-resource-dir", resource_dir])
    if sys.platform == "darwin" and "-isysroot" not in resolved_args:
        try:
            sdk_path = subprocess.check_output(
                ["xcrun", "--show-sdk-path"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if sdk_path:
                resolved_args.extend(["-isysroot", sdk_path])
        except Exception:
            pass

    index = cindex.Index.create()
    try:
        tu = index.parse(path, args=resolved_args)
    except Exception as e:
        print(f"ERROR parsing {path}: {e}", file=sys.stderr)
        raise
    
    if not tu:
        raise RuntimeError(f"Failed to parse {path}")
    
    has_errors = False
    if tu.diagnostics:
        for d in tu.diagnostics:
            if d.severity >= cindex.Diagnostic.Error:
                has_errors = True
            print(f"diag: {d.severity} {d.spelling}", file=sys.stderr)
    
    if has_errors:
        print(f"WARNING: {path} has parse errors, results may be incomplete", file=sys.stderr)
    
    root = build_tree(tu.cursor, os.path.realpath(path))
    return root


def main(argv):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--clang-args", nargs="*", default=[], help="Args passed to clang parser (e.g. -std=c++20)")
    args, unknown = parser.parse_known_args(argv)

    clang_args = args.clang_args or unknown
    if clang_args and clang_args[0] == "--":
        clang_args = clang_args[1:]

    root = parse_file(args.input, clang_args)
    regions = infer_regions(root)

    out = {
        "file": args.input,
        "regions": regions,
        "ast": node_to_dict(root),
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main(sys.argv[1:])
