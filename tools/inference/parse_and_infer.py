#!/usr/bin/env python3
"""Prototype: parse a C/C++ file with libclang and infer MLIR-like regions/blocks.

Usage:
  python parse_and_infer.py -i sample.cpp -o mapping.json -- -std=c++20

This is a small prototype that:
 - parses the source with clang.cindex
 - builds a tree of extents (start/end positions)
 - runs simple heuristics to infer 'regions' and 'blocks' (functions, compound stmts, control flow)
 - writes JSON output describing inferred regions

This is intentionally minimal; extend heuristics to match cppfront/cpp2 semantics.
"""

import sys
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

try:
    from clang import cindex
except Exception:
    print("ERROR: clang python bindings not found. Install with `pip install clang` and ensure libclang is available.")
    sys.exit(2)


@dataclass
class Node:
    kind: str
    spelling: str
    start: Dict[str, int]
    end: Dict[str, int]
    children: List[Any]


def cursor_extent(cursor):
    start = cursor.extent.start
    end = cursor.extent.end
    return {
        "start": {"line": start.line, "col": start.column},
        "end": {"line": end.line, "col": end.column},
    }


def build_tree(cursor):
    node = Node(
        kind=str(cursor.kind).split('.')[-1],
        spelling=cursor.spelling or cursor.displayname or "",
        start=cursor_extent(cursor)["start"],
        end=cursor_extent(cursor)["end"],
        children=[],
    )
    for c in cursor.get_children():
        # skip implicit or builtin nodes to reduce noise
        if c.location.file is None:
            continue
        node.children.append(build_tree(c))
    return node


def node_to_dict(node: Node):
    return {
        "kind": node.kind,
        "spelling": node.spelling,
        "start": node.start,
        "end": node.end,
        "children": [node_to_dict(c) for c in node.children],
    }


def infer_regions(root: Node):
    """Simple heuristics mapping AST nodes to regions/blocks.

    - `FunctionDecl` -> region with entry block
    - `CompoundStmt` -> block
    - `IfStmt`/`ForStmt`/`WhileStmt` -> blocks/regions for branches/loops
    """
    regions = []

    def visit(n: Node, parent_kind=None):
        if n.kind == "FUNCTION_DECL":
            region = {
                "type": "function",
                "name": n.spelling,
                "span": {"start": n.start, "end": n.end},
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
            "children": [],
        }
        for c in n.children:
            if c.kind in ("IF_STMT", "FOR_STMT", "WHILE_STMT"):
                block["children"].append({
                    "control": c.kind,
                    "span": {"start": c.start, "end": c.end},
                })
            elif c.kind == "COMPOUND_STMT":
                block["children"].append(infer_block(c))
            else:
                block["children"].append({"kind": c.kind, "spelling": c.spelling})
        return block

    visit(root)
    return regions


def parse_file(path: str, clang_args: List[str]):
    index = cindex.Index.create()
    try:
        tu = index.parse(path, args=clang_args)
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
    
    root = build_tree(tu.cursor)
    return root


def main(argv):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--clang-args", nargs="*", default=[], help="Args passed to clang parser (e.g. -std=c++20)")
    args, unknown = parser.parse_known_args(argv)

    root = parse_file(args.input, args.clang_args or unknown)
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
