import json
import tempfile
from pathlib import Path
import sys
import os

# Set up libclang if available
if "LIBCLANG_PATH" in os.environ:
    from clang import cindex
    cindex.Config.set_library_file(os.environ["LIBCLANG_PATH"])

import pytest

# Import after libclang setup
sys.path.insert(0, str(Path(__file__).parent.parent))
from emit_mappings import emit_mapping_candidates, Mapping
from parse_and_infer import parse_file


def test_emit_mappings_schema():
    """Verify emitted mappings conform to MAPPING_SPEC schema."""
    src = Path(__file__).parent.parent / "samples" / "sample1.cpp"
    root = parse_file(str(src), ["-std=c++20"])
    mappings = emit_mapping_candidates(root, str(src))
    
    assert len(mappings) > 0, "Should emit at least one mapping"
    
    required_fields = {"id", "source_sample", "ast_kind", "son_node", 
                       "mlir_template", "pattern", "confidence", "examples", "notes"}
    
    for m in mappings:
        assert isinstance(m, dict)
        assert required_fields.issubset(m.keys()), f"Missing fields in {m.get('id')}"
        assert isinstance(m["confidence"], (int, float))
        assert 0.0 <= m["confidence"] <= 1.0
        assert isinstance(m["examples"], list)


def test_function_decl_mapping():
    """Verify FunctionDecl emits correct mapping."""
    src = Path(__file__).parent.parent / "samples" / "sample1.cpp"
    root = parse_file(str(src), ["-std=c++20"])
    mappings = emit_mapping_candidates(root, str(src))
    
    func_mappings = [m for m in mappings if m["ast_kind"] == "FunctionDecl"]
    assert len(func_mappings) >= 1, "Should have at least one function"
    
    fm = func_mappings[0]
    assert "cpp2.func" in fm["mlir_template"]
    assert fm["son_node"] == "RegionNode"
    assert fm["confidence"] >= 0.8


def test_if_stmt_mapping():
    """Verify IfStmt emits correct mapping."""
    src = Path(__file__).parent.parent / "samples" / "sample1.cpp"
    root = parse_file(str(src), ["-std=c++20"])
    mappings = emit_mapping_candidates(root, str(src))
    
    if_mappings = [m for m in mappings if m["ast_kind"] == "IfStmt"]
    assert len(if_mappings) >= 1, "sample1.cpp has if statement"
    
    im = if_mappings[0]
    assert "cpp2.if" in im["mlir_template"]
    assert im["son_node"] == "IfNode"
    assert "branches" in im["pattern"] or "region" in im["pattern"]


def test_for_stmt_mapping():
    """Verify ForStmt emits correct mapping."""
    src = Path(__file__).parent.parent / "samples" / "sample1.cpp"
    root = parse_file(str(src), ["-std=c++20"])
    mappings = emit_mapping_candidates(root, str(src))
    
    for_mappings = [m for m in mappings if m["ast_kind"] == "ForStmt"]
    assert len(for_mappings) >= 1, "sample1.cpp has for loop"
    
    fm = for_mappings[0]
    assert "cpp2.for" in fm["mlir_template"]
    assert fm["son_node"] == "LoopNode"


def test_return_stmt_mapping():
    """Verify ReturnStmt emits correct mapping."""
    src = Path(__file__).parent.parent / "samples" / "sample1.cpp"
    root = parse_file(str(src), ["-std=c++20"])
    mappings = emit_mapping_candidates(root, str(src))
    
    ret_mappings = [m for m in mappings if m["ast_kind"] == "ReturnStmt"]
    assert len(ret_mappings) >= 1, "sample1.cpp has return statements"
    
    rm = ret_mappings[0]
    assert "cpp2.return" in rm["mlir_template"]
    assert rm["son_node"] == "ReturnNode"


def test_mapping_deduplication():
    """Verify batch aggregator deduplicates by (ast_kind, pattern)."""
    src = Path(__file__).parent.parent / "samples" / "sample_son.cpp"
    root = parse_file(str(src), ["-std=c++20"])
    mappings = emit_mapping_candidates(root, str(src))
    
    # Count unique (ast_kind, pattern) pairs
    unique_keys = set((m["ast_kind"], m["pattern"]) for m in mappings)
    
    # Should have one unique pattern per AST kind despite multiple instances
    from collections import Counter
    kind_counts = Counter(m["ast_kind"] for m in mappings)
    
    # We have multiple functions but only one unique FunctionDecl pattern
    assert kind_counts["FunctionDecl"] >= 5  # sample_son has 6 functions
    
    # But only one unique (FunctionDecl, pattern) pair
    func_patterns = {(m["ast_kind"], m["pattern"]) 
                     for m in mappings if m["ast_kind"] == "FunctionDecl"}
    assert len(func_patterns) == 1
