import json
from pathlib import Path
import sys

try:
    from clang import cindex
    from clang_support import configure_libclang
    configure_libclang(cindex)
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))
from emit_mappings import emit_mapping_candidates
from parse_and_infer import parse_file
from batch_emit_mappings import aggregate_mappings


def load_mappings(sample_name: str):
    src = Path(__file__).parent.parent / "samples" / sample_name
    root = parse_file(str(src), ["-std=c++20"])
    return src, emit_mapping_candidates(root, str(src))


def test_emit_mappings_schema():
    src, mappings = load_mappings("sample1.cpp")

    assert mappings, "Should emit at least one mapping"

    required_fields = {
        "id",
        "source_sample",
        "ast_kind",
        "son_node",
        "mlir_template",
        "pattern",
        "semantic_signature",
        "grammar_fingerprint",
        "semantic_sections",
        "semantics_source",
        "grammar_source",
        "confidence",
        "examples",
        "notes",
    }

    for mapping in mappings:
        assert required_fields.issubset(mapping.keys()), f"Missing fields in {mapping.get('id')}"
        assert mapping["source_sample"]["file"] == str(src)
        assert mapping["semantics_source"] == "clang"
        assert mapping["grammar_source"] == "normalized_ast_shape"
        assert 0.0 <= mapping["confidence"] <= 1.0


def test_emit_mappings_attach_semantic_and_grammar_signatures():
    _, mappings = load_mappings("sample1.cpp")

    by_kind = {}
    for mapping in mappings:
        by_kind.setdefault(mapping["ast_kind"], []).append(mapping)

    function_mapping = by_kind["FunctionDecl"][0]
    assert function_mapping["semantic_signature"] == "function$arity=2|body=compound|controls=if+for|returns=1"
    assert function_mapping["semantic_sections"] == ["declaration", "callable", "return", "control-flow"]

    if_mapping = by_kind["IfStmt"][0]
    assert if_mapping["semantic_signature"] == "if|condition=binary|else=no"
    assert if_mapping["grammar_fingerprint"].startswith("if(binary)->")

    for_mapping = by_kind["ForStmt"][0]
    assert for_mapping["semantic_sections"] == ["control-flow", "loop"]
    assert for_mapping["grammar_fingerprint"].startswith("for(")

    return_mapping = by_kind["ReturnStmt"][0]
    assert return_mapping["semantic_signature"] == "return|value=name"


def test_aggregate_mappings_tracks_support_counts(tmp_path):
    for sample_name in ("sample1.cpp", "sample_son.cpp"):
        src, mappings = load_mappings(sample_name)
        output = tmp_path / f"{src.stem}_mappings.json"
        output.write_text(json.dumps({"mappings": mappings}, indent=2))

    aggregated = aggregate_mappings(tmp_path)

    return_mapping = next(
        mapping
        for mapping in aggregated
        if mapping["ast_kind"] == "ReturnStmt"
        and mapping["semantic_signature"] == "return|value=name"
    )
    assert return_mapping["support_count"] >= 2
    assert len(return_mapping["supporting_files"]) >= 2
