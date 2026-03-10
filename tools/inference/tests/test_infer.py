import json
from pathlib import Path

from tools.inference import parse_and_infer as pai


def test_parse_and_infer_roundtrip(tmp_path):
    src = Path(__file__).parent.parent / "samples" / "sample1.cpp"
    out = tmp_path / "out.json"
    # Use a common clang arg
    pai.main(["-i", str(src), "-o", str(out), "--clang-args", "-std=c++20"])
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["file"].endswith("sample1.cpp")
    regions = data["regions"]
    add_region = next(
        r for r in regions
        if r.get("type") == "function" and r.get("name") == "add"
    )
    assert add_region["semantic_signature"] == "function$arity=2|body=compound|controls=if+for|returns=1"
    assert add_region["grammar_fingerprint"] == "function(param,param)->compound[decl,if,for,return]"
    assert add_region["semantic_sections"] == ["declaration", "callable", "return", "control-flow"]
