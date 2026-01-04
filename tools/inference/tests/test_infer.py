import os
import json
import tempfile
from pathlib import Path

import pytest

from tools.inference import parse_and_infer as pai


def test_parse_and_infer_roundtrip(tmp_path):
    src = Path(__file__).parent.parent / "samples" / "sample1.cpp"
    out = tmp_path / "out.json"
    # Use a common clang arg
    pai.main(["-i", str(src), "-o", str(out), "--clang-args", "-std=c++20"])
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["file"].endswith("sample1.cpp")
    # One function region found
    regions = data["regions"]
    assert any(r.get("type") == "function" and r.get("name") == "add" for r in regions)
