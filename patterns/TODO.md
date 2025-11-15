# TODO: JSON-Based Pattern Loading

## ✅ COMPLETED: YAML → JSON Conversion (2025-11-14)

**Action:** Used python3.11 with PyYAML to convert patterns/cppfort_core_patterns.yaml → patterns/cppfort_core_patterns.json

**Result:**
- 19 patterns converted successfully
- Alternating anchors preserved: `[": "]`, `["(", ")"]`, `[":", "="]`
- Evidence types preserved: `["identifier", "type"]`, etc.
- Transformation templates preserved: `{"2": "$2 $1"}`, etc.
- YAML file deleted (single source of truth = JSON)

**Verification:**
```bash
python3.11 -c "import yaml; import json; data = yaml.safe_load_all(...); print(json.dumps(...))"
```

**JSON file:** patterns/cppfort_core_patterns.json

## 🚧 IN PROGRESS: JSON Scanner Integration

**Next Step:** Integrate/load JSON patterns instead of YAML

**Files to create:**
- src/stage0/json_pattern_loader.h
- src/stage0/json_pattern_loader.cpp

**Implementation plan:**
1. Use minimal JSON parser (nlohmann/json single header or custom)
2. Load patterns from JSON array
3. Extract: name, alternating_anchors, evidence_types, transformation_templates
4. Build PatternData structures (same as YAML loader, but correct)
5. Run test: verify templates load correctly

**Test case:**
```cpp
JsonPatternLoader loader;
auto patterns = loader.load("patterns/cppfort_core_patterns.json");
// Expected: patterns[0].templates[2] == "$2 $1"
// Actual before fix: "" (empty string)
```

## 🎯 BLOCKING ISSUE: Templates Empty in Current Loader

**Root cause:** YAML parser was ad-hoc string parsing, failed to extract nested structures

**Solution:** JSON loader will directly access:
```cpp
pattern["transformation_templates"]["2"].as_string()  // = "$2 $1"
```

## Phase 2: N-Way Semantic Transformation

**After JSON loading works:**
- Implement extract_alternating_segments() using JSON anchors
- Implement substitute_template() with evidence spans
- Test CPP2 → C++ transformation
- Add round-trip validation

## Phase 3: Regression Test Integration

**Once N-way works:**
- Run on pure2-hello.cpp2
- Run on mixed-function-expression.cpp2
- Goal: 0/192 → 192/192 passing

## Cleanup

- [x] Delete patterns/cppfort_core_patterns.yaml
- [ ] Delete src/stage0/pattern_loader.cpp (YAML loader)
- [ ] Delete csv_pattern_loader.* (CSV experiment)
- [ ] Keep only: json_pattern_loader

## Files Changed

**Added:**
- patterns/cppfort_core_patterns.json (19 patterns, full templates)

**Deleted:**
- patterns/cppfort_core_patterns.yaml (unreliable ad-hoc parser)

**To be deleted after JSON integration:**
- patterns/bnfc_patterns.csv (incomplete templates)
- patterns/bnfc_patterns_clean.csv
- patterns/bnfc_patterns_final.csv
- src/stage0/csv_pattern_loader.*

## Next Action

1. Create minimal json_pattern_loader.h/cpp
2. Use nlohmann/json or custom parser
3. Load patterns from patterns/cppfort_core_patterns.json
4. Verify templates load: should see "$2 $1", not ""
5. Run simple transformation test

**Owner:** jim
**Priority:** HIGH (blocks all other work)
**Estimated time:** 2-3 days
