# Full Corpus Transpile Validation - Match Cppfront Output

**Date**: 2025-12-30
**Track ID**: corpus_validation_20251230
**Corpus**: 189 .cpp2 files from third_party/cppfront/regression-tests
**Goal**: 100% transpile accuracy matching cppfront reference output

## Objective

Achieve full completion-level transpilation matching cppfront's reference output for all 189 corpus files. Process files sequentially in sorted order, recording results, fixing errors via git worktree merges, and validating against cppfront baseline.

## Success Criteria

**FULL COMPLETION LEVEL**: Cppfort output must be semantically and structurally equivalent to cppfront output.

- **Transpile success rate**: 189/189 files (100%)
- **Semantic loss**: <0.05 average across corpus (near-zero)
- **Parameter semantics**: 100% preservation (inout → T&, out → T&, etc.)
- **Mixed-mode support**: 50/50 mixed files transpiling successfully
- **Corpus preservation**: Zero modifications to corpus/inputs/*.cpp2 (whitespace excepted)

## Validation Approach

### Single Phase Workflow

**Phase**: Full Corpus Validation and Repair

1. **Setup**: Create git worktree `corpus-validation` from current master
2. **Sequential Processing**: For each of 189 files in sorted order:
   - Transpile with cppfort: `corpus/inputs/FILE.cpp2 → /tmp/cppfort-FILE.cpp`
   - Compare against reference: `corpus/reference/FILE.cpp` (cppfront output)
   - Calculate semantic loss score
   - Record results (pass/fail, loss score, errors, timing)
   - If errors: fix transpiler, rebuild, re-test
3. **Validation**: Verify all 189 files achieve <0.05 semantic loss
4. **Commit**: Merge worktree with fixes back to master
5. **Cleanup**: Remove worktree after successful merge

### Git Worktree Strategy

```bash
# Create isolated worktree
git worktree add ../cppfort-corpus-validation -b corpus/validation-20251230

# Work in isolation
cd ../cppfort-corpus-validation

# Process all 189 files sequentially
for file in corpus/inputs/*.cpp2; do
  ./build/src/cppfort "$file" /tmp/output.cpp
  # ... validate, record, fix if needed ...
done

# Commit all fixes
git add -A
git commit -m "fix: Achieve full corpus transpile parity with cppfront"

# Merge back to master
cd /Users/jim/work/cppfort
git merge --no-ff corpus/validation-20251230
git notes add -m "corpus validation: 189/189 files, <0.05 avg loss"

# Cleanup worktree
git worktree remove ../cppfort-corpus-validation
git branch -d corpus/validation-20251230
```

## Current Blockers (from regression_corpus_20251230)

### P0: Parameter Semantics Lost

**Issue**: `inout` parameters transpile as by-value instead of by-reference

**Example**:
```cpp2
decorate: (inout s: std::string) = { s = "[" + s + "]"; }
```

**Cppfort (WRONG)**:
```cpp
void decorate(std::string s) { ... }  // ❌ by-value
```

**Cppfront (CORRECT)**:
```cpp
auto decorate(std::string& s) -> void { ... }  // ✓ by-reference
```

**Fix Required**: Map parameter qualifiers correctly
- `in` → `const T&`
- `out` → `T&` (write-before-return enforced)
- `inout` → `T&`
- `move` → `T&&`
- `forward` → template forwarding reference

### P1: Mixed-Mode C++1 Syntax Support

**Issue**: Parser rejects C++1 syntax in mixed-mode files (50/189 blocked)

**Example**:
```cpp
auto main() -> int {  // ← C++1 syntax
    // ...
}
```

**Error**: Parser expects Cpp2 syntax, fails on `auto name() -> type`

**Fix Required**: Detect and pass-through C++1 syntax unchanged, only transpile Cpp2 portions

## File Categories

### pure2 Files (139 files)
- Pure Cpp2 syntax only
- Currently: Some transpile (e.g., pure2-hello.cpp2) but with semantic loss
- Target: 139/139 passing with <0.05 loss

### mixed Files (50 files)
- Mixed Cpp2 + C++1 syntax
- Currently: BLOCKED (parser fails)
- Target: 50/50 passing with <0.05 loss

## Semantic Loss Scoring

### Metrics

For each file, compute:
1. **Structural distance**: AST isomorph pattern matching
2. **Type distance**: Type signature preservation
3. **Operation distance**: Operator/call semantics preservation
4. **Combined loss**: Weighted average

### Baseline Comparison

**Reference**: `corpus/reference/FILE.cpp` (cppfront output)
**Candidate**: `/tmp/cppfort-FILE.cpp` (cppfort output)

```bash
# Generate AST and isomorphs
clang++ -std=c++20 -Xclang -ast-dump -fsyntax-only /tmp/cppfort-FILE.cpp > /tmp/candidate.ast.txt
python3 tools/extract_ast_isomorphs.py --ast /tmp/candidate.ast.txt --output /tmp/candidate.isomorph.json
python3 tools/tag_mlir_regions.py --isomorphs /tmp/candidate.isomorph.json --output /tmp/candidate.tagged.json

# Score against reference
python3 tools/score_semantic_loss.py \
  --reference corpus/tagged/FILE.tagged.json \
  --candidate /tmp/candidate.tagged.json \
  --output /tmp/FILE.loss.json
```

**Target**: Combined loss <0.05 for all 189 files

## Constraints

1. **Corpus Preservation**: No modifications to `corpus/inputs/*.cpp2` files (whitespace excepted)
2. **Sequential Processing**: Files processed in sorted alphabetical order
3. **Full Completion**: All 189 files must transpile successfully
4. **Accuracy Target**: Match cppfront output semantically and structurally
5. **Worktree Isolation**: All work done in isolated git worktree
6. **Clean Merge**: Single merge commit with all fixes after validation
7. **Worktree Cleanup**: Remove worktree after successful merge

## Infrastructure

### Existing Assets

- ✅ `build/src/cppfort` - Transpiler binary
- ✅ `third_party/cppfront/source/cppfront` - Reference transpiler
- ✅ `corpus/reference/*.cpp` - 158 reference outputs from cppfront
- ✅ `corpus/tagged/*.tagged.json` - Precomputed reference isomorphs
- ✅ `tools/score_semantic_loss.py` - Loss scoring framework

### Test Commands

```bash
# Transpile single file
./build/src/cppfort corpus/inputs/pure2-hello.cpp2 /tmp/output.cpp

# Run full corpus validation
./build/tests/cppfront_full_regression --test-dir third_party/cppfront/regression-tests

# Generate loss score for single file
./tools/score_semantic_loss.py \
  --reference corpus/tagged/pure2-hello.tagged.json \
  --candidate /tmp/candidate.tagged.json
```

## Expected Deliverables

1. **All transpiler fixes**: Merged to master via single commit
2. **Validation report**: `corpus_validation_report.md` with per-file results
3. **Loss score matrix**: CSV with all 189 files and their loss scores
4. **Pass/fail summary**: 189/189 PASS, 0/189 FAIL
5. **Clean git history**: Worktree merged and removed

## Timeline

**Single Phase**: Complete all 189 files

**Estimated Duration**: Depends on blocker fixes
- If parameter semantics fix: ~2-4 hours for full validation
- If mixed-mode parser fix required: +4-8 hours
- Total: 6-12 hours

## Risk Assessment

- **High Risk**: Blockers (P0 parameter semantics, P1 mixed-mode) must be fixed first
- **Medium Risk**: Semantic loss scoring may reveal additional issues
- **Low Risk**: Corpus infrastructure complete, validation straightforward once blockers resolved
