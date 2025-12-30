# Implementation Plan: Regression Test Corpus Semantic Preservation

## Phase 1: Core Fixes (P0 Blockers)

### Fix Parameter Semantics
- [ ] Implement parameter qualifier→C++ type mapping in code generator
  - Map `in` → `const T&`
  - Map `out` → `T&`
  - Map `inout` → `T&`
  - Map `move` → `T&&`
  - Map `forward` → template forwarding reference `T&&`
- [ ] Add tests for all parameter qualifiers (in/out/inout/move/forward)
- [ ] Re-test pure2-hello.cpp2 to verify `decorate` generates `std::string&`
- [ ] Verify semantic loss reduction from 1.0 baseline
- [ ] Update code generator to emit correct reference types

### Add Mixed-Mode C++1 Support
- [ ] Update parser to detect C++1 function syntax patterns
  - Detect `auto name(...) -> type` (trailing return)
  - Detect `type name(...)` (standard function)
- [ ] Implement C++1 passthrough mechanism (preserve unchanged)
- [ ] Add tests for C++1 detection and passthrough
- [ ] Test on mixed-hello.cpp2
- [ ] Run full mixed category (50 tests) to verify parser handles mixed syntax

## Phase 2: Corpus Processing

### Generate Candidate Outputs
- [ ] Transpile all 189 .cpp2 files with cppfort
- [ ] Generate Clang AST dumps for all cppfort outputs
- [ ] Extract isomorphs from candidate ASTs
- [ ] Tag candidate isomorphs with MLIR region types
- [ ] Verify JSON output structure matches scoring tool expectations

### Run Semantic Loss Scoring
- [ ] Score all 189 files against cppfront reference
- [ ] Compute average corpus loss
- [ ] Identify high-loss files (>0.5) requiring fixes
- [ ] Identify zero-loss files (<0.01) as success cases
- [ ] Generate loss distribution report

## Phase 3: Code Quality Improvements

### Reduce Output Noise
- [ ] Eliminate extra nested blocks in generated code
- [ ] Remove unnecessary parentheses around expressions
- [ ] Add #line directives for source mapping (debugging support)
- [ ] Improve output readability (formatting, spacing)
- [ ] Test that changes don't affect semantic correctness

### Runtime Library Support (Optional)
- [ ] Design cpp2util.h equivalent for cppfort
- [ ] Implement contract checking infrastructure
- [ ] Add safety bounds checking helpers
- [ ] Integrate runtime library with generated code
- [ ] Test contract enforcement in pure2 files

## Phase 4: Validation and Documentation

### Measure Results
- [ ] Verify pure2 tests: target 139/139 passing
- [ ] Verify mixed tests: target 50/50 passing
- [ ] Verify parameter semantics: 100% correct
- [ ] Verify average corpus loss: <0.15
- [ ] Verify pure2-hello.cpp2 loss: <0.05
- [ ] Document semantic preservation improvements

### Update Documentation
- [ ] Update product.md with corpus test results
- [ ] Update tech-stack.md with new capabilities
- [ ] Document parameter qualifier semantics
- [ ] Create semantic preservation report
