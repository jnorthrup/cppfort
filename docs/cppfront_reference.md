# Using Cppfront as a Reference Implementation

## Overview

Cppfront (by Herb Sutter) is available as a reference implementation to validate cppfort's transpilation output. This provides a simpler, more direct validation without additional features.

## Setup

**Reference Implementation Location**:
- Path: `/tmp/justfoirtests/cppfront`
- Source: https://github.com/hsutter/cppfront

**Built and Ready**: cppfront binary is compiled and functional at `/tmp/justfoirtests/cppfront`

## Usage

### 1. Generate Reference Outputs

Run a single test through cppfront:
```bash
cd /tmp/justfoirtests
./cppfront -clean-cpp1 -o output.cpp /Users/jim/work/cppfort/tests/regression-tests/mixed-hello.cpp2
```

This produces transpiled C++ code that can be compared with cppfort's output.

### 2. Compare with Cppfort

```bash
# Run cppfront
/tmp/justfoirtests/cppfront -clean-cpp1 -o reference.cpp test.cpp2

# Run cppfort
/Users/jim/work/cppfort/build/src/stage0/stage0_cli transpile test.cpp2 output.cpp

# Compare
diff reference.cpp output.cpp
```

### 3. Batch Testing

Generate reference outputs for all regression tests:
```bash
/Users/jim/work/cppfort/scripts/use_cppfront_reference.sh
```

This creates `/tmp/cppfront_reference_results/` with all transpiled files.

## Limitations

**Known Issue**: Cppfort currently has a memory corruption bug in the LRU cache eviction (Test 3 of integration tests). This affects some test comparisons but does not impact:
- Packrat cache standalone tests (all passing)
- Pool standalone tests (all passing)
- Simple file transpilation (most work)

**Workaround**: Use cppfront directly for tests that don't hit the cache issue.

## Architecture Comparison

### Cppfront (Reference)
- **Approach**: Traditional compiler (Lexer → Parser → AST → Emitter)
- **Size**: ~180 lines in main driver, ~3,100 lines in cpp2util.h
- **Usage**: Simpler, more direct transpilation

### Cppfort (This Implementation)
- **Approach**: Pattern-matching with speculative parsing
- **Unique Features**:
  - Orbit-based pattern matching
  - Merkle trees for attestation
  - Consensus mechanisms
  - Packrat parser combinators
- **Usage**: More experimental features, some stability issues

## Benefits of Reference

Reasons to use cppfront as reference:

1. **Simplicity**: Known-good implementation
2. **Validation**: Compare outputs line-by-line
3. **Debugging**: When cppfort has issues
4. **Regression**: Generate expected test outputs
5. **Verification**: Ensure semantic correctness

## Documentation

For full cppfront documentation, see:
- https://hsutter.github.io/cppfront/
- https://github.com/hsutter/cppfront
