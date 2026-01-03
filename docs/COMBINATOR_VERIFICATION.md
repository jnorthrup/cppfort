# Combinator Verification Test Suite

## Overview

This document describes the verification test suite for the Cppfort combinators library.
These tests verify correctness, performance, and zero-copy properties of the combinator
implementation.

## Test Files

### 1. Property-Based Tests (`combinator_laws_test.cpp2`)

Verifies mathematical laws that combinators should satisfy:

| Law Category | Tests |
|--------------|-------|
| **Functor Laws** | Identity (`map(id) = id`), Composition (`map(f.g) = map(f).map(g)`) |
| **Structural Laws** | `take(0) = empty`, `take(n).take(m) = take(min(n,m))`, `skip(n).skip(m) = skip(n+m)` |
| **Filter Laws** | `filter(true) = id`, `filter(false) = empty`, `filter(p1).filter(p2) = filter(p1 && p2)` |
| **Reduction Laws** | Monoid associativity, `count = fold(0, +1)`, `any(p) = !all(!p)` |
| **Pipeline Laws** | Associativity of `|>` operator |

**Test methodology:** Uses random input generation with 10+ seeds per test for property coverage.

### 2. Benchmark Suite (`benchmark_combinators.cpp2`)

Compares combinator performance against hand-written loops:

| Operation | Target Overhead |
|-----------|-----------------|
| fold (sum) | <5% |
| filter + count | <5% |
| take(N) | <5% |
| skip(N) | <5% |
| map | <5% |
| pipeline (skip+take+map+fold) | <5% |
| find | <5% |
| all | <5% |

**Test methodology:** 
- Warmup iterations: 100
- Benchmark iterations: 10,000
- Data sizes: 1KB, 10KB

### 3. Zero-Copy Verification (`zero_copy_verification_test.cpp2`)

Verifies that combinators don't allocate memory during operation:

| Test | Verification |
|------|--------------|
| ByteBuffer slice | Pointer equality check |
| Take iterator | Reads from original memory |
| Skip iterator | Reads from original memory |
| Map iterator | No intermediate buffering |
| Filter iterator | No intermediate buffering |
| Chained operations | Only final collection allocates |
| Split | Returns views into original |
| 1MB scale test | Processes without duplication |

**Run with ASAN:** `clang++ -fsanitize=address ...` to catch memory issues.

### 4. Integration Tests (`combinator_integration_test.cpp2`)

Real-world parsing scenarios from the documentation:

| Recipe | Description |
|--------|-------------|
| HTTP Headers | Parse `Key: Value` lines with CR/LF handling |
| C Strings | Null-terminated string extraction |
| Binary Protocol | Length-prefixed message parsing |
| CSV Lines | Comma-separated value parsing with trim |
| Log Entries | `[timestamp] LEVEL: message` parsing |
| Config Files | `key=value` with comment handling |
| Word Count | Split + map + collect patterns |
| Pattern Finding | find, any, all predicates |

### 5. Corpus Test (`corpus/inputs/combinator_corpus.cpp2`)

Basic sanity tests for the transpiler's handling of combinator code:

- ByteBuffer creation and iteration
- `take`, `skip` operations
- Pipeline operator `|>`
- `map`, `filter` transformations
- `fold`, `split`, `enumerate`
- `find`, `any`, `all` predicates

## Running Tests

### Individual Test Files

```bash
# Run existing compiled tests
./tests/structural_combinators_test
./tests/transformation_combinators_test
./tests/reduction_combinators_test
./tests/pipeline_operator_test
./tests/parsing_combinators_test
```

### Full Combinator Test Suite

```bash
./tools/run_combinator_tests.sh
```

### With AddressSanitizer

```bash
# Compile with ASAN
clang++ -std=c++20 -fsanitize=address -fno-omit-frame-pointer \
    -I include tests/zero_copy_verification_test.cpp -o zero_copy_test_asan

# Run
./zero_copy_test_asan
```

### Benchmarks

```bash
# Compile with optimizations
clang++ -std=c++20 -O2 -I include tests/benchmark_combinators.cpp -o benchmark

# Run benchmarks
./benchmark
```

## Expected Results

All tests should:

1. **Pass without assertion failures**
2. **Complete within timeout** (30 seconds per test)
3. **Show <5% overhead** in benchmarks (with -O2)
4. **Report no ASAN errors** for zero-copy tests

## Verification Checklist

- [x] Property-based tests for combinator laws
- [x] Benchmark suite vs hand-written loops
- [x] Zero-copy verification (ASAN)
- [x] Integration tests with real parsers
- [x] Corpus tests for combinator compositions

## Test Coverage Summary

| Category | Files | Tests |
|----------|-------|-------|
| Unit Tests | 8 | ~150 |
| Property Tests | 1 | 17 |
| Benchmarks | 1 | 8 |
| Zero-Copy | 1 | 14 |
| Integration | 1 | 18 |
| Corpus | 1 | 11 |
| **Total** | **13** | **~218** |
