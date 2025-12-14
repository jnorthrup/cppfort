# Regression Testing System

## Overview

Cppfort includes a comprehensive regression testing system that:

1. **Validates integrity** of test files using SHA256 checksums
2. **Generates corpus** of transpiled outputs from both cppfront and cppfort
3. **Performs isomorphic comparison** of outputs (semantic equivalence, not textual)

## Quick Start

```bash
# Build cppfort
cmake -G Ninja -B build
ninja -C build

# Run regression tests
./scripts/run_regression.sh

# Import upstream cppfront regression tests (if needed)
```bash
./scripts/import_cppfront_tests.sh --clone
```
```

## Architecture

### Components

**`regression_runner`** - Main test executable that:
- Discovers `.cpp2` test files in cppfront repository
- Validates SHA256 checksums for file integrity
- Runs both cppfront and cppfort transpilers
- Compares outputs isomorphically
- Generates corpus with detailed reports

**`run_regression.sh`** - Convenience script that:
- Locates cppfront binary and tests
- Sets up environment
- Executes regression_runner
- Displays summary statistics

### Isomorphic Comparison

Unlike simple text diffing, isomorphic comparison:

1. **Normalizes whitespace** - Collapses multiple spaces, ignores formatting
2. **Strips comments** - Removes comment-only differences
3. **Parses structure** - Builds simplified AST from both outputs
4. **Compares semantics** - Checks for structural equivalence:
   - Function signatures
   - Class definitions
   - Include directives
   - Control flow

Two outputs are **semantically equivalent** if they produce the same AST structure, even if formatting differs.

### Corpus Structure

After running, the corpus directory contains:

```
corpus/
├── sha256_database.txt      # Integrity checksums for test files
├── summary.csv               # Statistical summary
├── cppfront/                 # Reference outputs from cppfront
│   ├── test1.cpp
│   └── test2.cpp
├── cppfort/                  # Outputs from cppfort
│   ├── test1.cpp
│   └── test2.cpp
└── diffs/                    # Isomorphic comparison reports
    ├── test1.diff
    └── test2.diff
```

## Usage

### Basic Usage

```bash
./scripts/run_regression.sh
```

### Custom Paths

```bash
export CPPFRONT_TESTS="/path/to/cppfront/regression-tests"
export CPPFRONT_BIN="/path/to/cppfront"
export CORPUS_DIR="/path/to/corpus"
./scripts/run_regression.sh
```

### Direct Invocation

```bash
./build/tests/regression_runner \
    /path/to/cppfront/regression-tests \
    /path/to/cppfront \
    ./build/src/cppfort \
    ./corpus
```

## Output Analysis

### Summary CSV

Format: `source,cppfront_success,cppfort_success,semantically_equivalent,semantic_diffs`

```csv
test_file.cpp2,1,1,1,0
error_test.cpp2,0,0,N/A,N/A
feature_test.cpp2,1,1,0,3
```

Fields:
- `cppfront_success`: 1 if cppfront transpiled successfully
- `cppfort_success`: 1 if cppfort transpiled successfully
- `semantically_equivalent`: 1 if outputs are isomorphic
- `semantic_diffs`: Count of semantic differences found

### Diff Reports

Each `.diff` file contains:

```
Source: test.cpp2
Semantically equivalent: NO

Semantic differences:
  - Name mismatch: 'foo' vs 'bar'
  - Child count mismatch: 2 vs 3
```

## SHA256 Database

The `sha256_database.txt` file tracks test file integrity:

```
<hash> relative/path/to/test.cpp2
<hash> another/test.cpp2
```

On each run:
- **New files**: Added to database
- **Changed files**: Warning issued, hash updated
- **Unchanged files**: Validated silently

This ensures:
- Test files haven't been corrupted
- Changes to upstream cppfront tests are detected
- Reproducible test results

## Statistics

The runner reports:

```
=== Regression Test Summary ===
Total tests: 450
Cppfront success: 425 (94.4%)
Cppfort success: 380 (84.4%)
Both succeeded: 375 (83.3%)
Semantically equivalent: 340 (90.7% of both succeeded)
```

Metrics:
- **Total tests**: All `.cpp2` files discovered
- **Cppfront success**: cppfront transpiled without error
- **Cppfort success**: cppfort transpiled without error
- **Both succeeded**: Both transpilers succeeded
- **Semantically equivalent**: Outputs are isomorphic

## Integration with CI

### GitHub Actions

```yaml
- name: Run Regression Tests
  run: |
    ./scripts/run_regression.sh

- name: Upload Corpus
  uses: actions/upload-artifact@v3
  with:
    name: regression-corpus
    path: corpus/
```

### Test Filtering

To test specific categories:

```bash
# Only test pure-cpp2 files
find $CPPFRONT_TESTS/pure-cpp2 -name "*.cpp2" | \
    xargs -I {} ./build/tests/regression_runner {} ...
```

## Troubleshooting

### No cppfront binary found

Install cppfront:

```bash
git clone https://github.com/hsutter/cppfront
cd cppfront
cmake -B build
cmake --build build
export CPPFRONT_BIN=$PWD/build/cppfront
```

### Test directory not found

Script will auto-clone if `CPPFRONT_TESTS` is unset. Manual clone:

```bash
git clone https://github.com/hsutter/cppfront
export CPPFRONT_TESTS=$PWD/cppfront/regression-tests
```

### High difference count

Check diff files in `corpus/diffs/` to understand:
- Missing features in cppfort
- Different code generation strategies
- Bugs in either transpiler

### SHA256 mismatches

Indicates test files changed. Expected if:
- Upstream cppfront updated tests
- Local modifications to tests

Unexpected if:
- File corruption
- Concurrent modification

## Development

### Adding New Comparisons

Edit `IsomorphicComparator` class in `regression_runner.cpp`:

```cpp
CppNode parse_simplified(const std::string& code) {
    // Add new parsing logic for additional constructs
}
```

### Custom Metrics

Extend `print_summary()` to report additional statistics:

```cpp
void print_summary(const std::vector<TranspileResult>& results) {
    // Calculate custom metrics
    int feature_X_count = ...;
    std::println("Feature X usage: {}", feature_X_count);
}
```

## References

- [cppfront repository](https://github.com/hsutter/cppfront)
- [Cpp2 language specification](https://github.com/hsutter/cppfront/blob/main/docs/cpp2/README.md)
- [Sea of Nodes architecture](../docs/ARCHITECTURE.md)