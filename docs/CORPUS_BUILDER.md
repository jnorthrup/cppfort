# Corpus Builder & Rule Extractor

## Purpose

Build a verified corpus of Cpp2→C++ transpilations from cppfront and extract transformation rules for cppfort implementation.

## Architecture

### Three-Phase Process

**Phase 1: SHA256 Verification**
- Compute SHA256 checksums of all `.cpp2` test files
- Compare against previous run to detect upstream changes
- Record checksums in `sha256_database.txt`

**Phase 2: Corpus Generation**
- Execute cppfront on each `.cpp2` file
- Capture transpiled C++ output
- Compute SHA256 of each output
- Store outputs in `corpus/outputs/`

**Phase 3: Rule Extraction**
- Parse input/output pairs using regex patterns
- Extract transformation rules:
  - `return_statement`: `return expr;` → `return expr;`
  - `function_declaration`: `name: (params) -> type = {` → `auto name(params) -> type {`
  - `class_declaration`: `name: type = {` → `class name {`
  - `namespace_declaration`: `name: namespace = {` → `namespace name {`
  - `variable_declaration`: `name: type = value;` → `type name = value;`
- Deduplicate rules and collect examples
- Save to `transpile_rules.txt`

## Usage

```bash
./scripts/run_regression.sh
```

Or direct invocation:

```bash
./build/tests/regression_runner \
    /path/to/cppfront/regression-tests \
    /path/to/cppfront \
    ./corpus
```

## Output Structure

```
corpus/
├── sha256_database.txt        # SHA256 checksums
│   ├── [inputs]               # Input .cpp2 files
│   └── [outputs]              # Cppfront outputs
├── outputs/                   # Transpiled C++ files
│   ├── test1.cpp
│   └── test2.cpp
├── transpile_rules.txt        # Extracted transformation patterns
└── report.txt                 # Summary statistics
```

## SHA256 Database Format

```
# Input file SHA256 checksums
[inputs]
<sha256> filename.cpp2
<sha256> another.cpp2

# Cppfront output SHA256 checksums
[outputs]
<sha256> filename.cpp2
<sha256> another.cpp2
```

The output section uses the original `.cpp2` filename as the key, mapping it to the SHA256 of the transpiled `.cpp` output.

## Transpile Rules Format

```
[rule:function_declaration]
pattern: (\w+):\s*\(([^)]*)\)\s*->\s*(\w+)\s*=\s*\{
output: auto $1($2) -> $3 {
examples:
  - add: (x: int, y: int) -> int = {
  - multiply: (a: double, b: double) -> double = {

[rule:variable_declaration]
pattern: (\w+):\s*(\w+)\s*=\s*([^;]+);
output: $2 $1 = $3;
examples:
  - x: int = 42;
  - name: string = "hello";
```

## Rule Extraction Logic

### Pattern Detection

1. **Return Statement**
   - Input: `return <expr>;`
   - Output: `return <expr>;`
   - Identity transformation (validates parser)

2. **Function Declaration**
   - Input: `name: (params) -> type = {`
   - Output: `auto name(params) -> type {`
   - Captures: name, params, return type

3. **Class Declaration**
   - Input: `name: type = {`
   - Output: `class name {`
   - Captures: class name

4. **Namespace Declaration**
   - Input: `name: namespace = {`
   - Output: `namespace name {`
   - Captures: namespace name

5. **Variable Declaration**
   - Input: `name: type = value;`
   - Output: `type name = value;`
   - Captures: name, type, initializer

### Deduplication

Rules with the same `pattern_name` are merged:
- Single pattern and template kept
- Examples from all occurrences combined
- Ensures unique rule set

## Integration with Cppfort

### Using Extracted Rules

The rules can drive cppfort's transpiler:

```cpp
// Load rules
std::vector<TranspileRule> rules = load_rules("corpus/transpile_rules.txt");

// Apply to Cpp2 source
std::string cpp2_source = load_file("input.cpp2");
std::string cpp_output = apply_rules(cpp2_source, rules);
```

### Rule Application Order

1. Namespace declarations (outermost scope)
2. Class declarations
3. Function declarations
4. Variable declarations
5. Statement-level transformations

### Validation

Compare cppfort output against corpus:

```bash
# Generate with cppfort
./cppfort input.cpp2 output.cpp

# Compute SHA256
sha256sum output.cpp

# Compare against corpus/sha256_database.txt [outputs]
grep "input.cpp2" corpus/sha256_database.txt
```

## Report Contents

```
Cppfort Regression Report
=========================

Test Files: 450
Successful transpilations: 425 (94.4%)
Failed transpilations: 25 (5.6%)

Extracted Rules: 5

Rules by category:
  return_statement: 1
  function_declaration: 1
  class_declaration: 1
  namespace_declaration: 1
  variable_declaration: 1

Corpus generated at: ./corpus
  - outputs/: Cppfront transpiled outputs
  - sha256_database.txt: Integrity checksums
  - transpile_rules.txt: Extracted transformation rules
```

## Detecting Upstream Changes

When test files change:

```
WARNING: test_new_feature.cpp2 changed (SHA256 mismatch)
```

Actions:
1. Review upstream changes in cppfront repository
2. Regenerate corpus to capture new patterns
3. Update extracted rules
4. Re-validate cppfort against new corpus

## Implementation Notes

### No Cppfort Execution

This tool does **not** run cppfort. It:
- Runs **only** cppfront
- Builds corpus from cppfront outputs
- Extracts rules for cppfort to **implement**

### Pattern Matching

Uses C++ `<regex>` for pattern extraction. Limitations:
- Simple syntactic patterns only
- No semantic analysis
- May miss complex transformations

### Rule Coverage

Extracted rules cover basic constructs. Not extracted:
- Template instantiation patterns
- UFCS transformations
- Contract expansions
- Metafunction invocations

These require semantic analysis beyond regex patterns.

## Future Enhancements

1. **Semantic Rule Extraction**
   - Parse ASTs instead of regex
   - Capture context-dependent transformations
   - Handle nested scopes

2. **Rule Validation**
   - Apply rules back to corpus inputs
   - Compare against original outputs
   - Measure rule accuracy

3. **Incremental Updates**
   - Only re-transpile changed files
   - Diff against previous corpus
   - Track rule evolution

4. **Category-Specific Rules**
   - Pure Cpp2 vs mixed syntax
   - Feature-specific patterns
   - Error case handling