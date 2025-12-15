# Clang AST Dump & N-Way Mapping Status

**Last Updated**: 2025-12-14

## Overview

This document tracks the generation of **Clang C++ AST dumps** from the cppfront regression test corpus and the creation of **n-way semantic mappings** (Cpp2 → C++1 → Clang AST).

## Objectives

1. **Generate Clang AST dumps** for all cppfront regression tests
2. **Create semantic mappings** documenting transformations at each level:
   - Cpp2 syntax → C++1 code
   - C++1 code → Clang AST nodes
   - AST patterns → MLIR Cpp2Dialect ops
3. **Build automated tooling** for corpus-scale processing
4. **Extract reusable patterns** for compiler development

## Current Status

### ✅ Completed

1. **Infrastructure Setup**
   - Created `corpus/ast_mappings/` directory
   - Built `tools/generate_ast_mappings.py` automation script
   - Documented mapping methodology in `corpus/ast_mappings/README.md`

2. **Example Mapping: mixed-hello.cpp2**
   - **Source**: `corpus/inputs/mixed-hello.cpp2`
   - **C++1 Translation**: `corpus/ast_mappings/mixed-hello.cpp`
   - **AST Dump**: `corpus/ast_mappings/mixed-hello.ast.txt`
   - **Mapping Doc**: `corpus/ast_mappings/mixed-hello.mapping.md`

   Documented transformations:
   - Function signatures (postfix → trailing return type)
   - Variable declarations (name-first → type-first)
   - Parameter qualifiers (`inout` → `&`)
   - UFCS resolution
   - String concatenation operators
   - Stream insertion chains

3. **Mapping Documentation Template**
   - Established format for mapping files
   - Includes: Cpp2 source, C++1 translation, AST patterns, semantic analysis
   - Side-by-side syntax comparison
   - AST node structure visualization

### 🔄 In Progress

1. **Corpus Processing**
   - **Total test files**: ~195 cpp2 files in `third_party/cppfront/regression-tests/`
   - **Processed**: 1 (mixed-hello.cpp2)
   - **Remaining**: 194

2. **Tooling Dependencies**
   - ❌ **cppfront binary**: Not yet built (required for Cpp2 → C++1 transpilation)
   - ✅ **clang++**: Available (LLVM 21.1.7)
   - ✅ **Python**: Available for automation scripts

### ⏳ Pending

1. **Build cppfront Transpiler**
   - Source: `third_party/cppfront/`
   - Required for batch processing of cpp2 files
   - Alternative: Use our own cppfort transpiler (once build errors resolved)

2. **Batch AST Generation**
   - Run `generate_ast_mappings.py` on full corpus
   - Generate summary report
   - Identify coverage gaps

3. **Pattern Extraction**
   - Analyze AST dumps to extract common patterns
   - Create pattern library for compiler development
   - Map patterns to MLIR Cpp2Dialect ops

4. **MLIR Integration**
   - Document Clang AST → MLIR Cpp2Dialect transformations
   - Create three-way mapping: Cpp2 → C++1 AST → MLIR
   - Validate against Sea-of-Nodes IR requirements

## Mapping Examples

### 1. Function Declaration Mapping

```
Cpp2:           name: () -> std::string = { ... }
                ↓
C++1:           auto name() -> std::string { ... }
                ↓
Clang AST:      FunctionDecl 'std::string ()'
                ├─CompoundStmt
                └─ReturnStmt
                ↓
MLIR:           cpp2.func @name() -> !cpp2.struct<"std::string"> {
                  ...
                  cpp2.return %result
                }
```

**Transform Rules**:
- Postfix return type → Trailing return type → FunctionDecl node
- Function body `= { }` → `{ }` → CompoundStmt
- Implicit `void` omitted in Cpp2, explicit in C++1

### 2. Variable Declaration Mapping

```
Cpp2:           s: std::string = "world";
                ↓
C++1:           std::string s = "world";
                ↓
Clang AST:      VarDecl 's' 'std::string'
                └─StringLiteral "world"
                ↓
MLIR:           %s = cpp2.var "s" : !cpp2.struct<"std::string">
                %str = cpp2.constant "world" : !cpp2.ptr<i8>
                cpp2.store %str to %s
```

**Transform Rules**:
- Name-first syntax → Type-first syntax → VarDecl node
- Cpp2 requires initialization; C++1 allows uninitialized (AST shows initializer)
- MLIR uses SSA form with explicit store operation

### 3. Parameter Qualifier Mapping

```
Cpp2:           decorate: (inout s: std::string) = { ... }
                ↓
C++1:           auto decorate(std::string& s) -> void { ... }
                ↓
Clang AST:      FunctionDecl 'void (std::string &)'
                ├─ParmVarDecl 's' 'std::string &'
                └─CompoundStmt
                ↓
MLIR:           cpp2.func @decorate(%s: !cpp2.ptr<!cpp2.struct<"std::string">, inout>) {
                  ...
                }
```

**Transform Rules**:
- `inout` qualifier → `&` reference → ParmVarDecl with reference type
- Cpp2 guarantees definite initialization for `inout` (not encoded in C++1/AST)
- MLIR can preserve `inout` semantic in custom type

## Directory Structure

```
corpus/
├── inputs/                      # Original .cpp2 test files (195 files)
├── ast_mappings/                # Generated mappings
│   ├── README.md               # Mapping methodology documentation
│   ├── mixed-hello.cpp         # C++1 translation example
│   ├── mixed-hello.ast.txt     # Clang AST text dump
│   ├── mixed-hello.ast.json    # Clang AST JSON dump (future)
│   ├── mixed-hello.mapping.md  # Detailed semantic mappings
│   └── MAPPING_SUMMARY.md      # Summary report (generated)
└── sha256_database.txt         # Corpus checksums

tools/
└── generate_ast_mappings.py    # Automation script
```

## Automation Script Usage

### Basic Usage

```bash
# Process all corpus files
./tools/generate_ast_mappings.py

# Process specific pattern
./tools/generate_ast_mappings.py --pattern "mixed-*.cpp2"

# Limit for testing
./tools/generate_ast_mappings.py --limit 5

# Specify cppfront path
./tools/generate_ast_mappings.py --cppfront third_party/cppfront/build/cppfront
```

### Prerequisites

1. **Build cppfront**:
   ```bash
   cd third_party/cppfront
   # Check for build instructions (CMakeLists.txt or Makefile)
   cmake -B build
   cmake --build build
   ```

2. **Verify clang**:
   ```bash
   clang++ --version  # Should be 14+ with AST dump support
   ```

### Script Features

- **Automatic transpilation**: Cpp2 → C++1 using cppfront
- **Dual-format AST dumps**: Text (readable) and JSON (parseable)
- **Pattern extraction**: Identifies common AST structures
- **Summary reporting**: Generates corpus-wide statistics
- **Error handling**: Continues on individual file failures

## Key Semantic Differences

The mappings document where Cpp2 provides stronger guarantees than C++:

| Feature | Cpp2 Guarantee | C++1 Behavior | AST Representation | Impact |
|---------|----------------|---------------|-------------------|--------|
| Variable init | Always required | Optional | VarDecl may lack initializer | Prevents uninitialized reads |
| `inout` params | Definite init | Pass-by-ref (no guarantee) | ParmVarDecl with `&` | Clearer ownership semantics |
| `out` params | Definite assignment | Pass-by-ref (no guarantee) | ParmVarDecl with `&` | Function must initialize |
| Bounds checks | Automatic | Manual/opt-in | No AST difference | Runtime safety |
| Null checks | Automatic for pointers | Manual | No AST difference | Prevents null dereference |

## Next Steps

### Immediate

1. ✅ Build/locate cppfront binary
2. ⏳ Run `generate_ast_mappings.py` on first 10 corpus files
3. ⏳ Review generated mappings for quality
4. ⏳ Iterate on mapping template format

### Short-term

1. Process entire corpus (~195 files)
2. Generate comprehensive `MAPPING_SUMMARY.md`
3. Extract common AST patterns into reusable catalog
4. Cross-reference with existing `tools/inference/emit_mappings.py` output

### Long-term

1. Create three-way mapping: Cpp2 → C++1 AST → MLIR Cpp2Dialect
2. Build pattern library for automated translation
3. Integrate with cppfort compiler pipeline
4. Validate mappings against cppfront's official output

## Integration Points

### With Existing Tooling

1. **`tools/inference/emit_mappings.py`**
   - Generates Clang AST → MLIR mappings from C++ code
   - Can consume our C++1 translations
   - Produces schema-compliant mapping candidates

2. **`include/Cpp2Dialect.td`**
   - Defines MLIR Cpp2Dialect ops
   - Target for AST → MLIR transformations
   - Referenced in mapping validation

3. **Regression Tests**
   - `tests/cppfront_regression_tests.cpp`
   - Can validate transpilation against AST mappings
   - Cross-check semantic preservation

## Metrics

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Cpp2 files in corpus | 195 | 195 | ✅ |
| Files transpiled | 195 | 0 | ⏳ |
| AST dumps generated | 195 | 1 | ⏳ |
| Mappings documented | 195 | 1 | ⏳ |
| Pattern catalog entries | 50+ | 6 | ⏳ |
| MLIR integration examples | 10 | 0 | ⏳ |

## References

- **Clang AST Documentation**: https://clang.llvm.org/docs/IntroductionToTheClangAST.html
- **Cpp2 Specification**: https://github.com/hsutter/cppfront
- **MLIR Dialect Tutorial**: https://mlir.llvm.org/docs/Tutorials/CreatingADialect/
- **Sea of Nodes**: Simple chapters (Cliff Click)

## Questions & Issues

1. **cppfront Build**:
   - Need to verify build process for third_party/cppfront
   - May need specific dependencies (C++20 compiler, CMake, etc.)

2. **AST Dump Size**:
   - Some AST dumps may be very large (MB+)
   - Consider compression or selective dumping

3. **Pattern Ambiguity**:
   - Some Cpp2 features map to multiple C++ patterns
   - Need heuristics to choose canonical mapping

4. **MLIR Verification**:
   - How to validate that AST → MLIR transformations preserve semantics?
   - Need roundtrip tests or golden outputs

---

**Status Summary**: Infrastructure ready, example mapping complete, awaiting cppfront build for batch processing.
