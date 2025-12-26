# Plan: Establish Core Cpp2 to MLIR Front-IR Conversion and Sea of Nodes Dialect Integration

## Phase 1: Foundational MLIR and FIR Dialect Setup [checkpoint: 5a18d93]

- [x] **Task:** Define the initial MLIR Dialect for the Front-IR (FIR).
    - [x] **Sub-task:** Write tests for the basic FIR dialect operations and types.
    - [x] **Sub-task:** Implement the basic FIR dialect operations and types.
- [x] **Task:** Implement the AST to FIR conversion for a basic "Hello, World" style function.
    - [x] **Sub-task:** Write tests for converting a simple function AST node to FIR.
    - [x] **Sub-task:** Implement the AST to FIR converter for simple functions.
- [x] **Task:** Clang AST corpus processing for cppfront regression tests.
    - [x] **Sub-task:** Process 146 cppfront regression tests (cpp2 → C++1 → Clang AST).
    - [x] **Sub-task:** Extract 40 function signatures with parameter qualifiers.
    - [x] **Sub-task:** Integrate corpus-derived patterns into AST definition:
        - ParameterQualifier enum (InOut, Out, Move, Forward, Virtual, Override)
        - Qualifiers on FunctionDeclaration::Parameter, LambdaExpression::Parameter, VariableDeclaration
        - UFCS tracking (is_ufcs flag on CallExpression)
        - Bounds checking (has_bounds_check on SubscriptExpression, BoundsCheckExpression)
- [x] **Task:** Complete semantic mapping from corpus patterns.
    - [x] **Sub-task:** Parse Cpp2 qualifiers (inout, out, move, forward, virtual, override).
    - [x] **Sub-task:** Convert Clang AST patterns to cpp2 AST nodes using corpus mappings.
    - [x] **Sub-task:** AST→FIR conversion using corpus-derived semantics.
    - [x] **Sub-task:** MLIR ops tagged with corpus semantics.
- [x] **Task:** Conductor - User Manual Verification 'Foundational MLIR and FIR Dialect Setup' (Protocol in workflow.md)

## Phase 2: Sea of Nodes (SON) Dialect and Lowering [checkpoint: d5b5758]

- [x] **Task:** Define the initial MLIR Dialect for the Sea of Nodes IR (`sond`).
    - [x] **Sub-task:** Write tests for the core `sond` operations (e.g., constants, basic arithmetic).
    - [x] **Sub-task:** Implement the core `sond` operations.
- [x] **Task:** Implement the lowering from the FIR dialect to the `sond` dialect for simple functions.
    - [x] **Sub-task:** Write tests for lowering a simple FIR function to `sond`.
    - [x] **Sub-task:** Implement the FIR to `sond` lowering pass.
- [x] **Task:** Conductor - User Manual Verification 'Sea of Nodes (SON) Dialect and Lowering' (Protocol in workflow.md)

## Phase 3: Pijul CRDT and Graph Serialization [checkpoint: 9cd579f]

- [x] **Task:** Implement the core Pijul CRDT logic from first principles.
    - [x] **Sub-task:** Write tests for the core CRDT functionalities (e.g., patch creation, application).
    - [x] **Sub-task:** Implement the Pijul CRDT data structures and algorithms.
- [x] **Task:** Implement serialization for the `sond` dialect using the Pijul CRDT implementation.
    - [x] **Sub-task:** Write tests for serializing a simple `sond` graph.
    - [x] **Sub-task:** Implement the `sond` serialization logic.
- [x] **Task:** Implement deserialization for the `sond` dialect.
    - [x] **Sub-task:** Write tests for deserializing a `sond` graph and verifying its integrity.
    - [x] **Sub-task:** Implement the `sond` deserialization logic.
- [x] **Task:** Conductor - User Manual Verification 'Pijul CRDT and Graph Serialization' (Protocol in workflow.md)

---

## Blockers

### [ ] **BLOCKER:** Cppfront Regression Test Parity - Code Generation Fixes Complete (2025-12-24)

**Category:** cppfront-parity

**Status:** IN PROGRESS - Core transpilation working

**Description:**
Cppfort had critical bugs in parsing and code generation. All major issues fixed - transpiler now generates valid C++ code for basic Cpp2 syntax.

**Fixes Completed (2025-12-24):**

**Fix 1: Unified Syntax Local Variable Declarations**
- Parser was NOT recognizing `name: type = value;` in statement position (inside function bodies)
- Fixed by adding unified syntax handling to `Parser::statement()` (src/parser.cpp:226-263)
- Variable declarations like `s: std::string = "world";` now generate correctly

**Fix 2: String Literal Double Quotes Bug**
- Parser stored full lexeme including quotes, code generator added quotes again
- Result: `""hello""` instead of `"hello"`
- Fixed by stripping surrounding quotes in `Parser::primary()` (src/parser.cpp:1261-1271)

**Fix 3: Missing Operator Support in Code Generator**
- `<<` and `>>` operators for streams - added `LeftShift`/`RightShift` cases
- `=` assignment operator - added `Equal` case
- Result: `?op?` placeholders replaced with correct operators

**Fix 4: Forward Declarations (2025-12-24)**
- Added `generate_function_forward_declaration()` function
- Generate forward declarations for all functions (except main) before definitions
- Functions can now call other functions regardless of declaration order

**Fix 5: [[nodiscard]] Attribute Placement (2025-12-24)**
- Moved [[nodiscard]] before return type for widest compatibility
- Exclude main() from getting [[nodiscard]] (entry point restriction)
- Fixed compile error: "'nodiscard' attribute cannot be applied to types"

**Current Status:**
- Local variable declarations: ✅ WORKING
- String literals: ✅ WORKING
- Stream operators (`<<`): ✅ WORKING
- Assignment operator (`=`): ✅ WORKING
- Forward declarations: ✅ WORKING
- [[nodiscard]] placement: ✅ WORKING
- **Transpiled code compiles and runs successfully!** ✅

**Test Result:**
```
$ ./cppfort pure2-hello.cpp2 | clang++ -x c++ - -o test && ./test
Hello world
```

**Remaining Issues:**
- Many Cpp2 features still unsupported (contracts, inspect, UFCS, string interpolation, ranges)
- Test framework doesn't validate output (only checks exit codes)
- Only basic function/variable syntax supported

**Blocker Details:**
Working Cpp2 syntax:
- Function declarations (`name: (params) -> type = { body }`) - ✅ WORKING
- Variable declarations (`name: type = value;`) - ✅ WORKING
- Stream operators (`<<`, `>>`) - ✅ WORKING
- Assignment operator (`=`) - ✅ WORKING
- Forward declarations - ✅ WORKING

Still unsupported:
- Type declarations (`name: type = { ... }`)
- Contracts (`assert`, pre/postconditions)
- Inspect expressions (`inspect value { ... }`)
- UFCS (uniform function call syntax)
- String interpolation (`"Hello $(name)!"`)
- Range operators (`0..<5`, `5..=10`)
- Pattern matching with `is`/`as`
- All other Cpp2 language features

**Resolution Criteria:**
- [ ] At least 90% of valid Cpp2 tests pass (transpile successfully)
- [ ] Transpiled output compiles with clang++
- [ ] Compiled binary runs and produces expected output
- [ ] All expected-error tests continue to correctly reject invalid syntax
- [ ] Performance within 2x of cppfront
- [ ] **15-second timeout per test:** All individual regression tests MUST complete within 15 seconds

**Quality Assertions (REQUIRED):**
- [ ] **Comment by Markdown Specification:** Cppfort dialect MUST preserve Cpp2 markdown-style comments as attributed metadata in the FIR/SON IR
- [x] **SHA256 Merkle Trees:** Source AST MUST be serializable with SHA256 Merkle tree hash for CRDT-based diff/merge (Pijul integration)
  - ✅ Implemented `include/semantic_hash.hpp` - SHA256Hash, SemanticHash, CRDTPatch
  - ✅ Implemented `src/semantic_hash.cpp` - Cpp2SemanticHashVisitor
  - ✅ Pure C++ SHA256 implementation (no OpenSSL dependency)
  - ✅ Merkle tree propagation from content hashes and children
- [x] **CRDT Patch Generation from AST Diffs:** Compute diffs between AST versions and generate CRDT patches
  - ✅ Implemented `include/crdt_patch.hpp` - ASTDiff, ASTDiffEngine, ASTPatchApplier, ASTMerge
  - ✅ Implemented `src/crdt_patch.cpp` - Diff computation, patch application, three-way merge
  - ✅ Supports InsertNode, DeleteNode, UpdateNode, MoveNode operations
  - ✅ Conflict resolution with Last-Writer-Wins semantics
- [x] **Bidirectional Semantic Mapping (cpp2 ↔ Clang AST):**
  - ✅ Implemented `include/clang_ast_reverse.hpp` - Clang AST → cpp2 AST conversion
  - ✅ Implemented `src/clang_ast_reverse.cpp` - ClangToCpp2Visitor with parameter qualifier inference
  - ✅ Maps Clang types to cpp2 qualifiers (inout/T&, out/T&, move/T&&, forward/auto&&, in/const T&)
  - ✅ Design document: `docs/CRDT_SEMANTIC_MAPPING.md`
- [x] **Test Framework Validation:** Tests MUST validate transpilation output, not just exit codes
  - ✅ Compile transpiled .cpp output using clang++ (30s timeout)
  - ✅ Run compiled binary and capture stdout (5s timeout)
  - ✅ Report compilation/execution errors as test failures
  - ✅ `execute_command()` helper function for fork/exec with timeout
  - ✅ `actual_output` field in TestResult struct

---

## Phase 4: Test Framework Fix - Output Validation [COMPLETED]

**Status:** COMPLETED

- [x] **Task:** Fix test framework to validate transpiled output, not just exit codes
    - [x] **Sub-task:** Compile transpiled .cpp output using clang++
    - [x] **Sub-task:** Run compiled binary and capture stdout
    - [x] **Sub-task:** Compare against expected output (from .cpp2 comments or .expected files)
    - [x] **Sub-task:** Handle tests that require stdin input
    - [x] **Sub-task:** Update TEST_TIMEOUT_MS if compilation/execution takes longer
    - **Files:** `tests/cppfront_test_framework.cpp`, `tests/cppfront_full_regression.cpp`

---

## Phase 5: Gap Resolution - Verification Protocol Findings

**Status:** IN PROGRESS

Added after independent verification identified gaps in spec compliance.

### Task 1: Complete AST to FIR Bridge [PARTIAL → COMPLETE]
- [x] **Task:** Add support for complex expressions in AST to FIR conversion
    - [x] **Sub-task:** Implement binary expression conversion (arithmetic, logical, comparison)
    - [x] **Sub-task:** Implement function call expression conversion
    - [x] **Sub-task:** Implement control flow (if/else, loops) conversion
    - [x] **Sub-task:** Write tests for complex expression conversion
- [x] **Task:** Add contract operation support to FIR dialect
    - [x] **Sub-task:** Define AssertOp, PreconditionOp, PostconditionOp in Cpp2FIRDialect.td
    - [x] **Sub-task:** Implement contract conversion in ASTToFIRConverter
    - [x] **Sub-task:** Write tests for contract operations

### Task 2: Enhanced FIR Dialect Operations [PARTIAL → COMPLETE]
- [x] **Task:** Implement unified function call (UFCS) operations
    - [x] **Sub-task:** Define UfcsCallOp in FIR dialect
    - [x] **Sub-task:** Convert UFCS expressions from AST
    - [x] **Sub-task:** Write tests for UFCS conversion
- [x] **Task:** Add advanced type support to FIR dialect
    - [x] **Sub-task:** Support function types and higher-order functions
    - [x] **Sub-task:** Support optional and variant types
    - [x] **Sub-task:** Write tests for advanced types

### Task 3: Comprehensive Error Handling [MISSING → IMPLEMENT]
- [ ] **Task:** Implement robust error reporting for AST to FIR conversion
    - [ ] **Sub-task:** Create DiagnosticCollector class for error aggregation
    - [ ] **Sub-task:** Add detailed error messages with source locations
    - [ ] **Sub-task:** Implement error recovery strategies
    - [ ] **Sub-task:** Write tests for error handling (invalid syntax, type errors)

### Task 4: SON Dialect Verification [MISSING → IMPLEMENT]
- [ ] **Task:** Add verifiers to ensure SON graph integrity
    - [ ] **Sub-task:** Implement dominance relationship verification
    - [ ] **Sub-task:** Implement type consistency verification for edges
    - [ ] **Sub-task:** Add cycle detection for data flow
    - [ ] **Sub-task:** Write tests for verifier logic

### Task 5: Code Coverage Reporting [MISSING → IMPLEMENT]
- [ ] **Task:** Set up code coverage measurement
    - [ ] **Sub-task:** Add CMake coverage support (ENABLE_COVERAGE option)
    - [ ] **Sub-task:** Generate coverage reports (lcov/gcov)
    - [ ] **Sub-task:** Add coverage reporting to CI
    - [ ] **Sub-task:** Verify 20% coverage threshold is met

### Task 6: Track Verification
- [ ] **Task:** Re-run independent verification after gap resolution
    - [ ] **Sub-task:** Spawn verification agent with updated codebase
    - [ ] **Sub-task:** Confirm all requirements verified
    - [ ] **Sub-task:** Mark track as complete
