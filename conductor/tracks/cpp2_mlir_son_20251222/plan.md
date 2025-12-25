# Plan: Establish Core Cpp2 to MLIR Front-IR Conversion and Sea of Nodes Dialect Integration

## Phase 1: Foundational MLIR and FIR Dialect Setup [checkpoint: 89ca649]

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
- [ ] **Task:** Conductor - User Manual Verification 'Foundational MLIR and FIR Dialect Setup' (Protocol in workflow.md)

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

### [ ] **BLOCKER:** Cppfront Regression Test Parity - Variable Declarations and Code Generation Fixed (2025-12-24)

**Category:** cppfront-parity

**Status:** IN PROGRESS - Major fixes completed

**Description:**
Cppfort had critical bugs in parsing and code generation. Fixed unified syntax local variable declarations, string literals, and operator handling.

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

**Current Status:**
- Local variable declarations: ✅ WORKING
- String literals: ✅ WORKING
- Stream operators (`<<`): ✅ WORKING
- Assignment operator (`=`): ✅ WORKING

**Remaining Issues:**
- Forward declarations not generated for functions called before definition
- `[[nodiscard]]` attribute placement incorrect (should be after function signature)
- Many Cpp2 features still unsupported (contracts, inspect, UFCS, string interpolation, ranges)
- Test framework doesn't validate output (only checks exit codes)

**Blocker Details:**
Previously broken Cpp2 syntax now working:
- Function declarations (`name: (params) -> type = { body }`) - ✅ WORKING
- Variable declarations (`name: type = value;`) - ✅ WORKING
- Stream operators (`<<`, `>>`) - ✅ WORKING
- Assignment operator (`=`) - ✅ WORKING

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
- [ ] **SHA256 Merkle Trees:** Source AST MUST be serializable with SHA256 Merkle tree hash for CRDT-based diff/merge (Pijul integration)
- [ ] **Test Framework Validation:** Tests MUST validate transpilation output, not just exit codes

---

## Phase 4: Test Framework Fix - Output Validation

**Status:** BLOCKED (depends on semantic analysis fixes)

- [ ] **Task:** Fix test framework to validate transpiled output, not just exit codes
    - [ ] **Sub-task:** Compile transpiled .cpp output using clang++
    - [ ] **Sub-task:** Run compiled binary and capture stdout
    - [ ] **Sub-task:** Compare against expected output (from .cpp2 comments or .expected files)
    - [ ] **Sub-task:** Handle tests that require stdin input
    - [ ] **Sub-task:** Update TEST_TIMEOUT_MS if compilation/execution takes longer
    - **Files:** `tests/cppfront_test_framework.cpp`, `tests/cppfront_full_regression.cpp`
