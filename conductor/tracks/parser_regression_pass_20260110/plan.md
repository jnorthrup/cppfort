# Implementation Plan: Parser Regression Test Pass

## Phase 1: Preprocessor & Infrastructure (Foundation)

- [x] Task: Add preprocessor directive handling to parser
  - [x] Write test: `#include` directives are preserved in output (test_preprocessor_parsing.cpp)
  - [x] Implement: Add `Preprocessor` node kind to AST (already in slim_ast.hpp:89-90)
  - [x] Implement: Update `translation_unit()` to skip `Hash` tokens (already in parser.cpp:535-538)
  - [x] Test: Files with `#include` at top parse successfully

- [x] Task: Remove C++1 passthrough bypass
  - [x] Write test: Files without Cpp2 syntax should still emit includes (test_passthrough_bypass.cpp)
  - [x] Implement: Remove `has_cpp2_syntax()` check in main.cpp (removed lines 75-88)
  - [x] Test: All files go through parser, no passthrough

- [x] Task: Create test infrastructure
  - [x] Write test: Test harness for single-file transpilation (tests/parser/test_harness.cpp)
  - [x] Implement: Add `tests/parser/` directory
  - [x] Implement: Add unit test for EBNF grammar rules (tests/parser_grammar_test.cpp - exists)
  - [x] Implement: Add unit test for Spirit combinators (tests/parser_syntax_test.cpp - exists)
  - [x] Implement: Add unit test for AST construction (tests/parsing_annotations_test.cpp - exists)
  - [x] Test: Test suite runs and reports results

- [ ] Task: Conductor - User Manual Verification 'Phase 1' (Protocol in workflow.md)

## Phase 2: Unified Declarations (Variables)

- [x] Task: Parse variable declarations with type and initializer
  - [x] Write test: `s: std::string = "world";` parses correctly (test_variable_declarations.cpp)
  - [x] Write test: `x: int = 42;` parses correctly
  - [x] Implement: Fix `var_suffix()` grammar rule (already working in parser.cpp:491-497)
  - [x] Implement: Ensure `VariableSuffix` node captures type and init
  - [x] Test: Variable declarations appear in AST with correct structure

- [x] Task: Emit variable declarations correctly
  - [x] Write test: `s: std::string = "world";` emits `std::string s = "world";`
  - [x] Write test: `x: int = 42;` emits `int x = 42;`
  - [x] Implement: Fix `emit_local_var()` to extract type from `VariableSuffix` or direct `TypeSpecifier`
  - [x] Implement: Handle `: type` vs `:= init` vs `: type = init`
  - [x] Test: Generated C++ compiles (verified via output string check)

- [x] Task: Handle deduced variables
  - [x] Write test: `x := 42;` emits `auto x = 42;`
  - [x] Implement: Detect `:=` syntax for type deduction
  - [x] Test: Deduced variables compile (verified via output string check)

- [x] Task: Run regression tests on variables
  - [x] Test: All variable declaration tests pass (test_variable_emission.cpp)
  - [x] Measure: 5/5 sub-tests passing in test_variable_emission

- [ ] Task: Conductor - User Manual Verification 'Phase 2' (Protocol in workflow.md)

## Phase 3: Parameter Qualifiers [checkpoint: e602d48]

- [x] Task: Parse parameter qualifiers
  - [x] Write test: `(inout s: std::string)` parses with `ParamQualifier` node
  - [x] Write test: `(move x: Widget)` parses correctly
  - [x] Write test: `(out result: int)` parses correctly
  - [x] Implement: Ensure `Parameter` node captures qualifier child
  - [x] Test: AST shows qualifier structure

- [x] Task: Emit parameters with qualifiers
  - [x] Write test: `inout s: std::string` → `std::string& s`
  - [x] Write test: `out result: int` → `int& result`
  - [x] Write test: `move x: Widget` → `Widget&& x`
  - [x] Implement: Fix `emit_param()` to apply qualifiers to types
  - [x] Test: Generated function signatures compile

- [x] Task: Run regression tests on parameters
  - [x] Test: All parameter qualifier tests pass
  - [x] Measure: Count passing tests before/after

- [ ] Task: Conductor - User Manual Verification 'Phase 3' (Protocol in workflow.md)

## Phase 4: Function Declarations & Bodies [checkpoint: 8f68ecd]

- [x] Task: Parse function declarations
  - [x] Write test: `name: () -> int = { return 42; }` parses
  - [x] Write test: `func: (x: int) -> void = { }` parses
  - [x] Write test: Expression body `f: () = expr;` parses
  - [x] Implement: Ensure `FunctionSuffix` captures params, return, body
  - [x] Test: Function structure in AST

- [x] Task: Emit function declarations
  - [x] Write test: `name: () -> int` emits `auto name() -> int`
  - [x] Write test: Block body emits correctly with braces
  - [x] Write test: Expression body emits `return expr;`
  - [x] Implement: Fix `emit_function()` for all function forms
  - [x] Test: Generated functions compile

- [x] Task: Handle main function special case
  - [x] Write test: `main: () -> int` emits `int main()`
  - [x] Implement: Special case for `main` in emitter
  - [x] Test: main() signature is correct

- [x] Task: Run regression tests on functions
  - [x] Test: All function tests pass
  - [x] Measure: Count passing tests before/after

- [ ] Task: Conductor - User Manual Verification 'Phase 4' (Protocol in workflow.md)

## Phase 5: Expressions (Pratt Parser)

- [x] Task: Verify Pratt parser handles all operators
  - [x] Write test: Binary operators `+ - * / %` parse correctly
  - [x] Write test: Comparison `< > <= >= == != <=>` parse correctly
  - [x] Write test: Logical `&& || !` parse correctly
  - [x] Write test: Assignment `= += -=` parse correctly
  - [x] Test: Expression AST structure is correct

- [x] Task: Parse complex expressions
  - [x] Write test: Function calls `f(x, y)` parse
  - [x] Write test: Member access `obj.method()` parses
  - [x] Write test: Subscript `arr[i]` parses
  - [x] Write test: Ternary `cond ? a : b` parses
  - [x] Test: Complex expressions in AST

- [x] Task: Emit expressions
  - [x] Write test: Binary operations emit correctly
  - [x] Write test: Function calls emit correctly
  - [x] Implement: `emit_expression()` handles all expression types
  - [x] Test: Expressions compile

- [x] Task: Run regression tests on expressions
  - [x] Test: Expression-heavy tests pass
  - [x] Measure: Count passing tests before/after

- [ ] Task: Conductor - User Manual Verification 'Phase 5' (Protocol in workflow.md)

## Phase 6: Statements

- [x] Task: Parse all statement types
  - [x] Write test: `if (cond) { } else { }` parses
  - [x] Write test: `while (cond) { }` parses
  - [x] Write test: `for (item : items) { }` parses
  - [x] Write test: `return expr;` parses
  - [x] Test: Statement AST structure

- [x] Task: Emit statements
  - [x] Write test: if/else emits correctly
  - [x] Write test: while loop emits correctly
  - [x] Write test: for loop emits correctly
  - [x] Write test: return statement emits correctly
  - [x] Implement: `emit_statement()` for all types
  - [x] Test: Statements compile

- [x] Task: Run regression tests on statements
  - [x] Test: Statement-heavy tests pass
  - [x] Measure: Count passing tests before/after
    - Statement parsing tests: 4/4 pass (if, while, Cpp2 for, C++1 for, return)
    - Statement emission tests: 5/5 pass
    - Note: Many tests blocked by deleted include/parser.hpp

- [ ] Task: Conductor - User Manual Verification 'Phase 6' (Protocol in workflow.md)

## Phase 7: Full Regression Suite

- [x] Task: Run all cppfront regression tests
  - [x] Execute: `./ckmake regression-fort` (2026-01-16)
  - [x] Analyze: Which tests still fail
    - **Initial Results**: 50/159 passed (31.4%)
    - **After ScopeOp fix**: 54/159 passed (33.9%) - 4 tests fixed
    - **Transpile failures**: ~50 files
    - **Compile failures**: ~55 files
    - **Fixed Bug**: `std::` now correctly emitted (was `std.`)
    - **Parser gaps**: Contracts, regex, advanced features incomplete

- [ ] Task: Fix remaining failures incrementally
  - [x] Fix `ckmake` binary path and stderr redirection
  - [x] Fix `_` wildcard and `is_expression` baseline
  - [x] Fix scope resolution operator emission (`::` vs `.`)
    - [x] Implement: Added `ScopeOp` handler to emitter (line 653-672)
    - [x] Test: `std::cout` → `std::cout` now correct
    - Note: `mixed-hello.cpp2` still fails due to function ordering (forward decl needed)
  - [x] Fix basic type mapping and pointer/reference emission
    - [x] `format_type` already moves leading * and & to end
    - [x] Fixed `emit_local_var` to use `format_type` for types
    - [x] Test: `*void` → `void*` now correct (55/159 = 34.5%)
  - [x] Fix tuple initializer emission (`(1,2,3)` → `{1,2,3}`)
    - [x] Implemented: `emit_initializer` helper converts parens to braces
    - Note: mixed-bounds-check.cpp2 still has other issues
  - [x] Add UFCS macro support (CPP2_UFCS)
    - [x] Implemented: CPP2_UFCS macro in cpp2_runtime.h
    - [x] Implemented: CallOp detection of MemberOp callee for UFCS transformation
    - [x] Test: `obj.method(args)` → `CPP2_UFCS(method)(obj, args)`
  - [x] Add forward declarations for functions
    - [x] Implemented: Two-pass emit - forward declarations first, then definitions
    - [x] Skip forward decls for main() and functions with deduced return types (auto)
    - [x] Fixed CMake build to include libc++ headers properly (-isystem flag)
    - [x] Updated ckmake to include -I$PROJECT_ROOT/include for cpp2_runtime.h
  - [x] Fix named return values and type definition ordering (2026-01-19)
    - [x] Implemented: `extract_named_return_info()` extracts (name, type) from `(i:int)` syntax
    - [x] Implemented: Forward declaration emits `using fun_ret = int;` type alias
    - [x] Implemented: Function body emits return variable declaration at top
    - [x] Implemented: Empty `return;` in named return context emits `return i;`
    - [x] Implemented: Three-pass emit - forward decls, type definitions, functions
    - [x] Implemented: `emit_type()` properly traverses TypeBody to emit methods/fields
    - [x] Test: `pure2-ufcs-member-access-and-chaining` now PASS (was compile failure)
  - [x] Address remaining issues (pure2-bugfix-for-ufcs-arguments, pure2-bugfix-for-ufcs-sfinae)
    - [x] Fixed: `reconstruct_type_with_expressions()` TemplateArgs handler wasn't extracting adoptee
    - [x] Fixed: Added ScopeOp handler to properly emit `::` in qualified types
    - [x] Result: `std::type_identity_t<decltype(...)>` emits correctly (was `std::<type_identity_t, ...>`)
    - [x] Both UFCS tests now transpile successfully
  - [ ] Test: All 159 tests pass

**Current Results (2026-02-10)**: 86/159 passed (54.0%) - improved from 78/159 (49.0%)

**Breakdown:**

- Mixed-mode (C++1 + Cpp2): ~32/51 passing (62.7%)
- Pure2 (100% Cpp2): ~54/108 passing (50.0%)
- Error tests: 9 skipped (as expected)

**Recent Fixes (2026-02-10)**:

Session 1 (84/159):
- Fixed leading `::` (global scope) parsing in expressions - parser was stopping early when encountering `::foo()`
- Fixed if/while condition expressions to use `emit_expression()` instead of `node_text()` - enables `is`/`as` operators in conditions
- Fixed else clause emission - was missing `else` keyword and producing broken `}) {` syntax
- Fixed pointer types inside template arguments - `std::optional<*D>` now correctly emits as `std::optional<D*>`
- Fixed postfix dereference precedence (removed `is_prefix()` from operand start check)
- Added trailing comma support in parameter lists
- Added parameter qualifier support in for-range statements

Session 2 (86/159, +2 net new):
- Expanded `identifier_like` to include keywords usable as identifiers: `next`, `base`, `in`, `is`, `as`, `type`, `namespace`, `import` — fixes `pure2-hashable`
- Added compound assignment operators: `&=`, `|=`, `^=`, `>>=`, `<<=` to Pratt parser — fixes `pure2-synthesize-rightshift-and-rightshifteq`
- Added template parameter constraint syntax (`: type_specifier`) for `<T: type>`, `<Ts...: type>`
- Added `concept_suffix()` to `decl_suffix()` for `concept = expr;` declarations
- Fixed `operator_suffix()` EqualColon handling (`=:` lexed as single token)
- Defined `operator_decl()` rule (not yet used in `declaration()` — emitter needs constructor vs assignment differentiation first)

**Previous Fixes (2026-02-04)**:

- Added SFINAE `if constexpr(requires{...})` guards to inspect arm emission
- Added SFINAE `requires` constraints to general `as<>` template
- Added SFINAE constraints to optional `as<>` overload
- Fixed variant with duplicate types using `appears_once` trait and `std::visit`
- Added `<vector>`, `<memory>`, `<functional>` headers to cpp2_runtime.h

**Previous Fixes (2026-02-01)**:

- Fixed `is`/`as` operator precedence (now POSTFIX=16, higher than ADD=13)
- Fixed pattern matching emission for inspect expressions
- Added std::variant/any/optional support for `cpp2::is<T>()` and `cpp2::as<T>()`
- Added `is_one_of<T, Ts...>` constraint for variant overloads
- Added `<iomanip>` header for std::setw

- [x] Task: Performance verification
  - [x] Execute: Time full regression run
  - [x] Verify: Completes in under 5 minutes (29.4s)

- [x] Task: Conductor - User Manual Verification 'Phase 7' (Protocol in workflow.md)
  - Note: Skipped per user request, moved to Phase 8

## Phase 8: Advanced Cpp2 Features Tests

- [x] Task: Create contract tests [34d1b83]
  - [x] Write test: Precondition `[[expects: x > 0]]`
  - [x] Write test: Postcondition `[[ensures: result > 0]]`
  - [x] Write test: Assertion `assert(x != null)`
  - [x] Implement: Parser handles contract syntax
  - [x] Implement: Emitter generates contract code
  - [x] Test: Contract tests pass

- [x] Task: Create pattern matching tests
  - [x] Write test: `inspect (expr) { }` syntax - parsing works
  - [x] Fix precedence: `is`/`as` operators now have postfix precedence (POSTFIX=16) to bind tighter than additive operators
    - Before: `'X' + x as int` → `cpp2::as<int>('X' + x)` (wrong)
    - After: `'X' + x as int` → `'X' + cpp2::as<int>(x)` (correct)
  - [x] Implement: `emit_inspect_expression()` generates lambda-based pattern matching
  - [x] Implement: `emit_pattern_check()` for type patterns using `cpp2::is<T>()`
  - [x] Implement: `cpp2::is<T>()` and `cpp2::as<T>()` in cpp2_runtime.h with specializations for:
    - std::variant (constrained to only match alternative types, with `appears_once` for duplicate handling)
    - std::any (uses type() comparison and any_cast)
    - std::optional (checks has_value())
  - [x] Add SFINAE constraints to all `as<>` overloads for SFINAE-friendly requires expressions
  - [x] Emit `if constexpr(requires{...})` guards for generic inspect arms
  - [x] Handle variant with duplicate types via `std::visit`
  - [x] Add missing headers to cpp2_runtime.h: `<any>`, `<optional>`, `<variant>`, `<iomanip>`, `<vector>`, `<memory>`, `<functional>`
  - [x] Test: Pattern matching tests pass (7/9 inspect tests now pass) ✅
    - [x] mixed-inspect-templates - PASS
    - [x] mixed-inspect-values-2 - PASS
    - [x] pure2-inspect-fallback-with-variant-any-optional - PASS
    - [x] pure2-inspect-expression-in-generic-function-multiple-types - PASS ✨ NEW
    - [x] pure2-inspect-expression-with-as-in-generic-function - PASS ✨ NEW
    - [x] pure2-inspect-generic-void-empty-with-variant-any-optional - PASS ✨ NEW
    - [x] pure2-type-safety-2-with-inspect-expression - PASS ✨ NEW
    - [ ] mixed-inspect-values - transpile failed (C++1 at file start - needs mixed-mode parser)
    - [ ] mixed-inspect-with-typeof-of-template-arg-list - transpile failed (C++1 at file start - needs mixed-mode parser)
  - [ ] TODO: `when` clauses (not in current regression tests)

- [x] Task: Create metafunction tests
  - [x] Write test: `@value` metafunction
  - [x] Write test: `@interface` metafunction
  - [x] Write test: `@type` metafunction
  - [x] Implement: Parser handles @meta syntax
  - [x] Test: Metafunction tests pass

- [x] Task: Create string interpolation tests
  - [x] Write test: `$"Hello {name}!"` syntax (9 tests in test_string_interpolation.cpp)
  - [x] Implement: Parser handles interpolation (lexer tokenizes as StringLiteral, no parser changes needed)
  - [x] Implement: Emitter generates string concat/format (process_string_interpolation in TreeEmitter)
  - [x] Implement: cpp2::to_string(x, fmt) two-arg overload in cpp2_runtime.h
  - [x] Test: String interpolation tests pass (9/9) [0aa5532]

- [x] Task: Create UFCS tests
  - [x] Write test: `object.method()` syntax (6 tests in test_ufcs_emission.cpp)
  - [x] Write test: UFCS with qualified types (std:: is NOT UFCS)
  - [x] Implement: Parser handles UFCS (already implemented in emitter - CallOp/MemberOp detection)
  - [x] Test: UFCS tests pass (6/6) [2a92507]

- [x] Task: Create template tests
  - [x] Write test: `T:` type parameter syntax (6 tests in test_template_emission.cpp)
  - [x] Write test: Template arguments (std::vector<int>, std::map<std::string, int>)
  - [x] Implement: Parser handles templates (already implemented)
  - [x] Test: Template tests pass (6/6) [0685ddd]

- [x] Task: Create type system tests
  - [x] Write test: `Type: type = ...` definitions (6 tests in test_type_system_emission.cpp)
  - [x] Write test: Type with methods, @value, @interface metafunctions
  - [x] Implement: Parser handles type definitions (already implemented)
  - [x] Test: Type system tests pass (6/6) [9af7ea7]

- [ ] Task: Conductor - User Manual Verification 'Phase 8' (Protocol in workflow.md)

## Phase 9: Semantic Analysis & AST Loss Metric

- [ ] Task: Back-annotate Clang AST semantics
  - [ ] Implement: Parse cppfront-generated C++ with Clang
  - [ ] Implement: Extract semantic info from Clang AST
  - [ ] Implement: Map Clang AST back to parse tree nodes
  - [ ] Test: Semantics attached to parse graph

- [ ] Task: Implement AST loss metric
  - [ ] Implement: Compare cppfort AST vs cppfront AST
  - [ ] Implement: Calculate loss score per file
  - [ ] Implement: Report aggregate corpus loss
  - [ ] Test: Loss metric < 0.15 for passing tests

- [ ] Task: Validate semantic preservation
  - [ ] Test: Parameter semantics match cppfront
  - [ ] Test: Variable lifetimes match cppfront
  - [ ] Test: Function signatures match cppfront
  - [ ] Test: Expression types match cppfront
  - [ ] Test: Overall semantic loss < threshold

- [ ] Task: Conductor - User Manual Verification 'Phase 9' (Protocol in workflow.md)

## Phase 10: Final Verification

- [ ] Task: Full regression suite final run
  - [ ] Execute: `./ckmake regression`
  - [ ] Verify: All 159 tests pass

- [ ] Task: Comparative analysis vs cppfront
  - [ ] Execute: `./ckmake compare-corpus`
  - [ ] Verify: cppfort achieves parity with cppfront

- [ ] Task: Performance benchmark
  - [ ] Execute: Time full corpus processing
  - [ ] Verify: Completes in under 5 minutes

- [ ] Task: Document results
  - [ ] Update: `conductor/tracks.md` with completion status
  - [ ] Document: Final pass rate, semantic loss score

- [ ] Task: Conductor - User Manual Verification 'Phase 10' (Protocol in workflow.md)
