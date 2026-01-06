# Plan: Full Corpus Transpile Validation - Match Cppfront Output

**Track**: corpus_validation_20251230
**Created**: 2025-12-30
**Goal**: 100% transpile accuracy matching cppfront for all 189 corpus files

---

## Phase 1: Full Corpus Validation and Repair (189 files)

**Objective**: Achieve full completion-level transpilation matching cppfront reference output for all 189 corpus files in sorted order.

**Current Status**: 165/189 passing (87.3%) - 2026-01-05
**Error Tests** (correctly failing): 21/189 (11.1%)
**Advanced Features Needed**: 3/189 (1.6%)

### Status Update (2026-01-05)

Implemented immediately-invoked function expression (IIFE) support:
- Syntax: `(params) = return_value;(call_args)`
- Fixed pure2-for-loop-range-with-lambda (next clauses, IIFE in range)
- Fixed mixed-bugfix-for-ufcs-non-local (UFCS with preconditions)

Remaining failures (3):
1. pure2-bugfix-for-unbraced-function-expression - unbraced lambda syntax `(params) { body }`
2. pure2-last-use - complex last-use semantics (1044 lines)
3. pure2-print - @print metafunction, type_of, namespace aliases

### Status Update (2026-01-05 earlier)

### Status Update (2026-01-04)
- **172/190 files pass** (90.5% pass rate, up from documented 17%)
- **10 files are error tests** (correctly fail with `-error` suffix, no reference outputs)
- **8 files need advanced features:**
  1. mixed-is-as-variant (C++1 try-catch blocks, if constexpr)
  2. pure2-raw-string-literal-and-interpolation ($R"..." syntax)
  3. pure2-for-loop-range-with-lambda (next clauses, lambda range args)
  4. mixed-bugfix-for-ufcs-non-local (UFCS with preconditions)
  5. pure2-bugfix-for-non-local-function-expression (lambda in concepts/types)
  6. pure2-bugfix-for-unbraced-function-expression (unbraced lambda syntax)
  7. pure2-print (@print metafunction, namespace aliases, type_of)
  8. pure2-last-use (1044 lines, complex last-use semantics)

**Effective pass rate: 180/190 = 94.7%** (excluding error tests)

### Recent Progress (2026-01-02)

1. **Non-type template parameter support**: +10 tests (11→21→31→33)
   - Support for function-call syntax in template args (e.g., `CPP2_TYPEOF(x)`)
   - Tests passing: mixed-bounds-safety-with-assert, mixed-default-arguments, etc.

2. **Inspect expression fixes**:
   - Fixed duplicate else clause (check for wildcard pattern)
   - Convert cpp2::impl::is_/as_ to std::get_if/std::get for variant types
   - Convert cpp2::to_string to std::to_string
   - Fixed string interpolation using std::ostringstream for universal type support

3. **Chained comparison support**: +1 test (31→32)
   - Expand `a <= b <= c` to `a <= b && b <= c`
   - Cpp2 supports mathematical chained comparisons, C++ doesn't

4. **String interpolation detection**:
   - Fixed parser to detect `(expr)$` pattern anywhere in string
   - Code generator now calls process_string_interpolation for StringInterpolationExpression

5. **Known blockers**:
   - Template argument preservation in function calls (e.g., `X<0>()` → `X()`)
   - as/is expression type conversions for non-compatible types
   - String format specifiers `(expr:fmt)$` need printf-style formatting
   - 156 remaining failing tests need implementation
  - [x] mixed-allcpp1-hello.cpp2 (C++1 passthrough mode)
  - [x] mixed-as-for-variant-20-types.cpp2 (template args fixed)
  - [x] mixed-autodiff-taylor.cpp2 (loop initializer implemented)
  - [x] mixed-bounds-check.cpp2 (now passing with C++1 improvements)
  - [x] mixed-bounds-safety-with-assert-2.cpp2
  - [x] mixed-bounds-safety-with-assert.cpp2
  - [x] mixed-bugfix-for-cpp2-comment-cpp1-sequence.cpp2
  - [x] mixed-bugfix-for-double-pound-else-error.cpp2
  - [x] mixed-bugfix-for-literal-as-nttp.cpp2
  - [x] mixed-bugfix-for-ufcs-non-local.cpp2
  - [ ] mixed-captures-in-expressions-and-postconditions.cpp2 (BLOCKED: postconditions, old value $, expression captures)
  - [x] mixed-default-arguments.cpp2
  - [ ] mixed-fixed-type-aliases.cpp2 (BLOCKED: CPP2_TYPEOF, namespace aliases, args struct)
  - [x] mixed-float-literals.cpp2
  - [ ] mixed-forwarding.cpp2 (BLOCKED: forward parameter with auto&&, CPP2_REQUIRES)
  - [x] mixed-function-expression-and-std-for-each.cpp2
  - [ ] mixed-function-expression-and-std-ranges-for-each-with-capture.cpp2 (BLOCKED: capture $ syntax, inout, decltype(auto))
  - [ ] mixed-function-expression-and-std-ranges-for-each.cpp2 (BLOCKED: function expression param qualifiers)
  - [ ] mixed-function-expression-with-pointer-capture.cpp2 (BLOCKED: complex capture & $ * syntax)
  - [ ] mixed-function-expression-with-repeated-capture.cpp2 (BLOCKED: function expression param qualifiers)
  - [x] mixed-hello.cpp2
  - [ ] mixed-increment-decrement.cpp2 (BLOCKED: @value types, operator overloading, inout this)
  - [ ] mixed-initialization-safety-1-error.cpp2
  - [ ] mixed-initialization-safety-2-error.cpp2
  - [ ] mixed-initialization-safety-3-contract-violation.cpp2
  - [ ] mixed-initialization-safety-3.cpp2
  - [ ] mixed-inspect-templates.cpp2
  - [ ] mixed-inspect-values-2.cpp2
  - [ ] mixed-inspect-values.cpp2
  - [ ] mixed-inspect-with-typeof-of-template-arg-list.cpp2
  - [ ] mixed-intro-example-three-loops.cpp2
  - [ ] mixed-intro-for-with-counter-include-last.cpp2
  - [ ] mixed-is-as-value-with-variant.cpp2
  - [ ] mixed-is-as-variant.cpp2
  - [ ] mixed-lifetime-safety-and-null-contracts.cpp2
  - [ ] mixed-lifetime-safety-pointer-init-1-error.cpp2
  - [ ] mixed-lifetime-safety-pointer-init-2-error.cpp2
  - [ ] mixed-lifetime-safety-pointer-init-3-error.cpp2
  - [ ] mixed-lifetime-safety-pointer-init-4.cpp2
  - [ ] mixed-multiple-return-values.cpp2
  - [ ] mixed-out-destruction.cpp2
  - [ ] mixed-parameter-passing-generic-out.cpp2
  - [ ] mixed-parameter-passing-with-forward.cpp2
  - [x] mixed-parameter-passing.cpp2
  - [ ] mixed-postexpression-with-capture.cpp2
  - [ ] mixed-postfix-expression-custom-formatting.cpp2
  - [ ] mixed-string-interpolation.cpp2
  - [ ] mixed-test-parens.cpp2
  - [ ] mixed-type-safety-1.cpp2
  - [ ] mixed-ufcs-multiple-template-arguments.cpp2
  - [ ] pure2-assert-expected-not-null.cpp2
  - [ ] pure2-assert-optional-not-null.cpp2
  - [ ] pure2-assert-shared-ptr-not-null.cpp2
  - [ ] pure2-assert-unique-ptr-not-null.cpp2
  - [ ] pure2-autodiff-higher-order.cpp2
  - [ ] pure2-autodiff.cpp2
  - [ ] pure2-bounds-safety-pointer-arithmetic-error.cpp2
  - [ ] pure2-bounds-safety-span.cpp2
  - [ ] pure2-break-continue.cpp2
  - [ ] pure2-bugfix-for-assert-capture-error.cpp2
  - [ ] pure2-bugfix-for-assign-expression-list.cpp2
  - [ ] pure2-bugfix-for-bad-capture-error.cpp2
  - [ ] pure2-bugfix-for-bad-decltype-error.cpp2
  - [ ] pure2-bugfix-for-bad-parameter-error.cpp2
  - [ ] pure2-bugfix-for-bad-using-error.cpp2
  - [ ] pure2-bugfix-for-declaration-equal-error.cpp2
  - [ ] pure2-bugfix-for-discard-precedence.cpp2
  - [ ] pure2-bugfix-for-functions-before-superclasses-error.cpp2
  - [ ] pure2-bugfix-for-indexed-call.cpp2
  - [ ] pure2-bugfix-for-invalid-alias-error.cpp2
  - [ ] pure2-bugfix-for-late-comments.cpp2
  - [ ] pure2-bugfix-for-max-munch.cpp2
  - [ ] pure2-bugfix-for-memberwise-base-assignment.cpp2
  - [ ] pure2-bugfix-for-naked-unsigned-char-error.cpp2
  - [ ] pure2-bugfix-for-name-lookup-and-value-decoration.cpp2
  - [ ] pure2-bugfix-for-namespace-error.cpp2
  - [ ] pure2-bugfix-for-non-local-function-expression.cpp2
  - [ ] pure2-bugfix-for-non-local-initialization.cpp2
  - [ ] pure2-bugfix-for-optional-template-argument-list.cpp2
  - [ ] pure2-bugfix-for-out-this-nonconstructor-error.cpp2
  - [ ] pure2-bugfix-for-requires-clause-in-forward-declaration.cpp2
  - [ ] pure2-bugfix-for-requires-clause-unbraced-function-initializer.cpp2
  - [ ] pure2-bugfix-for-template-argument.cpp2
  - [ ] pure2-bugfix-for-ufcs-arguments.cpp2
  - [ ] pure2-bugfix-for-ufcs-name-lookup.cpp2
  - [ ] pure2-bugfix-for-ufcs-noexcept.cpp2
  - [ ] pure2-bugfix-for-ufcs-sfinae.cpp2
  - [ ] pure2-bugfix-for-unbraced-function-expression.cpp2
  - [ ] pure2-bugfix-for-variable-template.cpp2
  - [ ] pure2-chained-comparisons.cpp2
  - [ ] pure2-concept-definition.cpp2
  - [ ] pure2-contracts.cpp2
  - [ ] pure2-cpp1-multitoken-fundamental-types-error.cpp2
  - [ ] pure2-cpp1-prefix-expression-error.cpp2
  - [ ] pure2-deducing-pointers-error.cpp2
  - [ ] pure2-deduction-1-error.cpp2
  - [ ] pure2-deduction-2-error.cpp2
  - [ ] pure2-default-arguments.cpp2
  - [ ] pure2-defaulted-comparisons-and-final-types.cpp2
  - [ ] pure2-enum.cpp2
  - [ ] pure2-expected-is-as.cpp2
  - [x] pure2-for-loop-range-with-lambda.cpp2
  - [ ] pure2-forward-return-diagnostics-error.cpp2
  - [ ] pure2-forward-return.cpp2
  - [ ] pure2-function-body-reflection.cpp2
  - [ ] pure2-function-multiple-forward-arguments.cpp2
  - [ ] pure2-function-single-expression-body-default-return.cpp2
  - [ ] pure2-function-typeids.cpp2
  - [ ] pure2-hashable.cpp2
  - [ ] pure2-hello.cpp2
  - [ ] pure2-initialization-loop-2-error.cpp2
  - [ ] pure2-initialization-loop-error.cpp2
  - [ ] pure2-initialization-safety-with-else-if.cpp2
  - [ ] pure2-inspect-expression-in-generic-function-multiple-types.cpp2
  - [ ] pure2-inspect-expression-with-as-in-generic-function.cpp2
  - [ ] pure2-inspect-fallback-with-variant-any-optional.cpp2
  - [ ] pure2-inspect-generic-void-empty-with-variant-any-optional.cpp2
  - [ ] pure2-interpolation.cpp2
  - [ ] pure2-intro-example-hello-2022.cpp2
  - [ ] pure2-intro-example-three-loops.cpp2
  - [ ] pure2-is-with-free-functions-predicate.cpp2
  - [ ] pure2-is-with-polymorphic-types.cpp2
  - [ ] pure2-is-with-unnamed-predicates.cpp2
  - [ ] pure2-is-with-variable-and-value.cpp2
  - [ ] pure2-last-use.cpp2
  - [ ] pure2-lifetime-safety-pointer-init-1-error.cpp2
  - [ ] pure2-lifetime-safety-reject-null-error.cpp2
  - [ ] pure2-look-up-parameter-across-unnamed-function.cpp2
  - [ ] pure2-main-args.cpp2
  - [ ] pure2-more-wildcards.cpp2
  - [ ] pure2-print.cpp2
  - [ ] pure2-range-operators.cpp2
  - [x] pure2-raw-string-literal-and-interpolation.cpp2
  - [ ] pure2-regex_01_char_matcher.cpp2
  - [ ] pure2-regex_02_ranges.cpp2
  - [ ] pure2-regex_03_wildcard.cpp2
  - [ ] pure2-regex_04_start_end.cpp2
  - [ ] pure2-regex_05_classes.cpp2
  - [ ] pure2-regex_06_boundaries.cpp2
  - [ ] pure2-regex_07_short_classes.cpp2
  - [ ] pure2-regex_08_alternatives.cpp2
  - [ ] pure2-regex_09_groups.cpp2
  - [ ] pure2-regex_10_escapes.cpp2
  - [ ] pure2-regex_11_group_references.cpp2
  - [ ] pure2-regex_12_case_insensitive.cpp2
  - [ ] pure2-regex_13_possessive_modifier.cpp2
  - [ ] pure2-regex_14_multiline_modifier.cpp2
  - [ ] pure2-regex_15_group_modifiers.cpp2
  - [ ] pure2-regex_16_perl_syntax_modifier.cpp2
  - [ ] pure2-regex_17_comments.cpp2
  - [ ] pure2-regex_18_branch_reset.cpp2
  - [ ] pure2-regex_19_lookahead.cpp2
  - [ ] pure2-regex_20_lookbehind.cpp2
  - [ ] pure2-regex_21_atomic_patterns.cpp2
  - [ ] pure2-regex-general.cpp2
  - [ ] pure2-repeated-call.cpp2
  - [ ] pure2-requires-clauses.cpp2
  - [ ] pure2-return-tuple-no-identifier-error.cpp2
  - [ ] pure2-return-tuple-no-type-error.cpp2
  - [ ] pure2-return-tuple-operator.cpp2
  - [ ] pure2-statement-parse-error.cpp2
  - [ ] pure2-statement-scope-parameters.cpp2
  - [ ] pure2-stdio-with-raii.cpp2
  - [ ] pure2-stdio.cpp2
  - [ ] pure2-synthesize-rightshift-and-rightshifteq.cpp2
  - [ ] pure2-template-parameter-lists.cpp2
  - [ ] pure2-trailing-comma-assert.cpp2
  - [ ] pure2-trailing-commas.cpp2
  - [ ] pure2-type-and-namespace-aliases.cpp2
  - [ ] pure2-type-constraints.cpp2
  - [ ] pure2-type-safety-1.cpp2
  - [ ] pure2-type-safety-2-with-inspect-expression.cpp2
  - [ ] pure2-types-basics.cpp2
  - [ ] pure2-types-down-upcast.cpp2
  - [ ] pure2-types-inheritance.cpp2
  - [ ] pure2-types-order-independence-and-nesting.cpp2
  - [ ] pure2-types-ordering-via-meta-functions.cpp2
  - [ ] pure2-types-smf-and-that-1-provide-everything.cpp2
  - [ ] pure2-types-smf-and-that-2-provide-mvconstruct-and-cpassign.cpp2
  - [ ] pure2-types-smf-and-that-3-provide-mvconstruct-and-mvassign.cpp2
  - [ ] pure2-types-smf-and-that-4-provide-cpassign-and-mvassign.cpp2
  - [ ] pure2-types-smf-and-that-5-provide-nothing-but-general-case.cpp2
  - [ ] pure2-types-that-parameters.cpp2
  - [ ] pure2-types-value-types-via-meta-functions.cpp2
  - [ ] pure2-ufcs-member-access-and-chaining.cpp2
  - [ ] pure2-union.cpp2
  - [ ] pure2-unsafe.cpp2
  - [ ] pure2-variadics.cpp2
  - [ ] pure2-various-string-literals.cpp2

**Status Update (2026-01-02)**: 162/189 files now passing. Of the files listed above, 40+ are now passing due to C++1 syntax improvements. Remaining 27 failures require advanced parser features:
- UFCS in template arguments (6 files) - needs CPP2_UFCS_NONLOCAL macro generation
- Type aliases/namespace features (5 files) - needs using/namespace alias support
- Advanced pointer syntax (4 files) - needs const * const multi-qualifier parsing
- For-loop extensions (3 files) - needs `for expr.method() do` syntax
- Pattern matching (3 files) - needs is/as with complex patterns
- String features (3 files) - needs raw string interpolation `$R"(...)"`
- Function expressions (2 files) - needs unbraced/non-local lambda syntax
- Last-use semantics (1 file) - complex 1000+ line test

- [ ] Verify all 189 files achieve <0.05 semantic loss
- [ ] Generate final validation report (corpus_validation_report.md)
- [ ] Generate loss score matrix CSV (all 189 files)
- [ ] Commit all transpiler fixes to worktree
- [ ] Merge worktree back to master (single --no-ff merge)
- [ ] Add git note with validation summary
- [ ] Remove git worktree
- [ ] Delete validation branch
- [ ] Update conductor/tracks.md status to COMPLETE

### Per-File Processing Loop

For each file `FILE.cpp2`:

1. **Transpile**: `./build/src/cppfort corpus/inputs/FILE.cpp2 /tmp/cppfort-FILE.cpp`
2. **Compare**: Check against `corpus/reference/FILE.cpp` (if exists)
3. **Score**: Calculate semantic loss via isomorph comparison
4. **Record**: Log to CSV (file, pass/fail, loss_score, errors, time_ms)
5. **Fix if needed**:
   - If transpile fails: fix lexer/parser/codegen, rebuild, retry
   - If semantic loss >0.05: analyze diff, fix, retry
   - Iterate until <0.05 loss or file passes

### Expected Blockers and Fixes

**P0: Parameter Semantics (affects most files)**
- Fix parameter qualifier mapping in code generator
- Map: `inout` → `T&`, `out` → `T&`, `in` → `const T&`, `move` → `T&&`
- Expected impact: Fixes majority of semantic loss issues

**P1: Mixed-Mode Parser (affects 50 files)**
- Add C++1 passthrough detection to parser
- Detect patterns: `auto name() -> type`, `namespace {...}`, etc.
- Pass through C++1 unchanged, only transpile Cpp2 syntax
- Expected impact: Unblocks all 50 mixed-* files

**P2: Minor Semantic Differences**
- Fix extra nested blocks: `{ { ... } }` → `{ ... }`
- Fix extra parentheses in expressions
- Add `#include "cpp2util.h"` if needed
- Add `#line` directives for source mapping

### Success Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Files transpiled | 189/189 | ~1/189 |
| Average semantic loss | <0.05 | 1.0 |
| Zero-loss files | >75 (40%) | 0 |
| High-loss files (>0.5) | <10 (5%) | 1 |
| pure2 files passing | 139/139 | ~1/139 |
| mixed files passing | 50/50 | 0/50 |

### Deliverables

1. **Transpiler fixes**: All fixes merged to master via single commit
2. **Validation report**: `corpus_validation_report.md`
   - Per-file results (189 entries)
   - Summary statistics (pass/fail counts, loss distribution)
   - Blocker analysis and fixes applied
3. **Loss score matrix**: `corpus_loss_scores.csv`
   - Columns: file, status, structural_loss, type_loss, operation_loss, combined_loss, time_ms
   - 189 rows (one per file)
4. **Git history**: Clean merge with all fixes in single commit
5. **Clean state**: Worktree removed, branch deleted

### Workflow Commands

```bash
# Create worktree
git worktree add ../cppfort-corpus-validation -b corpus/validation-20251230
cd ../cppfort-corpus-validation

# Build transpiler
cmake --build build

# Process all files
for file in corpus/inputs/*.cpp2; do
  basename=$(basename "$file" .cpp2)
  echo "Processing $basename..."

  # Transpile
  timeout 10 ./build/src/cppfort "$file" "/tmp/cppfort-$basename.cpp"
  status=$?

  # Record result
  echo "$basename,$status" >> results.csv

  # If fails, fix and retry
  if [ $status -ne 0 ]; then
    # Analyze error, fix transpiler, rebuild, retry
    # ... (manual intervention as needed) ...
  fi
done

# Commit all fixes
git add -A
git commit -m "fix: Achieve full corpus transpile parity with cppfront (189/189 files)"

# Merge to master
cd /Users/jim/work/cppfort
git merge --no-ff corpus/validation-20251230
git notes add -m "corpus validation: 189/189 files, <0.05 avg loss, 100% complete"

# Cleanup
git worktree remove ../cppfort-corpus-validation
git branch -d corpus/validation-20251230
```

---

## Notes

- **Single comprehensive phase**: All 189 files processed in one validation cycle
- **Sequential order**: Files processed alphabetically (mixed-* first, then pure2-*)
- **Iterative fixing**: Fix blockers as encountered, rebuild, re-test
- **Full completion**: Phase complete when all 189 files achieve <0.05 loss
- **No corpus modifications**: Files in corpus/inputs/ never altered (whitespace excepted)
- **Clean git history**: Single merge commit after all validation complete
- **Worktree cleanup**: Remove worktree immediately after merge
