# Head-to-Head: cppfort vs cppfront

Generated: Fri Jan  2 18:53:59 CST 2026

## Results

| Status | Count |
|--------|-------|
| PASS | 26 |
| FAIL | 132 |
| SKIP | 0 |

## Passing Tests

```
mixed-allcpp1-hello
mixed-bugfix-for-cpp2-comment-cpp1-sequence
mixed-bugfix-for-literal-as-nttp
mixed-default-arguments
mixed-hello
mixed-intro-example-three-loops
pure2-assert-expected-not-null
pure2-break-continue
pure2-bugfix-for-optional-template-argument-list
pure2-bugfix-for-requires-clause-unbraced-function-initializer
pure2-concept-definition
pure2-expected-is-as
pure2-forward-return
pure2-hashable
pure2-hello
pure2-inspect-expression-in-generic-function-multiple-types
pure2-inspect-expression-with-as-in-generic-function
pure2-interpolation
pure2-intro-example-hello-2022
pure2-intro-example-three-loops
pure2-range-operators
pure2-synthesize-rightshift-and-rightshifteq
pure2-trailing-comma-assert
pure2-type-and-namespace-aliases
pure2-type-safety-1
pure2-type-safety-2-with-inspect-expression
```

## Failing Tests

```
mixed-as-for-variant-20-types: ref compiles, fort doesn't
mixed-autodiff-taylor: ref compiles, fort doesn't
mixed-bounds-check: output differs
mixed-bounds-safety-with-assert-2: output differs
mixed-bounds-safety-with-assert: output differs
mixed-bugfix-for-ufcs-non-local: cppfort transpile failed
mixed-captures-in-expressions-and-postconditions: ref compiles, fort doesn't
mixed-fixed-type-aliases: ref compiles, fort doesn't
mixed-float-literals: output differs
mixed-forwarding: ref compiles, fort doesn't
mixed-function-expression-and-std-for-each: output differs
mixed-function-expression-and-std-ranges-for-each-with-capture: ref compiles, fort doesn't
mixed-function-expression-and-std-ranges-for-each: ref compiles, fort doesn't
mixed-function-expression-with-pointer-capture: ref compiles, fort doesn't
mixed-function-expression-with-repeated-capture: ref compiles, fort doesn't
mixed-increment-decrement: ref compiles, fort doesn't
mixed-initialization-safety-3-contract-violation: ref compiles, fort doesn't
mixed-initialization-safety-3: ref compiles, fort doesn't
mixed-inspect-templates: ref compiles, fort doesn't
mixed-inspect-values-2: ref compiles, fort doesn't
mixed-inspect-values: ref compiles, fort doesn't
mixed-inspect-with-typeof-of-template-arg-list: cppfort transpile failed
mixed-intro-for-with-counter-include-last: output differs
mixed-is-as-value-with-variant: ref compiles, fort doesn't
mixed-is-as-variant: cppfort transpile failed
mixed-lifetime-safety-and-null-contracts: ref compiles, fort doesn't
mixed-lifetime-safety-pointer-init-4: ref compiles, fort doesn't
mixed-multiple-return-values: ref compiles, fort doesn't
mixed-out-destruction: ref compiles, fort doesn't
mixed-parameter-passing-generic-out: ref compiles, fort doesn't
mixed-parameter-passing-with-forward: ref compiles, fort doesn't
mixed-parameter-passing: ref compiles, fort doesn't
mixed-postexpression-with-capture: ref compiles, fort doesn't
mixed-postfix-expression-custom-formatting: ref compiles, fort doesn't
mixed-string-interpolation: ref compiles, fort doesn't
mixed-test-parens: ref compiles, fort doesn't
mixed-type-safety-1: ref compiles, fort doesn't
mixed-ufcs-multiple-template-arguments: cppfort transpile failed
pure2-assert-optional-not-null: ref compiles, fort doesn't
pure2-assert-shared-ptr-not-null: ref compiles, fort doesn't
pure2-assert-unique-ptr-not-null: ref compiles, fort doesn't
pure2-autodiff-higher-order: ref compiles, fort doesn't
pure2-autodiff: ref compiles, fort doesn't
pure2-bounds-safety-span: ref compiles, fort doesn't
pure2-bugfix-for-assign-expression-list: ref compiles, fort doesn't
pure2-bugfix-for-discard-precedence: ref compiles, fort doesn't
pure2-bugfix-for-indexed-call: ref compiles, fort doesn't
pure2-bugfix-for-late-comments: ref compiles, fort doesn't
pure2-bugfix-for-max-munch: ref compiles, fort doesn't
pure2-bugfix-for-memberwise-base-assignment: ref compiles, fort doesn't
pure2-bugfix-for-name-lookup-and-value-decoration: ref compiles, fort doesn't
pure2-bugfix-for-non-local-function-expression: cppfort transpile failed
pure2-bugfix-for-non-local-initialization: cppfort transpile failed
pure2-bugfix-for-requires-clause-in-forward-declaration: ref compiles, fort doesn't
pure2-bugfix-for-template-argument: ref compiles, fort doesn't
pure2-bugfix-for-ufcs-arguments: ref compiles, fort doesn't
pure2-bugfix-for-ufcs-name-lookup: cppfort transpile failed
pure2-bugfix-for-ufcs-noexcept: cppfort transpile failed
pure2-bugfix-for-ufcs-sfinae: ref compiles, fort doesn't
pure2-bugfix-for-unbraced-function-expression: cppfort transpile failed
pure2-bugfix-for-variable-template: ref compiles, fort doesn't
pure2-chained-comparisons: ref compiles, fort doesn't
pure2-contracts: ref compiles, fort doesn't
pure2-default-arguments: ref compiles, fort doesn't
pure2-defaulted-comparisons-and-final-types: ref compiles, fort doesn't
pure2-enum: ref compiles, fort doesn't
pure2-for-loop-range-with-lambda: ref compiles, fort doesn't
pure2-function-body-reflection: cppfort transpile failed
pure2-function-multiple-forward-arguments: ref compiles, fort doesn't
pure2-function-single-expression-body-default-return: ref compiles, fort doesn't
pure2-function-typeids: ref compiles, fort doesn't
pure2-initialization-safety-with-else-if: ref compiles, fort doesn't
pure2-inspect-fallback-with-variant-any-optional: ref compiles, fort doesn't
pure2-inspect-generic-void-empty-with-variant-any-optional: ref compiles, fort doesn't
pure2-is-with-free-functions-predicate: ref compiles, fort doesn't
pure2-is-with-polymorphic-types: ref compiles, fort doesn't
pure2-is-with-unnamed-predicates: ref compiles, fort doesn't
pure2-is-with-variable-and-value: ref compiles, fort doesn't
pure2-last-use: cppfort transpile failed
pure2-look-up-parameter-across-unnamed-function: ref compiles, fort doesn't
pure2-main-args: fort compiles, ref doesn't
pure2-more-wildcards: ref compiles, fort doesn't
pure2-print: cppfort transpile failed
pure2-raw-string-literal-and-interpolation: cppfort transpile failed
pure2-regex_01_char_matcher: ref compiles, fort doesn't
pure2-regex_02_ranges: ref compiles, fort doesn't
pure2-regex_03_wildcard: ref compiles, fort doesn't
pure2-regex_04_start_end: ref compiles, fort doesn't
pure2-regex_05_classes: ref compiles, fort doesn't
pure2-regex_06_boundaries: ref compiles, fort doesn't
pure2-regex_07_short_classes: ref compiles, fort doesn't
pure2-regex_08_alternatives: ref compiles, fort doesn't
pure2-regex_09_groups: ref compiles, fort doesn't
pure2-regex_10_escapes: ref compiles, fort doesn't
pure2-regex_11_group_references: ref compiles, fort doesn't
pure2-regex_12_case_insensitive: ref compiles, fort doesn't
pure2-regex_13_possessive_modifier: ref compiles, fort doesn't
pure2-regex_14_multiline_modifier: ref compiles, fort doesn't
pure2-regex_15_group_modifiers: ref compiles, fort doesn't
pure2-regex_16_perl_syntax_modifier: ref compiles, fort doesn't
pure2-regex_17_comments: ref compiles, fort doesn't
pure2-regex_18_branch_reset: ref compiles, fort doesn't
pure2-regex_19_lookahead: ref compiles, fort doesn't
pure2-regex_20_lookbehind: ref compiles, fort doesn't
pure2-regex_21_atomic_patterns: ref compiles, fort doesn't
pure2-regex-general: cppfort transpile failed
pure2-repeated-call: ref compiles, fort doesn't
pure2-requires-clauses: ref compiles, fort doesn't
pure2-return-tuple-operator: ref compiles, fort doesn't
pure2-statement-scope-parameters: ref compiles, fort doesn't
pure2-stdio-with-raii: ref compiles, fort doesn't
pure2-stdio: ref compiles, fort doesn't
pure2-template-parameter-lists: ref compiles, fort doesn't
pure2-trailing-commas: ref compiles, fort doesn't
pure2-type-constraints: ref compiles, fort doesn't
pure2-types-basics: ref compiles, fort doesn't
pure2-types-down-upcast: ref compiles, fort doesn't
pure2-types-inheritance: ref compiles, fort doesn't
pure2-types-order-independence-and-nesting: ref compiles, fort doesn't
pure2-types-ordering-via-meta-functions: ref compiles, fort doesn't
pure2-types-smf-and-that-1-provide-everything: ref compiles, fort doesn't
pure2-types-smf-and-that-2-provide-mvconstruct-and-cpassign: ref compiles, fort doesn't
pure2-types-smf-and-that-3-provide-mvconstruct-and-mvassign: ref compiles, fort doesn't
pure2-types-smf-and-that-4-provide-cpassign-and-mvassign: ref compiles, fort doesn't
pure2-types-smf-and-that-5-provide-nothing-but-general-case: ref compiles, fort doesn't
pure2-types-that-parameters: output differs
pure2-types-value-types-via-meta-functions: ref compiles, fort doesn't
pure2-ufcs-member-access-and-chaining: ref compiles, fort doesn't
pure2-union: ref compiles, fort doesn't
pure2-unsafe: ref compiles, fort doesn't
pure2-variadics: ref compiles, fort doesn't
pure2-various-string-literals: ref compiles, fort doesn't
```

## Skipped Tests

```

```
