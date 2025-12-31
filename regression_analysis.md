# Cppfort Regression Analysis - 184 Corpus Files

## Executive Summary

- **Total Files**: 184
- **PASS**: 21 (11.4%)
- **CPPFORT_FAIL**: 48 (26.1%)
- **SEMANTIC_ERROR**: 14 (7.6%)
- **CPPFRONT_FAIL**: 12 (6.5%)
- **TIMEOUT**: 89 (48.4%)

## Critical Issue: Parser Infinite Loop

**89 files (48.4%) timeout after 5 seconds**, indicating a severe bug in cppfort's parsing or error recovery logic. These files cause cppfort to enter an infinite loop that never terminates.

### Timeout Categories

The timeouts occur in files with these features:
- UFCS (Universal Function Call Syntax) with templates
- Inspect expressions and pattern matching
- Function expressions with captures
- Complex type definitions and meta-functions
- Regex examples (all 21 regex files timeout)
- Type system features (inheritance, ordering, SMF)

## PASS Files (21)

Both cppfort and cppfront successfully compile these files:

1. mixed-bugfix-for-cpp2-comment-cpp1-sequence.cpp2
2. mixed-bugfix-for-literal-as-nttp.cpp2
3. mixed-default-arguments.cpp2
4. mixed-float-literals.cpp2
5. mixed-hello.cpp2
6. mixed-increment-decrement.cpp2
7. mixed-intro-example-three-loops.cpp2
8. mixed-intro-for-with-counter-include-last.cpp2
9. mixed-lifetime-safety-and-null-contracts.cpp2
10. mixed-parameter-passing-with-forward.cpp2
11. pure2-break-continue.cpp2
12. pure2-bugfix-for-memberwise-base-assignment.cpp2
13. pure2-bugfix-for-requires-clause-in-forward-declaration.cpp2
14. pure2-bugfix-for-requires-clause-unbraced-function-initializer.cpp2
15. pure2-bugfix-for-template-argument.cpp2
16. pure2-bugfix-for-variable-template.cpp2
17. pure2-chained-comparisons.cpp2
18. pure2-hashable.cpp2
19. pure2-hello.cpp2
20. pure2-synthesize-rightshift-and-rightshifteq.cpp2
21. pure2-trailing-comma-assert.cpp2

## CPPFORT_FAIL Files (48)

Cppfront succeeds but cppfort fails with parse/semantic errors. Key patterns:

### Namespace Issues
- `mixed-fixed-type-aliases.cpp2`: "Expected '=' after namespace name"
- Suggests namespace alias parsing is broken

### Function Expression Issues
Multiple function expression files fail with "Expected expression":
- `mixed-function-expression-and-std-for-each.cpp2`
- `mixed-function-expression-with-pointer-capture.cpp2`
- `mixed-function-expression-with-repeated-capture.cpp2`
- `mixed-initialization-safety-3.cpp2`
- `mixed-string-interpolation.cpp2`

### Template/Type Issues
- `pure2-concept-definition.cpp2`: "Expected '(' after function name"
- `pure2-contracts.cpp2`: "Expected ':' after parameter name"
- `pure2-template-parameter-lists.cpp2`: Template parsing failures
- `pure2-variadics.cpp2`: "Expected '>' after template parameters"

### Assert Contract Issues
All assert contract files fail with semantic analysis errors:
- `pure2-assert-expected-not-null.cpp2`: 7 semantic errors
- `pure2-assert-optional-not-null.cpp2`: 5 semantic errors
- `pure2-assert-shared-ptr-not-null.cpp2`: 5 semantic errors
- `pure2-assert-unique-ptr-not-null.cpp2`: 5 semantic errors

## SEMANTIC_ERROR Files (14)

Parser succeeds but semantic analyzer fails:

### Lifetime Safety
- `mixed-lifetime-safety-pointer-init-1-error.cpp2`: "Expected type"
- `mixed-lifetime-safety-pointer-init-2-error.cpp2`: "Expected type"
- `mixed-lifetime-safety-pointer-init-3-error.cpp2`: "Expected type"
- `pure2-lifetime-safety-pointer-init-1-error.cpp2`: "Expected type"
- `pure2-lifetime-safety-reject-null-error.cpp2`: "Expected type"

### Type System
- `pure2-bugfix-for-bad-decltype-error.cpp2`: "Expected ';' after expression"
- `pure2-bugfix-for-bad-parameter-error.cpp2`: "Expected '>' after template parameters"
- `pure2-bugfix-for-invalid-alias-error.cpp2`: "Expected type"
- `pure2-bugfix-for-namespace-error.cpp2`: "Expected type"
- `pure2-bugfix-for-naked-unsigned-char-error.cpp2`: "Expected ';' after variable declaration"
- `pure2-cpp1-multitoken-fundamental-types-error.cpp2`: "Expected ';' after variable declaration"
- `pure2-deduction-1-error.cpp2`: "Expected ';' or function body"
- `pure2-deduction-2-error.cpp2`: "Expected ';' after variable declaration"
- `pure2-bounds-safety-pointer-arithmetic-error.cpp2`: "Expected type"

## CPPFRONT_FAIL Files (12)

Cppfort succeeds but cppfront fails - these may be cppfront bugs or test cases specifically for cppfront bugs:

1. `mixed-bugfix-for-double-pound-else-error.cpp2`
2. `pure2-bugfix-for-assert-capture-error.cpp2`
3. `pure2-bugfix-for-bad-capture-error.cpp2`
4. `pure2-bugfix-for-bad-using-error.cpp2`
5. `pure2-bugfix-for-declaration-equal-error.cpp2`
6. `pure2-bugfix-for-out-this-nonconstructor-error.cpp2`
7. `pure2-bugfix-for-cpp1-prefix-expression-error.cpp2`
8. `pure2-initialization-loop-2-error.cpp2`
9. `pure2-initialization-loop-error.cpp2`
10. `pure2-return-tuple-no-identifier-error.cpp2`
11. `pure2-return-tuple-no-type-error.cpp2`
12. `pure2-statement-parse-error.cpp2`

## Priority Issues to Fix

### P0 - Infinite Loop (48% of files)
1. Fix parser infinite loop in error recovery
2. Investigate why process doesn't exit after emitting errors
3. Check for while/for loops without proper termination conditions

### P1 - Function Expressions
Multiple failures in function expression parsing:
- Capture lists
- Lambda expressions in type context
- UFCS with function expressions

### P2 - Namespace Parsing
- Namespace aliases appear broken
- "Expected '=' after namespace name"

### P3 - Template Parsing
- Template parameter lists
- Template argument deduction
- Variadic templates

### P4 - Semantic Analysis
- Type checking in contracts
- Lifetime safety annotations
- Assert contract semantics

## Conclusion

Cppfort has a **critical infinite loop bug** affecting nearly half the corpus. This must be fixed before other issues can be properly diagnosed. The 21 passing files demonstrate basic functionality works, but complex C++2 features cause hangs or errors.

