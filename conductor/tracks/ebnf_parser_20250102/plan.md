# Implementation Plan: Combinator-Based Parser Rewrite

## Overview

Rewrite the existing hand-written recursive descent parser using orthogonal parser combinators.
The formal grammar is defined in `grammar/cpp2.combinators.md` and `grammar/cpp2.ebnf`.

**Grammar Reference:** `grammar/cpp2.combinators.md` (1,110 lines - combinator specification)
**Formal EBNF:** `grammar/cpp2.ebnf` (653 lines - formal grammar)
**Current State:** Hand-written parser in `src/parser.cpp` (5,800+ lines)
**Target State:** Combinator-driven parser using `include/combinators/` infrastructure

---

## Combinator Primitives

The parser will be built from these orthogonal combinators (see `grammar/cpp2.combinators.md`):

| Combinator | Type Signature | Description |
|------------|---------------|-------------|
| `seq(a, b, ...)` | `Parser<tuple<A,B,...>>` | Sequencing |
| `alt(a, b, ...)` | `Parser<variant<A,B,...>>` | Alternation |
| `many(p)` | `Parser<vector<T>>` | Zero or more |
| `many1(p)` | `Parser<vector<T>>` | One or more |
| `opt(p)` | `Parser<optional<T>>` | Optional |
| `sep_by(p, d)` | `Parser<vector<T>>` | Separated list |
| `between(l, p, r)` | `Parser<T>` | Bracketed |
| `map(p, f)` | `Parser<U>` | Transform |
| `try_(p)` | `Parser<T>` | Backtrack on fail |

---

## Phase 1: Grammar Validation ✓ [COMPLETE]

### Completed
- [x] Extract grammar from `src/parser.cpp` 
- [x] Create `grammar/cpp2.ebnf` (653 lines)
- [x] Create `grammar/cpp2.combinators.md` (1,110 lines)
- [x] Document disambiguation rules
- [x] Define precedence levels (16 expression levels)

### Grammar Files
```
grammar/
├── cpp2.combinators.md   # Combinator specification (authoritative)
└── cpp2.ebnf             # Formal EBNF (reference)
```

---

## Phase 2: Combinator Infrastructure [checkpoint: TBD]

### 2.1 Core Combinator Types

Location: `include/combinators/`

```cpp
// Parser result
template<typename T>
using ParseResult = std::expected<T, ParseError>;

// Parser type
template<typename T>
using Parser = std::function<ParseResult<T>(TokenStream&)>;

// Primitive combinators (from grammar/cpp2.combinators.md)
template<typename... Ps>
auto seq(Ps... parsers) -> Parser<std::tuple<parse_result_t<Ps>...>>;

template<typename... Ps>
auto alt(Ps... parsers) -> Parser<std::variant<parse_result_t<Ps>...>>;

template<typename P>
auto many(P parser) -> Parser<std::vector<parse_result_t<P>>>;

template<typename P>
auto opt(P parser) -> Parser<std::optional<parse_result_t<P>>>;
```

### 2.2 Token Matchers

```cpp
// Terminal matchers
auto token(TokenType t) -> Parser<Token>;
auto keyword(std::string_view kw) -> Parser<Token>;
auto punct(std::string_view p) -> Parser<Token>;

// From grammar/cpp2.combinators.md:
// identifier = token(Identifier) | filter(token(Keyword), is_contextual_keyword)
auto identifier() -> Parser<Token>;
```

### 2.3 Expression Combinators

```cpp
// Precedence climbing for expressions (16 levels in cpp2.combinators.md)
auto expression() -> Parser<ExprPtr>;
auto precedence_climb(int min_prec) -> Parser<ExprPtr>;

// From grammar spec:
// assignment = logical_or [ assignment_op assignment ]
// pipeline = ternary { "|>" ternary }
auto assignment() -> Parser<ExprPtr>;
auto pipeline() -> Parser<ExprPtr>;
```

### Success Criteria
- Combinator types compile with C++20
- Token matchers work with existing lexer
- Basic expression parsing passes tests

---

## Phase 3: Grammar Implementation [checkpoint: TBD]

### 3.1 Declarations (from `grammar/cpp2.combinators.md`)

```cpp
// declaration = alt(
//     namespace_decl,
//     function_decl,
//     type_decl,
//     variable_decl,
//     using_decl,
//     cpp1_passthrough
// )
auto declaration() -> Parser<DeclPtr> {
    return alt(
        namespace_decl(),
        function_decl(),
        type_decl(),
        variable_decl(),
        using_decl(),
        cpp1_passthrough()
    );
}

// function_decl = seq(
//     identifier,
//     opt(template_params),
//     punct(":"),
//     param_list,
//     opt(seq(punct("->"), return_type)),
//     opt(contracts),
//     function_body
// )
```

### 3.2 Statements (from `grammar/cpp2.combinators.md`)

```cpp
// statement = alt(
//     block_stmt,
//     if_stmt,
//     while_stmt,
//     for_stmt,
//     inspect_stmt,
//     return_stmt,
//     contract_stmt,
//     expression_stmt
// )
```

### 3.3 Types (from `grammar/cpp2.combinators.md`)

```cpp
// type = alt(
//     builtin_type,
//     qualified_id,
//     pointer_type,
//     reference_type,
//     array_type,
//     function_type,
//     template_type
// )
```

### Success Criteria
- All grammar rules from `cpp2.combinators.md` implemented
- Parser produces identical AST to current implementation
- All regression tests pass

---

## Phase 4: AST Construction [checkpoint: TBD]

### 4.1 Semantic Actions

```cpp
// Transform parse results into AST nodes
auto function_decl() -> Parser<DeclPtr> {
    return seq(
        identifier(),
        opt(template_params()),
        punct(":"),
        param_list(),
        opt(seq(punct("->"), return_type())),
        opt(contracts()),
        function_body()
    ) |> map([](auto&& parts) {
        auto [name, tparams, _, params, ret, contracts, body] = parts;
        return make_function_decl(name, tparams, params, ret, contracts, body);
    });
}
```

### 4.2 Error Recovery

```cpp
// Sync points from grammar spec
// statement ::= ... @sync(';', '}')
auto statement_with_recovery() -> Parser<StmtPtr> {
    return recover(statement(), sync_to(punct(";"), punct("}")));
}
```

### Success Criteria
- AST construction integrated with combinators
- Error recovery at statement/declaration boundaries
- Error messages include grammar context

---

## Phase 5: Migration [checkpoint: TBD]

### 5.1 Parallel Implementation
- [ ] Create `src/parser_combinators.cpp`
- [ ] Implement all combinators from `grammar/cpp2.combinators.md`
- [ ] Same public API as existing Parser class

### 5.2 Test Harness
- [ ] Run both parsers on regression corpus
- [ ] Compare AST output
- [ ] Head-to-head with cppfront reference

### 5.3 Switch Over
- [ ] Add `--parser=combinators` flag
- [ ] Validate against 158 reference tests
- [ ] Performance benchmarking

### Success Criteria
- All 26 passing tests still pass
- No regressions in 158-test corpus
- Performance within 20% of hand-written

---

## Phase 6: Optimization [checkpoint: TBD]

### 6.1 Memoization
- Packrat-style memoization for backtracking
- Cache expression parse results

### 6.2 Fast Paths
- Inline token matching for hot paths
- Specialized expression precedence climber

### 6.3 Grammar Compilation
- Pre-compute first/follow sets
- Static parse tables where applicable

### Success Criteria
- Performance within 10% of hand-written
- Memory usage acceptable

---

## Reference Materials

| Document | Purpose |
|----------|---------|
| `grammar/cpp2.combinators.md` | Authoritative combinator grammar |
| `grammar/cpp2.ebnf` | Formal EBNF reference |
| `include/combinators/` | Combinator implementation |
| `tests/reference/` | cppfront reference corpus |
| `tests/results/H2H_SUMMARY.md` | Head-to-head baseline |

---

## Current Baseline

From head-to-head testing against cppfront:
- **26/158 (16.5%)** tests passing
- **110** fail due to C++ compile differences  
- **14** fail at transpilation
- **7** have output differences
- **1** compiles when reference doesn't

---

## Risk Mitigation

**Risk: Combinator overhead**
- Mitigation: Inline critical parsers, memoization
- Fallback: Hybrid with hand-written expression parser

**Risk: Grammar coverage gaps**
- Mitigation: Reference corpus validation
- Fallback: Incremental migration per construct

**Risk: AST incompatibility**
- Mitigation: AST comparison test harness
- Fallback: Adapter layer for existing AST types
