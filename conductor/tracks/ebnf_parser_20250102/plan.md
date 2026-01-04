# Plan: Spirit-Like EBNF Combinator Mapping

**Track**: ebnf_parser_20250102
**Goal**: Implement "Spirit-like" operator overloading for parser combinators to map EBNF exactly to C++.

---

## Phase 1: Operator Infrastructure
**Goal**: Implement C++ operator overloads for core combinators.

- [ ] Create `include/combinators/operators.hpp`
  - [ ] Define `operator>>` (Sequence) -> `seq`
  - [ ] Define `operator|` (Alternation) -> `alt`
  - [ ] Define `operator*` (Many) -> `many`
  - [ ] Define `operator+` (Many1) -> `many1`
  - [ ] Define `operator-` (Optional) -> `opt`
  - [ ] Define `operator[]` (Transform) -> `map`
  - [ ] Define `operator%` (List) -> `sep_by`
- [ ] Implement named implementation types (midpoints)
  - `SequenceParser`, `AlternativeParser`, `ManyParser`, etc.
  - Ensure they store their sub-parsers (preserves structure).

## Phase 2: Syntax Verification
**Goal**: Verify the syntax looks correct and compiles.

- [ ] Create `tests/parser_syntax_test.cpp`
  - [ ] Define a mock grammar using the operators.
  - [ ] Verify compilation.
  - [ ] Verify runtime behavior against standard combinators.

## Phase 3: Recursive Rule Support
**Goal**: Support recursive grammar rules (references).

- [ ] Implement `recursive_rule<P>` or `reference<P>` wrapper.
- [ ] Allow `rule = rule >> ...` syntax (likely via lambda or semantic action delay).

## Phase 4: Pilot Mapping (Basic Types)
**Goal**: Map a subset of `cpp2.ebnf` to `cpp2::parser::rules`.

- [ ] Define `namespace cpp2::parser::rules`
- [ ] Implement `integer_literal` using operators.
- [ ] Implement `identifier` using operators.
- [ ] Implement simple recursive rule (e.g. `paren_expr`).

## Phase 5: Full Grammar Mapping
**Goal**: Translate entire `grammar/cpp2.ebnf` to C++.

- [ ] `lexical_structure`
- [ ] `expressions` (precedence climbing map)
- [ ] `statements`
- [ ] `declarations`

---

## Deliverables
- `include/combinators/operators.hpp`: The operator grammar library.
- `tests/parser_syntax_test.cpp`: Proof of concept.
- `src/parser_rules.cpp` (or header): The defined rules.
