# Implementation Plan: EBNF-Driven Parser Rewrite

## Overview

Rewrite the existing 5,700+ line hand-written recursive descent parser to be driven 
by a formal EBNF grammar specification. This enables:
- Grammar verification and validation
- Automatic parser generation/verification
- Cleaner separation of syntax from semantics
- Easier grammar extensions
- Better error messages with grammar context

**Current State:** Hand-written parser in `src/parser.cpp` (5,772 lines)
**Target State:** EBNF grammar file + grammar-driven parser infrastructure

---

## Phase 1: Grammar Extraction and Specification [checkpoint: TBD]

### 1.1 Document Existing Grammar

- [ ] Extract declaration grammar from `Parser::declaration()` (~250 lines)
  - Variable declarations: `name : type = expr`
  - Function declarations: `name : (params) -> type = body`
  - Type declarations: `name : type = { members }`
  - Namespace declarations
  - Using/import declarations
  - C++1 passthrough detection

- [ ] Extract statement grammar from `Parser::statement()` (~220 lines)
  - Block statements
  - Control flow: if/while/do/for/switch
  - Pattern matching: inspect
  - Contracts: pre/post/assert
  - Exception handling: try/throw

- [ ] Extract expression grammar (precedence levels)
  - Level 1: Assignment (`=`, `:=`, `+=`, etc.)
  - Level 2: Pipeline (`|>`)
  - Level 3: Ternary (`? :`)
  - Level 4: Logical OR (`||`)
  - Level 5: Logical AND (`&&`)
  - Level 6: Bitwise OR (`|`)
  - Level 7: Bitwise XOR (`^`)
  - Level 8: Bitwise AND (`&`)
  - Level 9: Equality (`==`, `!=`)
  - Level 10: Comparison (`<`, `>`, `<=`, `>=`, `<=>`)
  - Level 11: Range (`..`, `..<`, `..=`)
  - Level 12: Shift (`<<`, `>>`)
  - Level 13: Addition (`+`, `-`)
  - Level 14: Multiplication (`*`, `/`, `%`)
  - Level 15: Prefix (`!`, `-`, `~`, `&`, `*`, `++`, `--`)
  - Level 16: Postfix (call, member, subscript, `++`, `--`)
  - Level 17: Primary (literals, identifiers, grouping)

- [ ] Extract type grammar from `Parser::type()` (~250 lines)
  - Builtin types
  - User-defined types
  - Pointer/reference types
  - Template types
  - Function types
  - Qualified types (const, volatile)

### 1.2 Create EBNF Grammar File

- [ ] Create `grammar/cpp2.ebnf` with formal grammar
- [ ] Add grammar validation tests
- [ ] Document ambiguities and resolution rules
- [ ] Add precedence/associativity annotations

### Success Criteria
- Complete EBNF grammar covering all current parser functionality
- Grammar passes validation (no unreachable rules, no ambiguities)
- Documentation of all disambiguation rules

---

## Phase 2: Grammar Infrastructure [checkpoint: TBD]

### 2.1 EBNF Parser

- [ ] Create `include/grammar/ebnf_parser.hpp`
  - EBNF lexer for grammar files
  - EBNF AST representation
  - Grammar rule storage

- [ ] Create `src/grammar/ebnf_parser.cpp`
  - Parse EBNF syntax: `rule ::= alternatives ;`
  - Handle: sequences, alternatives, optionals, repetitions
  - Support: grouping, character classes, literals
  - Parse precedence annotations

- [ ] Unit tests for EBNF parser
  - Basic rule parsing
  - Complex alternatives
  - Recursive rules
  - Error reporting

### 2.2 Grammar Data Structures

- [ ] Create `include/grammar/grammar.hpp`
  ```cpp
  struct GrammarRule {
      std::string name;
      std::vector<Alternative> alternatives;
      int precedence = 0;
      Associativity assoc = Associativity::Left;
  };
  
  struct Alternative {
      std::vector<Symbol> symbols;
      std::optional<SemanticAction> action;
  };
  
  struct Symbol {
      enum Kind { Terminal, NonTerminal, Optional, Star, Plus };
      Kind kind;
      std::string name;
      TokenType token; // for terminals
  };
  ```

- [ ] Create `include/grammar/grammar_set.hpp`
  - First/Follow set computation
  - Nullable detection
  - LL(k) analysis utilities

### Success Criteria
- EBNF parser can load and validate grammar files
- Grammar data structures support all Cpp2 constructs
- First/Follow sets computed correctly

---

## Phase 3: Parser Generator Framework [checkpoint: TBD]

### 3.1 Parse Table Generation

- [ ] Create `include/grammar/parse_table.hpp`
  - LL(1) parse table representation
  - Conflict detection and reporting
  - Table serialization for caching

- [ ] Create `src/grammar/parse_table_gen.cpp`
  - Generate LL(1) parse tables
  - Handle left-recursion elimination
  - Handle left-factoring
  - Report conflicts with grammar context

### 3.2 Packrat Parser Support (for backtracking)

- [ ] Create `include/grammar/packrat.hpp`
  - Memoization table for parse results
  - Backtracking support for ambiguous grammars
  - Cut operator for committed choice

- [ ] Create `src/grammar/packrat.cpp`
  - Implement packrat parsing algorithm
  - Memory-efficient memoization
  - Integration with existing Token stream

### 3.3 Parser Driver

- [ ] Create `include/grammar/grammar_parser.hpp`
  - Generic grammar-driven parser
  - Semantic action hooks
  - Error recovery strategies

- [ ] Create `src/grammar/grammar_parser.cpp`
  - Table-driven parsing loop
  - AST construction via semantic actions
  - Error synchronization points

### Success Criteria
- Parse tables generated for Cpp2 grammar
- Packrat parser handles ambiguous constructs
- Parser driver produces correct AST for test cases

---

## Phase 4: Semantic Actions [checkpoint: TBD]

### 4.1 AST Builder Actions

- [ ] Create `include/grammar/ast_actions.hpp`
  - Semantic action interface
  - AST node factory methods
  - Action registration mechanism

- [ ] Create `src/grammar/ast_actions.cpp`
  - Declaration builders
  - Statement builders  
  - Expression builders
  - Type builders

### 4.2 Action DSL

- [ ] Design action syntax for EBNF file
  ```ebnf
  function_decl ::= 
      IDENTIFIER ':' param_list '->' type '=' body
      { make_function($1, $3, $5, $7) }
      ;
  ```

- [ ] Implement action code generation
  - Parse action expressions
  - Generate C++ action code
  - Type-safe argument passing

### Success Criteria
- Semantic actions produce identical AST to current parser
- Action DSL is concise and type-safe
- All 99 parser methods have equivalent actions

---

## Phase 5: Migration and Integration [checkpoint: TBD]

### 5.1 Parallel Implementation

- [ ] Create `include/parser_v2.hpp` - new grammar-driven parser
- [ ] Create `src/parser_v2.cpp` - parser driver implementation
- [ ] Load grammar from `grammar/cpp2.ebnf` at startup
- [ ] Implement same public API as existing Parser

### 5.2 Test Harness

- [ ] Create comparison test harness
  - Run both parsers on same input
  - Compare resulting AST
  - Report differences

- [ ] Port all existing parser tests
- [ ] Add grammar-specific tests
  - Ambiguity resolution
  - Error recovery
  - Precedence handling

### 5.3 Gradual Migration

- [ ] Add `--parser-v2` flag to cppfort
- [ ] Run corpus tests with both parsers
- [ ] Identify and fix discrepancies
- [ ] Performance comparison

### Success Criteria
- New parser passes all existing tests
- No regressions in corpus parsing
- Performance within 20% of hand-written parser

---

## Phase 6: Error Recovery and Diagnostics [checkpoint: TBD]

### 6.1 Grammar-Aware Errors

- [ ] Extract expected tokens from parse state
- [ ] Generate "expected X, found Y" messages
- [ ] Include grammar rule context in errors

### 6.2 Error Recovery

- [ ] Implement panic-mode recovery with sync points
- [ ] Add grammar annotations for recovery
  ```ebnf
  statement ::= ... @sync(';', '}') ;
  ```
- [ ] Implement Burke-Fisher error repair (optional)

### 6.3 Suggestion Engine

- [ ] Detect common mistakes (missing semicolon, etc.)
- [ ] Suggest fixes based on grammar
- [ ] Integration with IDE diagnostics

### Success Criteria
- Error messages include grammar context
- Parser recovers gracefully from errors
- Common mistakes get helpful suggestions

---

## Phase 7: Optimization and Finalization [checkpoint: TBD]

### 7.1 Performance Optimization

- [ ] Profile grammar-driven parser
- [ ] Optimize hot paths in parse loop
- [ ] Consider partial hand-optimization for expressions
- [ ] Cache compiled grammar

### 7.2 Grammar Tooling

- [ ] Grammar pretty-printer
- [ ] Railroad diagram generator
- [ ] Grammar diff tool

### 7.3 Documentation

- [ ] Document grammar syntax
- [ ] Document extension process
- [ ] Document semantic action interface

### 7.4 Deprecate Old Parser

- [ ] Mark old parser as deprecated
- [ ] Remove old parser (after stabilization)
- [ ] Update all dependent code

### Success Criteria
- Parser performance within 10% of hand-written
- Complete grammar documentation
- Clean removal of old parser code

---

## Track Completion Checklist

- [ ] Phase 1: Grammar extracted and documented
- [ ] Phase 2: EBNF infrastructure complete
- [ ] Phase 3: Parser generator working
- [ ] Phase 4: Semantic actions implemented
- [ ] Phase 5: Migration complete, tests passing
- [ ] Phase 6: Error recovery improved
- [ ] Phase 7: Optimization and cleanup done
- [ ] All corpus tests passing with new parser
- [ ] Old parser removed
- [ ] Documentation complete

---

## Risk Mitigation

**Risk: Performance regression**
- Mitigation: Keep hand-written expression parser as fast-path
- Fallback: Hybrid approach with critical paths hand-optimized

**Risk: Grammar ambiguities**
- Mitigation: Packrat parser with backtracking
- Fallback: Disambiguation predicates in grammar

**Risk: Missing edge cases**
- Mitigation: Comprehensive corpus testing
- Fallback: Maintain old parser for comparison during transition

---

## Estimated Timeline

| Phase | Effort | Dependencies |
|-------|--------|--------------|
| Phase 1 | 2-3 days | None |
| Phase 2 | 3-4 days | Phase 1 |
| Phase 3 | 4-5 days | Phase 2 |
| Phase 4 | 3-4 days | Phase 3 |
| Phase 5 | 3-4 days | Phase 4 |
| Phase 6 | 2-3 days | Phase 5 |
| Phase 7 | 2-3 days | Phase 6 |

**Total: ~20-26 days**
