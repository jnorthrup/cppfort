# Parser Orchestration: EBNF to Combinator Mappings

**Version**: cppfort EBNF standard v0.1.0 (2026-01-03)

**Objective**: Document how EBNF grammar design drives parser combinator orchestration for cppfort, enabling idempotent cppfront AST dumps to score loss against isomorph normalizations.

---

## Table of Contents

1. [EBNF Design](#1-ebnf-design)
2. [Combinator Mappings](#2-combinator-mappings)
3. [EBNF Standard Tracking](#3-ebnf-standard-tracking)
4. [AST Isomorph Normalization](#4-ast-isomorph-normalization)
5. [Semantic Loss Scoring](#5-semantic-loss-scoring)

---

## 1. EBNF Design

### 1.1 Overview

Cpp2 uses a **Unified Declaration Syntax**: `Name : Kind = Value`

This pattern applies consistently to variables, functions, types, and namespaces.

### 1.2 Lexical Structure

```ebnf
identifier          ::= [a-zA-Z_] { [a-zA-Z0-9_] } ;
integer_literal     ::= ? standard C++ integer literal ? ;
float_literal       ::= ? standard C++ floating point literal ? ;
string_literal      ::= ? standard C++ string literal, with (...)$ interpolation ? ;
char_literal        ::= ? standard C++ char literal ? ;
user_defined_literal ::= ( integer_literal | float_literal ) identifier ;
```

### 1.3 Top-Level Structure

```ebnf
translation_unit    ::= { declaration } ;

declaration         ::= namespace_declaration
                      | template_declaration
                      | type_declaration
                      | function_declaration
                      | variable_declaration
                      | using_declaration
                      | import_declaration
                      | statement
                      ;

name_declaration    ::= identifier [ template_params ] ":" declaration_body ;
```

### 1.4 Template Parameters

```ebnf
template_params     ::= "<" template_param_list ">" ;
template_param_list ::= template_param { "," template_param } [ "," ] ;
template_param      ::= identifier [ "..." ] [ ":" type_constraint ] [ "=" default_value ]
                      | "_" [ "..." ]
                      ;

type_constraint     ::= type_specifier | "_" | "type" ;
default_value       ::= type_specifier | expression ;
```

### 1.5 Function Declarations

```ebnf
function_declaration ::= identifier ":" function_signature [ contracts ] function_body ;

function_signature  ::= [ template_params ] "(" [ parameter_list ] ")"
                        [ throws_spec ] [ return_spec ] [ requires_clause ] ;

parameter_list      ::= parameter { "," parameter } [ "," ] ;
parameter           ::= [ param_qualifiers ] param_name [ ":" type_specifier ] [ "=" default_arg ]
                      | [ param_qualifiers ] param_name "..."
                      | [ param_qualifiers ] "this"
                      ;

param_qualifiers    ::= { param_qualifier } ;
param_qualifier     ::= "in" | "copy" | "inout" | "out" | "move" | "forward"
                      | "in_ref" | "forward_ref"
                      | "virtual" | "override" | "implicit"
                      ;

return_spec         ::= "->" return_list ;
return_list         ::= return_type | "(" named_return { "," named_return } ")" ;
named_return        ::= identifier ":" type_specifier ;
```

### 1.6 Type Declarations

```ebnf
type_declaration    ::= identifier ":" [ metafunctions ] "type" [ template_params ]
                        [ base_types ] "=" type_body ;

metafunctions       ::= { metafunction } ;
metafunction        ::= "@" identifier ;

base_types          ::= ":" type_specifier { "," type_specifier } ;
type_body           ::= "{" { type_member } "}" ;
```

### 1.7 Statements

```ebnf
statement           ::= block_statement
                      | if_statement
                      | loop_statement
                      | switch_statement
                      | inspect_statement
                      | return_statement
                      | throw_statement
                      | try_statement
                      | contract_statement
                      | expression_statement
                      | declaration
                      | ";"
                      ;

block_statement     ::= "{" { statement } "}" ;

if_statement        ::= "if" [ "constexpr" ] expression block_statement
                        [ "else" ( block_statement | if_statement ) ] ;

loop_statement      ::= [ identifier ":" ] [ loop_parameters ] loop_kind ;

loop_parameters     ::= "(" param_declaration { "," param_declaration } ")" ;
loop_kind           ::= while_loop | for_loop | do_loop ;

while_loop          ::= "while" expression [ "next" expression ] block_statement ;
for_loop            ::= "for" expression [ "next" expression ] "do" "(" parameter ")" block_statement ;
do_loop             ::= "do" block_statement [ "next" expression ] "while" expression ";" ;
```

### 1.8 Expressions

```ebnf
expression          ::= assignment_expression ;

assignment_expression
                    ::= ternary_expression [ assignment_operator assignment_expression ] ;
assignment_operator ::= "=" | "+=" | "-=" | "*=" | "/=" | "%="
                      | "<<=" | ">>=" | "&=" | "|=" | "^=" ;

ternary_expression  ::= logical_or_expression [ "?" expression ":" ternary_expression ] ;

logical_or_expression ::= logical_and_expression { "||" logical_and_expression } ;
logical_and_expression ::= bitwise_or_expression { "&&" bitwise_or_expression } ;
bitwise_or_expression ::= bitwise_xor_expression { "|" bitwise_xor_expression } ;
bitwise_xor_expression ::= bitwise_and_expression { "^" bitwise_and_expression } ;
bitwise_and_expression ::= equality_expression { "&" equality_expression } ;

equality_expression ::= comparison_expression { ( "==" | "!=" ) comparison_expression } ;
comparison_expression ::= range_expression { ( "<" | ">" | "<=" | ">=" ) range_expression } ;
range_expression    ::= shift_expression { ( "..=" | "..<" ) shift_expression } ;
shift_expression    ::= addition_expression { ( "<<" | ">>" ) addition_expression } ;
addition_expression ::= multiplication_expression { ( "+" | "-" ) multiplication_expression } ;
multiplication_expression ::= prefix_expression { ( "*" | "/" | "%" ) prefix_expression } ;

prefix_expression   ::= postfix_expression
                      | "await" prefix_expression
                      | "launch" prefix_expression
                      | "select" select_body
                      | ( "move" | "forward" | "copy" ) prefix_expression
                      | ( "+" | "-" | "!" | "~" | "++" | "--" | "&" | "*" ) prefix_expression
                      ;

postfix_expression  ::= primary_expression { postfix_op } ;

postfix_op          ::= "(" [ argument_list ] ")"
                      | "<" template_arg_list ">" [ "(" [ argument_list ] ")" ]
                      | "." member_access
                      | ".." identifier [ "(" [ argument_list ] ")" ]
                      | "[" expression "]"
                      | "*" | "&" | "++" | "--" | "$" | "..."
                      | "as" type_specifier
                      | "is" is_pattern
                      | "::" identifier
                      ;

primary_expression  ::= literal
                      | identifier_expression
                      | "this" | "that" | "_"
                      | "(" expression_or_tuple ")"
                      | "[" list_or_lambda "]"
                      | "{" struct_initializer "}"
                      | "inspect" inspect_expression
                      | "@" identifier [ "(" [ argument_list ] ")" ]
                      | ":" function_expression
                      ;
```

### 1.9 Pattern Matching

```ebnf
inspect_expression  ::= expression [ "->" type_specifier ] "{"
                        { inspect_arm } "}" ;
inspect_arm         ::= pattern "=>" statement ;
pattern             ::= "_"
                      | identifier
                      | identifier ":" type_specifier
                      | "is" type_specifier
                      | "is" "(" expression ")"
                      | "is" literal
                      | expression
                      ;
```

### 1.10 Type Specifiers

```ebnf
type_specifier      ::= function_type | pointer_type | qualified_type ;

function_type       ::= "(" [ param_type_list ] ")" "->" [ return_qualifier ] type_specifier ;
pointer_type        ::= "*" [ "const" ] type_specifier ;
qualified_type      ::= basic_type { "::" identifier [ template_args ] } { "*" | "&" } ;

basic_type          ::= [ type_modifier { type_modifier } ] type_name [ template_args ]
                      | "auto"
                      | "_" [ "is" type_constraint ]
                      | "type"
                      | "decltype" "(" expression ")"
                      | "const" type_specifier
                      ;

template_args       ::= "<" [ template_arg { "," template_arg } ] ">" ;
template_arg        ::= type_specifier | expression | identifier "(" expression ")" ;
```

### 1.11 Operator Precedence

| Level | Operators | Associativity |
|-------|-----------|---------------|
| 1 | `()` `[]` `.` `..` `::` `++` `--` `*` `&` `$` `...` `as` `is` | Left |
| 2 | Prefix: `++` `--` `+` `-` `!` `~` `*` `&` `await` `move` `forward` `copy` | Right |
| 3 | `*` `/` `%` | Left |
| 4 | `+` `-` | Left |
| 5 | `<<` `>>` | Left |
| 6 | `..=` `..<` | Left |
| 7 | `<` `>` `<=` `>=` | Left |
| 8 | `==` `!=` | Left |
| 9 | `&` (bitwise) | Left |
| 10 | `^` | Left |
| 11 | `|` | Left |
| 12 | `&&` | Left |
| 13 | `||` | Left |
| 14 | `?` `:` | Right |
| 15 | `=` `+=` `-=` etc. | Right |

---

## 2. Combinator Mappings

### 2.1 Overview

The parser is implemented using **orthogonal parser combinators**. Each EBNF rule maps to a composition of primitive combinators from the `include/combinators/` infrastructure.

### 2.2 Primitive Combinators

```
# Sequencing
seq(a, b, ...)      → a then b then ...
                    → Parser<std::tuple<A, B, ...>>

# Alternation
alt(a, b, ...)      → a or b or ...
                    → Parser<std::variant<A, B, ...>>

# Repetition
many(p)             → zero or more p → Parser<std::vector<T>>
many1(p)            → one or more p → Parser<std::vector<T>>
opt(p)              → zero or one p → Parser<std::optional<T>>

# Lookahead
peek(p)             → match p without consuming → Parser<bool>
not_peek(p)         → fail if p matches → Parser<()>

# Terminals
token(T)            → match token type T → Parser<Token>
keyword(K)          → match keyword K → Parser<Token>
punct(P)            → match punctuation P → Parser<Token>

# Transformations
map(p, f)           → apply f to result of p → Parser<U>
filter(p, pred)     → fail if pred(result) is false → Parser<T>

# Combinators
sep_by(p, delim)    → p (delim p)* → Parser<std::vector<T>>
sep_by1(p, delim)   → p (delim p)+ → Parser<std::vector<T>>
between(l, p, r)    → l p r, return p → Parser<T>

# Error handling
try_(p)             → backtrack on failure → Parser<T>
label(p, msg)       → replace error message → Parser<T>
recover(p, sync)    → on failure, skip to sync point → Parser<T>
```

### 2.3 Combinator Library

The cppfort combinator library provides zero-copy, lazy-evaluation operations:

| Category | Combinators | Description |
|----------|-------------|-------------|
| **Structural** | `take`, `skip`, `slice`, `split`, `chunk`, `window` | Sequence manipulation |
| **Transformation** | `map`, `filter`, `enumerate`, `zip` | Element-wise operations |
| **Reduction** | `fold`, `reduce`, `find`, `all`, `any` | Terminal operations |
| **Parsing** | `byte`, `bytes`, `until`, `le_i16`, `be_i16` | Binary parsing |

### 2.4 Mapping EBNF to Combinators

| EBNF Construct | Combinator Implementation |
|----------------|---------------------------|
| `seq(a, b)` | `combinators::seq<A, B>` |
| `alt(a, b)` | `combinators::alt<A, B>` |
| `many(p)` | `combinators::many<P>` |
| `many1(p)` | `combinators::many1<P>` |
| `opt(p)` | `combinators::opt<P>` |
| `sep_by(p, d)` | `combinators::sep_by<P, D>` |
| `between(l, p, r)` | `combinators::between<L, P, R>` |
| `try_(p)` | `combinators::try_<P>` |
| `map(p, f)` | `p \|> combinators::map(f)` |
| `filter(p, pred)` | `p \|> combinators::filter(pred)` |
| `peek(p)` | `combinators::lookahead<P>` |
| `not_peek(p)` | `combinators::not_followed_by<P>` |
| `token(T)` | `combinators::token<T>` |
| `punct(P)` | `combinators::punct<P>` |
| `keyword(K)` | `combinators::keyword<K>` |
| `label(p, msg)` | `combinators::labeled<P>(msg)` |
| `recover(p, s)` | `combinators::recover<P, S>` |

### 2.5 Semantic Actions (Map Functions)

```javascript
// Binary expression folder
fold_left_to_binary = (parts) => {
    let [first, rest] = parts;
    return rest.fold(first, (acc, [op, rhs]) => BinaryExpr(acc, op, rhs));
}

// Pipeline folder: a |> f |> g => PipelineExpr
fold_left_to_pipeline = (parts) => {
    let [first, rest] = parts;
    return rest.fold(first, (acc, [_, rhs]) => PipelineExpr(acc, rhs));
}

// Postfix folder
fold_left_to_postfix = (parts) => {
    let [base, ops] = parts;
    return ops.fold(base, (acc, op) => apply_postfix(acc, op));
}
```

### 2.6 Zero-Copy Invariants

The combinator library guarantees:
- **No allocation in structural ops** - `take`, `skip`, `slice` return views only
- **Lazy until terminal** - Transformation combinators defer work
- **View lifetime** - Views don't extend lifetime of underlying data

---

## 3. EBNF Standard Tracking

### 3.1 Versioning

| Version | Date | Changes |
|---------|------|---------|
| v0.1.0 | 2026-01-03 | Initial EBNF standard for cppfort |
| | 2025-01-02 | Corresponds to parser implementation |

### 3.2 Canonical Sources

The EBNF standard is maintained in:
1. **Combinator Grammar**: `grammar/cpp2.combinators.md` (authoritative for parser)
2. **Formal EBNF**: `docs/CPP2_GRAMMAR.md` (human-readable reference)
3. **This Document**: `conductor/PARSER_ORCHESTRATION.md` (unified reference)

### 3.3 Grammar Coverage

All cpp2 language features are covered:
- ✅ Unified template syntax (`name: <T> (params)`)
- ✅ For-do loops (`for collection do(item) { }`)
- ✅ Inspect pattern matching (`inspect value -> type { is pattern = result }`)
- ✅ Metafunction type decorators (16 metafunctions)
- ✅ All operators and precedence levels
- ✅ Template parameters and constraints
- ✅ Contracts (pre, post, assert)
- ✅ String interpolation
- ✅ Range operators (`..=`, `..<`)
- ✅ UFCS (Unified Function Call Syntax)

---

## 4. AST Isomorph Normalization

### 4.1 Overview

A **graph isomorph** is a structural pattern in the Clang AST that has a canonical mapping to an MLIR region.

**Properties**:
- **Structural equivalence**: Isomorphic AST subgraphs have same node types and edge relationships
- **Semantic preservation**: Isomorphic graphs represent the same computation
- **Normalization**: Multiple C++ syntactic forms map to the same isomorph

### 4.2 Extraction Process

1. **Parse Clang AST**: Parse `-Xclang -ast-dump` output into tree structure
2. **Identify Subgraphs**: Extract maximal connected subgraphs
3. **Compute Signature**: Hash of (node_types, edge_structure, semantic_attributes)
4. **Normalize**: α-renaming, type normalization, CFG normalization

### 4.3 Common Isomorph Patterns

| AST Pattern | MLIR Region | Confidence |
|-------------|-------------|------------|
| `FunctionDecl` → `CompoundStmt` | `cpp2.func { ... }` | 1.0 |
| `IfStmt` with both branches | `cpp2.if { ... } else { ... }` | 1.0 |
| `WhileStmt` | `cpp2.loop { cpp2.if %cond { ... } else { cpp2.break } }` | 0.9 |
| `ForStmt` (range-based) | `cpp2.loop { ... }` | 0.85 |
| `ReturnStmt` | `cpp2.return` | 1.0 |
| `VarDecl` with init | `cpp2.var` | 1.0 |
| `CallExpr` (free function) | `cpp2.call` or `cpp2.ufcs_call` | 0.9 |
| `BinaryOperator` (+/-/*//) | `cpp2.add` / `cpp2.sub` / `cpp2.mul` / `cpp2.div` | 1.0 |

### 4.4 Corpus Infrastructure

**Reference Generation** (using cppfront):
```bash
cppfront input.cpp2 -o corpus/reference/input.cpp
clang++ -std=c++20 -Xclang -ast-dump -fsyntax-only \
  corpus/reference/input.cpp > corpus/reference_ast/input.ast.txt
```

**Isomorph Extraction**:
```bash
./tools/extract_ast_isomorphs.py \
  --ast corpus/reference_ast/input.ast.txt \
  --output corpus/isomorphs/input.isomorph.json
```

**MLIR Tagging**:
```bash
./tools/tag_mlir_regions.py \
  --isomorphs corpus/isomorphs/input.isomorph.json \
  --dialect include/Cpp2Dialect.td \
  --output corpus/tagged/input.tagged.json
```

### 4.5 Corpus Status

| Metric | Count |
|--------|-------|
| Total cpp2 files | 189 |
| Successfully transpiled | 158 (83.6%) |
| Total isomorphs extracted | 1,414,202 |
| Unique patterns | 13,545 |
| MLIR region coverage | 100% of dialect ops |

---

## 5. Semantic Loss Scoring

### 5.1 Definition

**Semantic loss** measures divergence between cppfort and cppfront transpiler outputs:

$$
\text{Loss} = \frac{\sum_{i=1}^{n} \text{distance}(\text{AST}_{\text{cppfort}}^i, \text{AST}_{\text{cppfront}}^i)}{n}
$$

### 5.2 Loss Components

| Component | Weight | Metric |
|-----------|--------|--------|
| Structural distance | 50% | Normalized graph edit distance |
| Type distance | 30% | Type inference mismatches |
| Operation distance | 20% | MLIR operation differences |

### 5.3 Loss Categories

| Category | Range | Interpretation |
|----------|-------|----------------|
| Zero loss | 0.0 | Identical AST structure and semantics |
| Low loss | 0.0-0.1 | Minor syntactic differences (whitespace, const placement) |
| Medium loss | 0.1-0.5 | Different code generation strategies (same semantics) |
| High loss | >0.5 | Semantic differences (incorrect translation) |

### 5.4 Scoring Process

1. **Transpile with cppfort**:
   ```bash
   ./build/src/cppfort input.cpp2 output.cpp
   ```

2. **Generate candidate AST**:
   ```bash
   clang++ -std=c++20 -Xclang -ast-dump -fsyntax-only output.cpp > candidate.ast.txt
   ```

3. **Score against reference**:
   ```bash
   ./tools/score_semantic_loss.py \
     --reference corpus/tagged/input.tagged.json \
     --candidate corpus/candidate/input.isomorph.json \
     --output corpus/scores/input.loss.json
   ```

### 5.5 Target Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Average corpus loss | <0.15 | Pending (cppfort compilation blocked) |
| Zero-loss files | >40% | Pending |
| High-loss files | <5% | Pending |

### 5.6 Regression Testing

The regression testing system provides:
- **SHA256 validation**: Test file integrity verification
- **Isomorphic comparison**: Semantic equivalence (not textual diff)
- **Corpus generation**: Reference and candidate outputs
- **15-second timeout**: Per-test timeout enforcement

---

## Appendix A: Metafunction Reference

| Metafunction | Effect |
|--------------|--------|
| `@value` | Value semantics (copy/move/equality) |
| `@ordered` | Three-way comparison (`operator<=>`) |
| `@weakly_ordered` | Weak ordering (`std::weak_ordering`) |
| `@partially_ordered` | Partial ordering (`std::partial_ordering`) |
| `@interface` | Pure interface (virtual destructor, deleted copy/move) |
| `@polymorphic_base` | Virtual destructor for polymorphic bases |
| `@copyable` | Explicit copy operations |
| `@movable` | Explicit move operations |
| `@hashable` | `std::hash` specialization |
| `@print` | `to_string()` method for debugging |
| `@enum` | C++ enum class |
| `@flag_enum` | Enum class with bitwise operators |
| `@union` | Converts struct to union |
| `@regex` | Transforms regex members to `std::regex` |
| `@autodiff` | Automatic differentiation with derivatives |
| `@sample_traverser` | Visitor pattern for member traversal |
| `@struct` | No-op marker (struct is default) |

---

## Appendix B: Parameter Passing Semantics

| Mode | C++ Equivalent | Semantics |
|------|----------------|-----------|
| `in` | `T const&` | Read-only (default) |
| `copy` | `T` | By value |
| `inout` | `T&` | Read-write reference |
| `out` | `T&` (uninitialized) | Must be assigned before use |
| `move` | `T&&` | Takes ownership |
| `forward` | `T&&` (forwarding ref) | Perfect forwarding |
| `in_ref` | `T const&` (explicit) | Reference In |
| `forward_ref` | `T&&` (explicit) | Explicit forward |

---

## Appendix C: Built-in Type Aliases

| Cpp2 Type | C++ Equivalent |
|-----------|----------------|
| `i8` | `std::int8_t` |
| `i16` | `std::int16_t` |
| `i32` | `std::int32_t` |
| `i64` | `std::int64_t` |
| `u8` | `std::uint8_t` |
| `u16` | `std::uint16_t` |
| `u32` | `std::uint32_t` |
| `u64` | `std::uint64_t` |
| `f32` | `float` |
| `f64` | `double` |

---

**End of Document**
