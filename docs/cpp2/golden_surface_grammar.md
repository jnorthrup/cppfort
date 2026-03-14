# Cpp2 Surface Grammar

This file is the repo-owned human-readable grammar reference for cppfort.

It is not a generated artifact and not a build input. Executable acceptance still lives in `src/selfhost/rbcursive.cpp2` and `tests/selfhost_rbcursive_smoke.cpp`.

The executable mirror of the confirmed surface lives directly in `src/selfhost/rbcursive.cpp2`, which is the current JSON feature-stream scanner nucleus.

## Status Labels

- `CONFIRMED`: accepted by the current selfhost smoke path
- `NEXT`: the next bounded surface planned for the selfhost path
- `PROJECTED`: retained as design surface, not yet accepted by the selfhost path

## Confirmed Surface

These productions describe the surface already exercised by the selfhost scanner smoke tests.

```ebnf
translation_unit      ::= { top_level }

top_level             ::= bootstrap_tag_decl
                        | chart_definition
                        | manifold_decl
                        | atlas_literal
                        | coords_literal
                        | join_expr
                        | transition_expr

bootstrap_tag_decl    ::= identifier ":" identifier "=" integer ";"

chart_definition      ::= chart_decl chart_body
chart_decl            ::= "chart" identifier "(" parameter_text ")"
parameter_text        ::= ? balanced parameter source preserved structurally ?

chart_body            ::= "{" { chart_clause } "}"
chart_clause          ::= contains_clause
                        | project_clause
                        | embed_clause

contains_clause       ::= "contains" identifier relation scalar
project_clause        ::= "project" "->" coords_literal
embed_clause          ::= "embed" "(" identifier ")" "->" embed_expr

embed_expr            ::= transition_expr
                        | join_expr
                        | line_expr

atlas_literal         ::= "atlas" "[" identifier { "," identifier } "]"
manifold_decl         ::= "manifold" identifier "=" atlas_literal

coords_literal        ::= "coords" "[" coords_element { "," coords_element } "]"
coords_element        ::= scalar
                        | line_expr

transition_expr       ::= identifier "." "transition"
                          "(" string "," string "," coords_literal ")"

join_expr             ::= identifier "j" identifier

line_expr             ::= term { operator term }
term                  ::= identifier
                        | scalar
                        | local_index

local_index           ::= "local" "[" integer "]"

relation              ::= "<" | "<=" | ">" | ">=" | "==" | "!="
operator              ::= "+" | "-" | "*" | "/"
scalar                ::= integer | floating
```

## Confirmed Constraints

- `chart_decl` currently preserves the parameter list as balanced source text. The selfhost path records structure here before deeper semantic parsing.
- `transition_expr` currently requires string chart names and a `coords[...]` literal as the third argument.
- `join_expr` is currently the compact `a j b` surface, not a generalized infix precedence family.
- `line_expr` is intentionally shallow. The current selfhost path accepts linear arithmetic tails needed by the smoke corpus; it is not yet a full expression grammar.
- `src/selfhost/rbcursive.cpp2` is the executable mirror for the confirmed subset.

## Next Surface

This is the next bounded grammar slice already called out in conductor truth.

```ebnf
alpha_expr            ::= expression "α" "(" identifier ")" "=>" expression
```

The intended source form is:

```text
series α (x) => expr
```

## Projected Surface

These productions remain part of the larger cppfort design surface and stay here as a grammar target, not as a claim of current acceptance.

```ebnf
indexed_expr          ::= expression "j" "(" identifier ":" type ")" "=>" expression
series_literal        ::= "_s" "[" expression { "," expression } "]"
fold_expr             ::= expression ".fold(" expression "," expression ")"
slice_expr            ::= expression "[" expression ".." expression "]"
cursor_type           ::= "series" "<" "series" ">"

elementwise_mul       ::= expression "**" expression
elementwise_add       ::= expression "++" expression
indexed_view          ::= expression "*[" expression "]"
grad_diff             ::= "grad" "(" expression "," identifier ")"
dense_view            ::= "dense" "(" expression ")"

rank_annotation       ::= "<" "rank" "=" integer ">"
axis_annotation       ::= "<" "axis" "=" identifier { "," identifier } ">"
purity_contract       ::= "[[" ("pure" | "contiguous" | "non_aliasing" | "strided") "]]"
```

## Reference Boundary

- If this file and the executable selfhost parser disagree, fix one immediately and keep them aligned.
- `expanded_cpp2_spec.md` remains design lineage and rationale.
- This file is the shortest direct answer to "what is the cpp2 surface grammar here?"
