# Cpp2 Surface Grammar

This file is the repo-owned human-readable grammar reference for cppfort.

It is not a generated artifact and not a build input. Executable acceptance lives in `src/selfhost/cpp2.cpp2` (bitmap scanner + reify pipeline).

The executable mirror of the confirmed surface lives in `src/selfhost/cpp2.cpp2`, which implements the encode → decode → index → reify pipeline following the pattern from TrikeShed's JsonBitmap/JsonParser.

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
                        | series_literal
                        | join_expr
                        | transition_expr
                        | alpha_expr
                        | indexed_expr
                        | fold_expr
                        | grad_diff
                        | slice_expr
                        | purity_contract
                        | namespace_decl
                        | lowered_method
                        | chart_project_expr
                        | atlas_locate_expr
                        | struct_annotation
                        | function_declaration
                        | type_declaration
                        | precondition
                        | postcondition

bootstrap_tag_decl    ::= identifier ":" identifier "=" integer ";"

namespace_decl        ::= identifier ":" "namespace" "=" "{" { top_level } "}"

struct_annotation     ::= "@" "struct" type_parameters "type" "=" "{" "}"

function_declaration  ::= identifier ":" "(" [parameter_list] ")" ["->" type] "=" (body | "{" body "}")

type_declaration      ::= identifier ":" "type" "=" ("{" type_body "}" | type_expression)

precondition          ::= "pre" "(" expression ")"
postcondition         ::= "post" "(" expression ")"

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

series_literal        ::= "_s" "[" series_element { "," series_element } "]"
series_element        ::= integer
                        | identifier

transition_expr       ::= identifier "." "transition"
                          "(" string "," string "," coords_literal ")"

join_expr             ::= identifier "j" identifier

indexed_expr          ::= identifier "j" "(" identifier ":" type ")" "=>" expression

fold_expr             ::= expression "." "fold" "(" expression "," expression ")"

alpha_expr            ::= identifier "α" "(" identifier ")" "=>" expression

grad_diff             ::= "grad" "(" expression "," identifier ")"

purity_contract       ::= "[[" ("pure" | "contiguous" | "non_aliasing" | "strided") "]]"

lowered_method        ::= identifier "." "lowered" "(" ")"

chart_project_expr    ::= identifier "." "project" "(" identifier ")"

atlas_locate_expr    ::= identifier "." "locate" "(" identifier ")"

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
(* slice_expr is now CONFIRMED - see top_level *)
```

## Projected Surface

These productions remain part of the larger cppfort design surface and stay here as a grammar target, not as a claim of current acceptance.

```ebnf
type_parameters       ::= "<" type_param { "," type_param } ">"
type_param            ::= identifier
                        | identifier ":" type_constraint
type_constraint       ::= type
                        | type_constraint "requires" expression

cursor_type           ::= "series" "<" "series" ">"

elementwise_mul       ::= expression "**" expression
elementwise_add       ::= expression "++" expression
indexed_view          ::= expression "*[" expression "]"
dense_view            ::= "dense" "(" expression ")"

rank_annotation       ::= "<" "rank" "=" integer ">"

type                  ::= qualified_name
                        | type "<" type_arguments ">"
                        | "(" parameter_list ")" "->" type
```

## Reference Boundary

- If this file and the executable selfhost parser disagree, fix one immediately and keep them aligned.
- `expanded_cpp2_spec.md` remains design lineage and rationale.
- This file is the shortest direct answer to "what is the cpp2 surface grammar here?"
