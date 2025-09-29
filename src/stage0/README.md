# Stage0 Transpiler Architecture

Stage0 provides the initial cpp2→C++ transpiler built directly from the
reference material in `docs/**/*.md`. The implementation is intentionally
small and explicit so that later stages can grow it into a production
compiler while preserving transparent provenance.

## Pipeline overview

1. **Documentation corpus** – scans the repository documentation for
   fenced `cpp`/`cpp2` snippets and exposes them as structured examples.
   Stage0 uses these examples as smoke tests and to seed future feature
   induction steps.
2. **Lexer** – converts cpp2 source text into a token stream. The lexer
   currently understands identifiers, literals, punctuation required for
   declarations, and a minimal keyword set (`return`).
3. **Parser** – builds a lightweight AST featuring translation units,
   function declarations, parameter lists, blocks, variable declarations,
   and raw expression statements. Expression contents are preserved as
   source slices to keep the parser simple while still enabling later
   refinement.
4. **Emitter** – lowers the AST to idiomatic C++20 syntax using the
   `auto f(...) -> type` convention. Variable declarations are translated
   into `type name = expr;` statements, with expression and return bodies
   emitted verbatim.

Each stage is implemented as a dedicated component so that Stage1 and
Stage2 can independently extend the lexer, parser, and emitter. The CLI
(`src/stage0/main.cpp`) wires the pieces together and offers:

- `transpile <input.cpp2> <output.cpp>` – lower a single source file
  into C++.
- `scan-docs` – enumerate documentation snippets and attempt to lex/parse
  the ones compatible with the current feature set. This keeps Stage0
  grounded in the written specification while providing a harness for
  guided expansion ("three-way induction": docs ↔ parser ↔ emitter).

The code is written in portable C++20 using only the standard library.
