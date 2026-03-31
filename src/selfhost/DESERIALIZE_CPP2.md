# DESERIALIZE_CPP2

Pattern: ~/work/Trikeshed/src/commonMain/kotlin/borg/trikeshed/parse/json/JsonBitmap.kt
        ~/work/Trikeshed/src/commonMain/kotlin/borg/trikeshed/parse/json/Json.kt

Executable: src/selfhost/cpp2.cpp2

## Pipeline

```
encode(src)  → pixel per byte (2-bit structural + 2-bit lexer)
decode(bmp)  → state machine (quotes mask structural, escapes toggle, comments mask)
index(bmp)   → declaration boundaries (depth-tracked, string/comment-aware)
reify(src)   → construct types from bitmap positions
parse(src)   → decode → index → reify
```

This is the same pipeline as TrikeShed's JSON parser. Not invented here.
Ported from working Kotlin to cpp2.

## Bitmap Encoding

Each input byte → 4-bit pixel:
- bits 0-1: struct_event (unchanged, scope_open, scope_close, decl_delim)
- bits 2-3: lexer_event (unchanged, string_delim, escape, comment_start)

Structural chars for cpp2:
- ScopeOpen:  {, (, [
- ScopeClose: }, ), ]
- DeclDelim:  ;, ,
- StringDelim: ", `
- Escape:     \

## State Machine (decode)

- odd quotes mask structural events inside strings
- escapes toggle quote masking
- comments mask everything until newline

Same state machine as JsonBitmap.decode().

## Declaration Boundaries (index)

Walk source tracking depth:
- depth++ on {, (, [
- depth-- on }, ), ]
- at depth 0, ';' or '}' marks declaration boundary
- string/comment contents skipped

Output: vector of decl_boundary { lo, hi, depth }

## Types (reify)

From golden_surface_grammar.md:

```
decl ::= tag_decl | namespace_decl | type_decl | type_alias | func_decl

tag_decl      = name ':' type_name '=' value
namespace_decl = name ':' 'namespace' '=' '{' ... '}'
type_decl     = name ':' ['@' metafunc] 'type' ['==' value] '=' '{' ... '}'
type_alias    = name ':' 'type' '==' value
func_decl     = name ':' '(' params ')' ['->' return_type] '=' '{' ... '}'
```

Reify algorithm:
1. skip whitespace/comments
2. read name (word)
3. expect ':'
4. what follows ':' determines kind:
   - "namespace" → namespace_decl
   - "@word type" → type_decl (with metafunc)
   - "type ==" → type_alias
   - "type" → type_decl (plain)
   - "(" → func_decl
   - word "=" → tag_decl

## Current State

- [ ] Untested
- [ ] Does not compile (haven't verified cpp2 transpilation)
- [x] reify() covers all confirmed surface productions from golden_surface_grammar.md
- [x] Types mirror golden grammar: tag, ns, type, alias, func, chart, manifold, atlas, coords, series, join, transition, alpha, indexed, fold, grad, slice, purity, lowered, project, locate, pre, post
- [x] α detection via UTF-8 byte comparison (0xCE 0xB1)
- [x] namespace bodies recursively reified — parse_ns_body wired into reify, children emitted by generator
- [x] bitmap memoizes depth at every scope open/close, string open/close, comment start/end, decl delimiter
- [x] index_decls uses memoized depths — merge-walk over structural vectors, no rescanning
- [ ] Does not compile (haven't verified cpp2 transpilation)
- [ ] index_decls merge-walk ignores string/comment masking — structural events inside strings/comments are already excluded by decode(), so this is correct as long as decode() runs first

## Test Plan

1. Run parse() on bbcursive.cpp2 itself
2. Verify it finds: namespace_decl("cpp2"), func_decl("lex"), func_decl("reader"), etc.
3. Run parse() on cpp2.cpp2 itself (self-host)
4. Verify it finds its own declarations
5. Compare output to golden_surface_grammar.md coverage

## What This Is Not

- Not a combinator parser (seq/alt/opt/rep)
- Not a procedural call tree
- Not invented here — ported from json.kt
- Not complete — needs testing and expansion

## What This Is

- A bitmap scanner following a proven pattern
- A position-indexed data structure
- A deserializer that constructs types from bitmap positions
- The replacement for rbcursive.cpp2 (7150 lines → 456 lines)
