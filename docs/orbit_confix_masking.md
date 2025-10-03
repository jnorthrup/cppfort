# Orbit Confix Signal Masking

[`docs/orbit_confix_masking.md`](docs/orbit_confix_masking.md:1)

## Core principle

Each delimiter (confix) — braces `{}`, parentheses `()`, angle brackets `<>`, brackets `[]`, and quotes `""` — defines a grammatical region. Patterns are only considered valid when they are allowed by the current confix region and (optionally) by the nesting depth. Masking is implemented as a compact bitmask on each pattern so the scanner can quickly filter out false positives that are simply substring matches but not grammatical constructs.

## Masking matrix (summary)

The matrix below is a quick reference of common pattern visibility. A checked box (✅) indicates the pattern is allowed in that confix; an empty box (❌) means it should be ignored when scanning inside that confix.

| Pattern Type | Depth 0 | Inside `{}` | Inside `()` | Inside `<>` | Inside `[]` | Inside `""` |
|--------------|---------|-------------|-------------|-------------|-------------|-------------|
| Function decl `:(` | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Operator decl `operator=:` | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Param modes `out`, `inout` | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Type syntax `: i32` | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| Cast operator ` as ` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Inspect expr `inspect ` | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Template args | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |

Keep this matrix up-to-date as new patterns or confixes are added.

## Implementation

This section provides concrete C++ types and matching logic you can integrate into the orbit scanner.

### Confix mask enum and OrbitPattern

```cpp
// docs/orbit_confix_masking.md:implementation
#include <cstdint>
#include <string>

struct ConfixContext {
    // Which confix we are currently inside (single deepest active confix).
    // If multiple nested confixes exist, the scanner typically reports the deepest one.
    enum class Kind : uint8_t {
        TopLevel = 0,
        Brace,   // {}
        Paren,   // ()
        Angle,   // <>
        Bracket, // []
        Quote    // ""
    } kind = Kind::TopLevel;

    // Current nesting depth (0 == top-level)
    int depth = 0;
};

struct OrbitPattern {
    std::string signature;   // the literal signature to match ("inspect ", "operator=:", ":(", " as ", etc.)
    int expected_depth = -1; // -1 = don't care; otherwise exact depth required
    uint8_t confix_mask = 0x3F; // default visible everywhere (6 bits)

    enum ConfixMask : uint8_t {
        TopLevel  = 1 << 0, // depth 0
        InBrace   = 1 << 1, // inside {}
        InParen   = 1 << 2, // inside ()
        InAngle   = 1 << 3, // inside <>
        InBracket = 1 << 4, // inside []
        InQuote   = 1 << 5, // inside ""
    };

    // Utility: check if this pattern is allowed in the given confix context
    bool allows_confix(const ConfixContext &ctx) const noexcept {
        uint8_t bit = 0;
        switch (ctx.kind) {
        case ConfixContext::Kind::TopLevel: bit = TopLevel; break;
        case ConfixContext::Kind::Brace:   bit = InBrace;   break;
        case ConfixContext::Kind::Paren:   bit = InParen;   break;
        case ConfixContext::Kind::Angle:   bit = InAngle;   break;
        case ConfixContext::Kind::Bracket: bit = InBracket; break;
        case ConfixContext::Kind::Quote:   bit = InQuote;   break;
        }
        return (confix_mask & bit) != 0;
    }

    // Truth-based match: check signature, confix, and depth
    bool matches_at(const char *text, size_t pos, const ConfixContext &ctx) const noexcept {
        // 1) signature match
        for (size_t i = 0; i < signature.size(); ++i) {
            if (text[pos + i] != signature[i]) return false;
        }
        // 2) confix allowed?
        if (!allows_confix(ctx)) return false;
        // 3) depth check (if requested)
        if (expected_depth >= 0 && expected_depth != ctx.depth) return false;
        return true;
    }
};
```

Notes:
- confix_mask defaults to 0x3F (bits 0..5 set) meaning visible in all six regions. Adjust per pattern to implement the matrix.
- expected_depth is optional; use it only for patterns that only occur at a specific nesting (e.g., top-level function declarations).

### Example pattern table (C++ initializers)

```cpp
// Example patterns following the matrix above
OrbitPattern fn_decl { ":( ",      -1, uint8_t(OrbitPattern::TopLevel | OrbitPattern::InBrace) };
OrbitPattern op_decl { "operator=:", -1, uint8_t(OrbitPattern::InBrace) };
OrbitPattern param_out{ "out ",     -1, uint8_t(OrbitPattern::InParen) };
OrbitPattern type_colon{ ": ",      -1, uint8_t(
    OrbitPattern::TopLevel | OrbitPattern::InBrace | OrbitPattern::InParen | OrbitPattern::InAngle) };
OrbitPattern cast_as { " as ",     -1, uint8_t(
    OrbitPattern::TopLevel | OrbitPattern::InBrace | OrbitPattern::InParen | OrbitPattern::InAngle | OrbitPattern::InBracket) };
OrbitPattern inspect_ { "inspect ", -1, uint8_t(OrbitPattern::TopLevel | OrbitPattern::InBrace) };
OrbitPattern template_arg { "<",     -1, uint8_t(OrbitPattern::InAngle) }; // template args handled at angle-level
```

### Scanner integration (pseudocode)

The orbit scanner should maintain a lightweight ConfixContext while iterating source text. At each character position:

```text
PSEUDOCODE:

context = { kind: TopLevel, depth: 0 }
for pos in 0 .. text.length-1:
    // update context if char is an open/close confix
    update_confix_context(text[pos], context)

    // for each pattern candidate whose first char == text[pos]
    for pattern in patterns_starting_with(text[pos]):
        if pattern.matches_at(text, pos, context):
            report_match(pattern, pos, context)
```

Where update_confix_context processes tokens:
- On `{` -> push Brace, depth++
- On `}` -> pop to previous confix, depth--
- On `(` / `)` similar for Paren
- On `<` / `>` similar for Angle (careful: template `<` vs comparison)
- On `[` / `]` similar for Bracket
- On `"` toggle Quote (quotes often require escape-handling)

Important:
- The scanner should resolve ambiguous `<` (less-than) vs template start using heuristics (e.g., following identifier + `<` usually template).
- Quote regions should treat escapes (`\"`) so the scanner doesn't exit prematurely.

### Truth-based filtering benefits

- Eliminates false positives: e.g., `"inspect "` appearing in string literals won't be reported if InQuote is masked off.
- Performance: confix mask test is a single bitwise operation — cheap compared to regexes.
- Maintainability: adding a new pattern only requires setting its confix_mask and (optionally) expected_depth.

## Examples and test cases

### Minimal unit test (conceptual)
```cpp
// Minimal conceptual asserts - integrate into your test framework
const char *src = R"(
foo:( x: i32 ) { inspect x; }
)";

ConfixContext ctx_top{ ConfixContext::Kind::TopLevel, 0 };
assert(!inspect_.matches_at(src, 0, ctx_top)); // "f" != "inspect "

// find 'inspect ' position and ensure it's allowed InBrace
size_t pos_inspect = /* index of "inspect " inside braces */;
ConfixContext ctx_brace{ ConfixContext::Kind::Brace, 1 };
assert(inspect_.matches_at(src, pos_inspect, ctx_brace));
```

### Integration test ideas
- Strings: ensure patterns masked inside quotes are ignored.
- Templates: ensure `<` inside templates allows template-specific matches but hides others.
- Nested confixes: check depth-aware patterns (expected_depth usage).

## Performance considerations

- Build a prefix map (hash table or trie) of patterns keyed by their first character to avoid scanning every pattern at every position.
- Avoid repeatedly creating ConfixContext objects; update an existing context while iterating characters.
- Keep conformance checks (mask & bit and depth compare) inline and branch-predictable.

## Extending the system

- Support composite masks (allow sets of confix kinds) easily by OR'ing ConfixMask bits.
- Allow negative masks (deny-list) if desired, though positive allow-lists are simpler.
- Add an optional pattern priority so overlapping patterns resolve predictably.

## Summary

The confix masking system turns simple string matching into grammar-aware signal detection by combining:
1. Lightweight confix context tracking,
2. A per-pattern bitmask that encodes allowed grammatical regions,
3. Optional depth constraints.

Use [`docs/orbit_confix_masking.md`](docs/orbit_confix_masking.md:1) as the canonical reference when adding or tuning patterns in the orbit scanner.
