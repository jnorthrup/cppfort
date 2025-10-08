# Alternating Anchor/Evidence Pattern System

## Overview

The alternating pattern system enables deterministic pattern matching by alternating between fixed **anchors** (literal strings) and flexible **evidence** (typed spans). This approach supports bidirectional transformations between different grammar representations (e.g., CPP2 ↔ CPP).

## Pattern Structure

An alternating pattern consists of:

1. **Anchors**: Fixed literal strings that must appear in sequence
2. **Evidence Types**: Typed spans between/around anchors (e.g., `identifier`, `type_expression`)
3. **Transformation Templates**: Target grammar templates with `$1`, `$2`, ... placeholders

### Example Pattern (cpp2_template_type_alias)

```yaml
name: cpp2_template_type_alias
use_alternating: true
alternating_anchors:
  - "type"
  - "="
grammar_modes: 4
evidence_types:
  - "identifier_template"  # Type name with optional template args
  - "type_expression"      # The aliased type
transformation_templates:
  2: "using $1 = $2;"      # CPP output
  1: "typedef $2 $1;"      # C output
```

### Matching Process

For input: `type Pair<A,B>=std::pair<A,B>;`

1. **Find first anchor**: `"type"` at position 0
2. **Extract evidence[0]**: Between `"type"` and `"="` → `"Pair<A,B>"`
3. **Validate evidence type**: `identifier_template` ✓
4. **Extract evidence[1]**: After `"="` to end → `"std::pair<A,B>"`
5. **Validate evidence type**: `type_expression` ✓
6. **Apply template**: `"using $1 = $2;"` → `"using Pair<A,B> = std::pair<A,B>;"`

## Special Cases

### Single Anchor with Evidence Before and After

For patterns like `x := 42`, where evidence appears both before and after the anchor:

```yaml
name: cpp2_variable
use_alternating: true
alternating_anchors:
  - ":="
evidence_types:
  - "identifier"       # Variable name (before :=)
  - "expression"       # Initializer value (after :=)
transformation_templates:
  2: "auto $1 = $2"    # CPP output
```

The implementation detects this case (1 anchor, 2 evidence types) and extracts:
- Evidence[0]: Everything before anchor
- Evidence[1]: Everything after anchor

## Implementation Files

- **Pattern Loader** ([pattern_loader.cpp](../src/stage0/pattern_loader.cpp)): Parses YAML patterns
- **Speculation** ([rbcursive.cpp](../src/stage0/rbcursive.cpp)): `speculate_alternating()` validates anchors and evidence
- **Extraction** ([cpp2_emitter.cpp](../src/stage0/cpp2_emitter.cpp)): `extract_alternating_segments()` extracts evidence spans
- **Substitution** ([cpp2_emitter.cpp](../src/stage0/cpp2_emitter.cpp)): `apply_substitution()` applies transformation templates

## Supported Transformations

| CPP2 Input | CPP Output |
|------------|------------|
| `type Pair<A,B>=std::pair<A,B>;` | `using Pair<A,B> = std::pair<A,B>;` |
| `type Map<K,V>=std::map<K,V>;` | `using Map<K,V> = std::map<K,V>;` |
| `x := 42` | `auto x = 42` |
| `Integer: type = int;` | `using Integer = int;` |

## Evidence Type Validation

Evidence types are validated using simple heuristics (to be enhanced):

- `identifier`: Basic identifier pattern (`[a-zA-Z_][a-zA-Z0-9_]*`)
- `identifier_template`: Identifier with optional template args (contains `<` or `>`)
- `type_expression`: Any type expression
- `expression`: Any expression

## Future Enhancements

1. **Stricter Evidence Validation**: Use full parser for evidence type checking
2. **Nested Pattern Recursion**: Allow patterns within evidence spans
3. **Multi-Way Transformations**: Support more than 2 target grammars
4. **Pattern Composition**: Combine patterns for complex constructs
