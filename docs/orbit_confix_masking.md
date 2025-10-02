# Orbit Confix Signal Masking

## Core Principle
Each delimiter type creates a confix that selectively masks/unmasks pattern signals based on grammatical context.

## Masking Matrix

| Pattern Type | Depth 0 | Inside `{}` | Inside `()` | Inside `<>` | Inside `[]` | Inside `""` |
|--------------|---------|-------------|-------------|-------------|-------------|-------------|
| Function decl `:(` | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Operator decl `operator=:` | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Param modes `out`, `inout` | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Type syntax `: i32` | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| Cast operator ` as ` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Inspect expr `inspect ` | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Template args | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |

## Implementation

Each `OrbitPattern` needs:
```cpp
struct OrbitPattern {
    // Existing fields...

    // Confix visibility mask - bitfield of which confixes allow this pattern
    enum ConfixMask : uint8_t {
        TopLevel  = 1 << 0,  // depth 0
        InBrace   = 1 << 1,  // inside {}
        InParen   = 1 << 2,  // inside ()
        InAngle   = 1 << 3,  // inside <>
        InBracket = 1 << 4,  // inside []
        InQuote   = 1 << 5,  // inside ""
    };
    uint8_t confix_mask = 0xFF;  // Default: visible everywhere
};
```

## Truth-Based Matching

Pattern matches are only valid when:
1. Signature string matches at position
2. Current confix context is in pattern's visibility mask
3. Depth level matches expected depth (if specified)

This ensures orbit scanner reports **actual grammatical matches**, not accidental string occurrence.
