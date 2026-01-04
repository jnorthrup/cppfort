# Specification: Spirit-Like EBNF Combinator Mapping

## 1. Objective

Enable direct, 1-to-1 mapping of EBNF grammar rules to C++ parser combinator code using operator overloading and strict type aliasing. The goal is to make the C++ implementation visually identical to the EBNF definition `grammar/cpp2.ebnf`.

## 2. Requirements

### 2.1 Operator Grammar
Implement a "local namespace operator grammar" to provide Spirit-like syntax without global namespace pollution.

| Concept | EBNF | C++ Operator |
|---------|------|--------------|
| Sequence | `a b` | `a >> b` |
| Alternation | `a \| b` | `a \| b` |
| Zero-or-more | `{ a }` | `*a` |
| One-or-more | `{ a }+` | `+a` |
| Optional | `[ a ]` | `-a` |
| Transform | `a -> f` | `a[f]` |
| Difference | `a - b` | `a - b` |
| List | `a % b` | `a % b` |

### 2.2 Symbolic Typealiases
For every non-terminal in `grammar/cpp2.ebnf`, define a corresponding C++ `constexpr` object or type alias in a dedicated namespace (`cpp2::parser::rules`).

example:
```cpp
namespace cpp2::parser::rules {
    constexpr auto integer_literal = ...;
    constexpr auto identifier = ...;
    
    // Recursive rule definition support
    constexpr auto expression = recursive([](auto self) { 
        return ...; 
    });
}
```

### 2.3 Local Namespace
All operators must be confined to a specific namespace (e.g., `using namespace cpp2::parser::operators;`) or use ADL-enabled types to prevent conflicts with global operators.

### 2.4 Compositional Midpoints
Define "compositional midpoints" (intermediate combinator types) that are exposed as named entities, allowing them to be inspected, tested, and composed independently.

## 3. Reference Grammar

The source of truth is `grammar/cpp2.ebnf`. The C++ implementation must verify against this file.

## 4. Verification

- **Syntax Test**: A compilation test that verifies the C++ syntax matches the expected EBNF-like structure.
- **Behavior Test**: Unit tests ensuring the overloaded operators produce the same parser behavior as the functional `seq`/`alt`/etc. combinators.
