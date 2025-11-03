# Complete Cpp2 to C++ Semantic Mappings

This document defines the complete semantic mapping system for the Cpp2 transpiler, covering all language constructs from the ground up.

## Core Language Constructs

### 1. Function Definitions

**Cpp2 Syntax:**

```
name: (params) -> return_type = { body }
```

**Semantic Mapping:**

```
return_type name(param_transform) { body_transform }
```

**Parameter Transformations:**

- `in identifier: Type` → `Type identifier` (default)
- `copy identifier: Type` → `Type identifier`
- `inout identifier: Type` → `Type& identifier`
- `move identifier: Type` → `Type&& identifier`
- `forward identifier: Type` → `Type&& identifier` (with std::forward in usage)
- `out identifier: Type` → `Type& identifier` (initialized to default value)

**Complete Examples:**

```
// Input
main: () -> int = { return 0; }

// Output
int main() { return 0; }

// Input
add: (a: int, b: int) -> int = { return a + b; }

// Output
int add(int a, int b) { return a + b; }

// Input
process: (inout str: std::string) = { str += "processed"; }

// Output
void process(std::string& str) { str += "processed"; }
```

### 2. Variable Declarations

**Cpp2 Syntax:**

```
identifier: Type = initializer;
identifier := initializer;  // auto deduction
```

**Semantic Mapping:**

```
Type identifier = initializer;  // for explicit type
auto identifier = initializer;  // for auto deduction
```

**Complete Examples:**

```
// Input
x: int = 5;

// Output
int x = 5;

// Input
y := 10;

// Output
auto y = 10;

// Input
s: std::string = "hello";

// Output
std::string s = "hello";
```

### 3. Type System

**Cpp2 Syntax** → **C++ Equivalent:**

- `i8` → `std::int8_t`
- `i16` → `std::int16_t`
- `i32` → `std::int32_t`
- `i64` → `std::int64_t`
- `u8` → `std::uint8_t`
- `u16` → `std::uint16_t`
- `u32` → `std::uint32_t`
- `u64` → `std::uint64_t`
- `f32` → `float`
- `f64` → `double`
- `_` → `auto` (in function contexts)
- `type` → class/struct definition

### 4. Type Definitions

**Cpp2 Syntax:**

```
TypeName: type = {
  member_declarations...
}
```

**Semantic Mapping:**

```
struct TypeName {
  member_declarations...
};
```

**Complete Examples:**

```
// Input
Point: type = {
    x: i32 = 0;
    y: i32 = 0;
}

// Output
struct Point {
    std::int32_t x = 0;
    std::int32_t y = 0;
};
```

### 5. Contract Expressions

**Cpp2 Syntax:**

```
_pre: (condition) = "message";
_post: (return_value) = "message" requires condition;
```

**Semantic Mapping:**

```
// Using assertions or contract library
assert(condition && "message");  // for preconditions
// Postconditions typically require return value manipulation
```

### 6. Inspect Expressions

**Cpp2 Syntax:**

```
inspect value -> ReturnType {
    is pattern1 => expression1;
    is pattern2 => expression2;
    is _ => default_expression;
}
```

**Semantic Mapping:**

```
// Transform to if/else chain or std::visit for variants
if (std::holds_alternative<Pattern1Type>(value)) {
    auto&& temp = std::get<Pattern1Type>(value);
    return expression1;
} else if (std::holds_alternative<Pattern2Type>(value)) {
    auto&& temp = std::get<Pattern2Type>(value);
    return expression2;
} else {
    return default_expression;
}
```

### 7. Pattern Matching

**Cpp2 Syntax** → **C++ Equivalent:**

- `is Type` → `std::holds_alternative<Type>(variant)`
- `is identifier: Type` → `auto&& identifier = std::get<Type>(variant)`
- `as Type` → `std::get<Type>()` or `std::get<>()` for variants, `static_cast<Type>()` for other types

### 8. Loop Constructs

**Cpp2 Syntax** → **C++ Equivalent:**

- `for identifier : container` → `for (auto&& identifier : container)`
- `while condition { body }` → `while (condition) { body; }`
- `loop { body }` → `while (true) { body; }`

### 9. Range/Iterator Operations

**Cpp2 Syntax:**

```
start..end  // inclusive range
start..=end // inclusive range
start..<end // exclusive range
```

**Semantic Mapping:**

```
// Using ranges library or manual iteration
// Implementation-dependent
```

### 10. UFCS (Uniform Function Call Syntax)

**Cpp2 Syntax:**

```
obj.function(args)
```

**Semantic Mapping:**

```
function(obj, args)  // if function is defined as taking obj as first parameter
// or
obj.function(args)   // if function is a true member
```

### 11. Template Definitions

**Cpp2 Syntax:**

```
Name: <T> type = { ... }
```

**Semantic Mapping:**

```
template<typename T>
struct Name { ... };
```

### 12. Generic Functions

**Cpp2 Syntax:**

```
func: <T> (param: T) -> RetType = { ... }
```

**Semantic Mapping:**

```
template<typename T>
RetType func(T param) { ... }
```

## Pattern Definition Schema

For each semantic mapping, define a pattern with:

- Input pattern (with anchors)
- Output template
- Transformation rules
- Validation constraints

### Pattern Definition Format

```yaml
name: cpp2_to_cpp_function
use_alternating: true
alternating_anchors:
  - ":"
  - "="
grammar_modes: 7
evidence_types:
  - "function_identifier"
  - "function_signature"
  - "function_body"
priority: 250
transformation_templates:
  2: "{function_signature} {function_identifier} {function_body}"
```

## Include Generation System

The system should automatically detect required headers:

- `std::string` → `#include <string>`
- `std::vector` → `#include <vector>`
- `std::variant` → `#include <variant>`
- `std::optional` → `#include <optional>`
- `std::span` → `#include <span>`
- etc.

## Type Safety Transformations

### Safe Casts

- `as Type` → `static_cast<Type>()` or `std::get<Type>()` depending on context

### Bounds Checking

- Add automatic bounds checking where appropriate
- Use `at()` instead of `[]` for containers when safety is prioritized

## Complete Mapping Implementation Plan

1. **Lexer/Parser Mappings**: Define how Cpp2 tokens map to C++ constructs
2. **AST Transformation**: Define how Cpp2 AST nodes map to C++ AST nodes
3. **Type System Mapping**: Define how Cpp2 types map to C++ types
4. **Pattern Matching System**: Define how Cpp2 patterns map to C++ patterns
5. **Memory Management**: Define how Cpp2 references map to C++ references/value categories
6. **Safety Features**: Define how Cpp2 contracts and safety features map to C++ equivalents
7. **Code Generation**: Define how to generate valid C++ code from mapped AST

This comprehensive mapping system ensures all language features are handled systematically from the beginning, allowing corrections to be properly correlated and propagated throughout the system.
