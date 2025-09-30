# Coding Standards - cppfort

## Header Guards

**MANDATORY:** All header files (`.h`, `.hpp`) MUST use `#pragma once` for include guards.

### Rationale
- Simpler, less error-prone than traditional sentinel defines
- Supported by all modern compilers (GCC, Clang, MSVC, ICC)
- No risk of macro name collisions
- Cleaner, more maintainable code

### Standard
```cpp
// CORRECT: Use #pragma once
#pragma once

#include <vector>

namespace cppfort {
    // ... declarations
}
```

```cpp
// INCORRECT: Do not use sentinel defines
#ifndef CPPFORT_MYHEADER_H  // ❌ DO NOT USE
#define CPPFORT_MYHEADER_H  // ❌ DO NOT USE

namespace cppfort {
    // ... declarations
}

#endif // CPPFORT_MYHEADER_H  // ❌ DO NOT USE
```

### Migration
When encountering legacy sentinel-style header guards:
1. Replace the entire guard block with a single `#pragma once` at the top
2. Remove the trailing `#endif` comment
3. Verify compilation succeeds

### Enforcement
- All new header files must use `#pragma once`
- Existing headers should be migrated opportunistically during edits
- Code reviews must reject PRs introducing sentinel-style guards

## Namespace Conventions

All cppfort code resides in the `cppfort` namespace with the following sub-namespaces:

```cpp
cppfort::ir         // Intermediate representation (Sea of Nodes)
cppfort::parser     // CPP2 parsing
cppfort::emitter    // Code generation
cppfort::stage0     // Meta-transpiler stage 0
cppfront::stage1    // Meta-transpiler stage 1
```

## Include Order

Headers should be included in the following order, with blank lines between groups:

1. Corresponding header (for `.cpp` files)
2. C system headers (`<cstdlib>`, `<cassert>`)
3. C++ standard library (`<vector>`, `<string>`, `<memory>`)
4. Third-party library headers
5. Project headers (relative or with `"..."` notation)

Example:
```cpp
#include "node.h"

#include <cassert>
#include <cstdlib>

#include <memory>
#include <string>
#include <vector>

#include "type.h"
#include "parser.h"
```

## Naming Conventions

- **Classes/Types:** PascalCase (`StartNode`, `TypeInteger`)
- **Functions/Methods:** camelCase (`peephole()`, `computeType()`)
- **Variables:** camelCase (`myVar`, `nodeCount`)
- **Private members:** Leading underscore (`_inputs`, `_nid`)
- **Constants:** UPPER_SNAKE_CASE or camelCase depending on scope
- **Namespaces:** lowercase (`cppfort::ir`)

## Formatting

- **Indentation:** 4 spaces (no tabs)
- **Braces:** Opening brace on same line for functions/methods, new line for class definitions
- **Line length:** Soft limit of 100 characters, hard limit of 120
- **Pointer/Reference alignment:** `Type* ptr` and `Type& ref` (align with type)

## Documentation

- Document public APIs with clear comments
- Explain non-obvious algorithms with inline comments
- Reference Simple compiler chapter numbers for Sea of Nodes implementations
- Use `TODO:` markers for incomplete implementations with brief explanation

## Modern C++ Usage

- Prefer `nullptr` over `NULL` or `0`
- Use `auto` for complex iterator types and obvious initializations
- Prefer range-based for loops when appropriate
- Use smart pointers (`std::unique_ptr`, `std::shared_ptr`) for ownership semantics
- Manual memory management is acceptable for performance-critical graph structures (Node system)

## Compiler Compatibility

Target: C++17 minimum
- GCC 7+
- Clang 5+
- MSVC 2017+

Avoid C++20/23 features until broader adoption.
