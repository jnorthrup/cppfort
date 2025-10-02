# Coding Standards - cppfort

## 🚨 TOP PRIORITY CODING STANDARD 🚨

**ALL header files MUST use `#pragma once` as the first line (no exceptions).**

This is the single most important coding standard in cppfort. Header guard violations will be rejected in code review. No traditional sentinel defines (`#ifndef HEADER_H`) are permitted.

---

## Header Guards

**MANDATORY:** All header files (`.h`, `.hpp`) MUST use `#pragma once` for include guards.

- Simpler, less error-prone than traditional sentinel defines
- Supported by all modern compilers (GCC, Clang, MSVC, ICC)
- No risk of macro name collisions, cleaner code

**Standard:** `#pragma once` as first line only (no sentinel defines)

**Migration:** Replace legacy guards with `#pragma once` during edits

**Enforcement:** Code reviews reject sentinel-style guards

## Namespace Conventions

- `cppfort::ir` - Intermediate representation (Sea of Nodes)
- `cppfort::parser` - CPP2 parsing
- `cppfort::emitter` - Code generation
- `cppfort::stage0` - Meta-transpiler stage 0
- `cppfront::stage1` - Meta-transpiler stage 1

## Include Order

Headers in order: corresponding → C system → C++ stdlib → third-party → project (blank lines between groups)

## Naming Conventions

- Classes/Types: PascalCase (`StartNode`, `TypeInteger`)
- Functions/Methods: camelCase (`peephole()`, `computeType()`)
- Variables: camelCase (`myVar`, `nodeCount`)
- Private members: Leading underscore (`_inputs`, `_nid`)
- Constants: UPPER_SNAKE_CASE or camelCase depending on scope
- Namespaces: lowercase (`cppfort::ir`)

## Formatting

- Indentation: 4 spaces (no tabs)
- Braces: Same line for functions/methods, new line for classes
- Line length: Soft 100 chars, hard 120 chars
- Pointer/Reference: `Type* ptr` and `Type& ref` (align with type)

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
- Manual memory management acceptable for performance-critical graph structures (Node system)

## Compiler Compatibility

Target: C++17 minimum

- GCC 7+, Clang 5+, MSVC 2017+
- Avoid C++20/23 features until broader adoption
