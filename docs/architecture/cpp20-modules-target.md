# C++20 Modules (.cppm) as Sea of Nodes Target

**Status:** Appropriate primary target for SON-guided lowering

## Why C++20 Modules Are Ideal for Sea of Nodes

### 1. Clean Module Semantics

**Traditional C++ (header hell):**
```cpp
// foo.h
#ifndef FOO_H
#define FOO_H
#include <vector>
#include <memory>
// ... preprocessing mess
#endif
```

**C++20 modules (.cppm):**
```cpp
// foo.cppm
export module foo;
import std;

export class Foo { /* ... */ };
```

**Advantage:** Explicit boundaries match Sea of Nodes module structure.

### 2. Sea of Nodes Module Lowering

**SON graph structure:**
```
ModuleNode "foo"
  ↓ exports
  FunctionNode "bar"
  ↓ imports
  ModuleNode "std"
```

**Direct mapping to .cppm:**
```cpp
export module foo;    // ModuleNode
import std;           // Import dependency

export int bar() {    // Exported FunctionNode
    return 42;
}
```

**Pattern matching:**
```cpp
Pattern modulePattern = engine.createPattern()
    .match<ModuleNode>()
    .rewrite([](ModuleNode* m) {
        emit("export module ");
        emit(m->name());
        emit(";\n");
        for (auto* imp : m->imports()) {
            emit("import ");
            emit(imp->name());
            emit(";\n");
        }
    });
```

### 3. Better Build Performance

**Traditional C++ compilation:**
```
Parse header.h (10,000 lines)
  ↓ for EVERY .cpp file
Preprocess macros
  ↓
Parse again
  ↓
Total: O(n * m) where n = files, m = header size
```

**C++20 modules:**
```
Parse module once
  ↓
Cache compiled module interface
  ↓
Reuse cached module
  ↓
Total: O(n) - parse each module once
```

**Advantage:** Faster iteration during development.

### 4. Explicit Export Control

**Sea of Nodes visibility:**
```cpp
class FunctionNode {
    bool _is_exported;  // Public API
    bool _is_internal;  // Private impl
};
```

**Maps to C++20 exports:**
```cpp
export module foo;

export int public_api();     // SONFunctionNode(_is_exported=true)

int internal_helper();       // SONFunctionNode(_is_internal=true)
```

**Advantage:** SON visibility semantics map directly to module exports.

### 5. No Macro Pollution

**Traditional C++ problem:**
```cpp
// lib.h
#define MAX 100  // Oops, leaked to all includers

// user.cpp
#include "lib.h"
#define MAX 200  // Conflict!
```

**C++20 modules:**
```cpp
// lib.cppm
export module lib;
// No macros leak outside module boundary
```

**Advantage:** SON doesn't have macros - modules align philosophically.

## Sea of Nodes → C++20 Pattern Examples

### Example 1: Module Declaration

**SON graph:**
```
ModuleNode("geometry")
  exports: [Point, Line, Circle]
  imports: [std, math]
```

**Lowered .cppm:**
```cpp
export module geometry;
import std;
import math;

export class Point { /* ... */ };
export class Line { /* ... */ };
export class Circle { /* ... */ };
```

### Example 2: Template Functions

**SON graph:**
```
TemplateFunctionNode("max")
  type_params: [T]
  constraint: Comparable<T>
  body: ...
```

**Lowered .cppm:**
```cpp
export module algorithms;

export template<typename T>
    requires std::totally_ordered<T>
T max(T a, T b) {
    return (a > b) ? a : b;
}
```

**Pattern:**
```cpp
Pattern templatePattern = engine.createPattern()
    .match<TemplateFunctionNode>()
    .whereConstraint(hasConceptConstraint())
    .rewrite([](TemplateFunctionNode* fn) {
        emit("export template<typename ");
        emit(fn->typeParam());
        emit(">\n    requires ");
        emit(fn->conceptConstraint());
        emit("\n");
        emitFunctionSignature(fn);
        emitFunctionBody(fn);
    });
```

### Example 3: Import Dependencies

**SON graph:**
```
ModuleNode("app")
  imports: [
    ModuleNode("std"),
    ModuleNode("geometry"),
    ModuleNode("algorithms")
  ]
```

**Lowered .cppm:**
```cpp
export module app;
import std;
import geometry;
import algorithms;

export void run() {
    auto p = Point{1, 2};
    // ...
}
```

## Why Not Legacy C++ Headers?

### Problem 1: No Module Boundaries

**Headers:**
```cpp
// foo.h - everything is global
class Foo { /* ... */ };
```

**Modules:**
```cpp
// foo.cppm - explicit exports
export module foo;
export class Foo { /* ... */ };  // Only this is public
class FooImpl { /* ... */ };     // Private to module
```

### Problem 2: Include Order Matters

**Headers:**
```cpp
#include "a.h"
#include "b.h"  // May break if order swapped
```

**Modules:**
```cpp
import a;
import b;  // Order independent
```

### Problem 3: Compilation Speed

**Headers:** Parse repeatedly (once per includer)
**Modules:** Parse once, cache compiled interface

## Integration with CPP2 Semantics

### CPP2 Philosophy Aligns with Modules

**CPP2 goals:**
- Simplify C++ complexity
- Better defaults
- Explicit semantics
- No legacy baggage

**C++20 modules:**
- Explicit module boundaries (vs include mess)
- No macro leakage (clean semantics)
- Better build performance (better defaults)
- Modern C++ (no legacy baggage)

**Perfect alignment.**

### CPP2 → SON → .cppm Pipeline

```
CPP2 Source
    ↓ parse
Sea of Nodes IR
    ↓ optimize (Band 1-4)
Sea of Nodes (optimized)
    ↓ pattern match
C++20 Module (.cppm)
    ↓ compile
Module Binary Interface (.ifc or .pcm)
    ↓ link
Executable
```

**Each stage has clean semantics:**
- CPP2: High-level intent
- SON: Optimization IR
- .cppm: Modern C++ target
- .ifc: Compiled module

## Comparison: C++20 vs Other Targets

| Target | Semantics Clarity | Build Speed | Interop | Maturity |
|--------|------------------|-------------|---------|----------|
| **C++20 .cppm** | ✅ Excellent | ✅ Fast | ✅ Native C++ | ⚠️ New (2020+) |
| Legacy C++ | ⚠️ Headers messy | ❌ Slow | ✅ Universal | ✅ Mature |
| C | ✅ Simple | ✅ Fast | ✅ Universal | ✅ Mature |
| MLIR | ✅ Excellent | ✅ Fast | ⚠️ Limited | ⚠️ Niche |
| Rust | ✅ Excellent | ⚠️ Medium | ❌ FFI only | ✅ Modern |

**C++20 modules: Best semantic match for CPP2 → SON → Modern C++.**

## Compiler Support (2025)

| Compiler | C++20 Modules Support | Status |
|----------|----------------------|--------|
| **GCC 14+** | Full | ✅ Stable |
| **Clang 17+** | Full | ✅ Stable |
| **MSVC 19.28+** | Full | ✅ Stable |
| **Intel oneAPI** | Full | ✅ Stable |

**All major compilers support C++20 modules as of 2025.**

## Implementation Strategy

### Phase 1: Basic Module Lowering

```cpp
void lowerToModule(ModuleNode* m, std::ostream& out) {
    // Module declaration
    out << "export module " << m->name() << ";\n";

    // Imports
    for (auto* imp : m->imports()) {
        out << "import " << imp->name() << ";\n";
    }
    out << "\n";

    // Exports
    for (auto* exp : m->exports()) {
        out << "export ";
        emitDeclaration(exp, out);
    }
}
```

### Phase 2: Pattern-Based Lowering

```tablegen
// patterns.td
def SONModuleToModuleDecl : Pat<
    (SON_ModuleNode $name),
    (CXX20_ModuleDecl $name)
>;

def SONImportToImportDecl : Pat<
    (SON_ImportNode $name),
    (CXX20_ImportDecl $name)
>;

def SONExportFunctionToExportFunc : Pat<
    (SON_FunctionNode $name, (IsExported)),
    (CXX20_ExportFunc $name)
>;
```

### Phase 3: Optimization

```cpp
// Module-aware optimizations
void optimizeModule(ModuleNode* m) {
    // Only exported functions need full optimization
    for (auto* fn : m->exports()) {
        applyAggressiveOptimization(fn);
    }

    // Internal functions can use simpler optimizations
    for (auto* fn : m->internals()) {
        applyBasicOptimization(fn);
    }
}
```

## Build System Integration

### CMakeLists.txt Example

```cmake
# Enable C++20 modules
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API "3e8dd03e-4273-4e3b-81e5-1febd9ed48a7")

# Define module
add_library(geometry)
target_sources(geometry
  PUBLIC
    FILE_SET CXX_MODULES FILES
      geometry.cppm
)

# Use module
add_executable(app main.cpp)
target_link_libraries(app PRIVATE geometry)
```

**cppfort can generate both .cppm files and CMakeLists.txt.**

## Advantages for Cppfort

1. **Clean lowering target** - SON module structure → .cppm module structure
2. **Better build times** - Modules compile once, cached
3. **Semantic alignment** - CPP2 philosophy matches module philosophy
4. **Modern C++** - Target forward-looking standard
5. **Interop** - Still C++, works with existing code
6. **Tooling support** - All major compilers (2025+)

## Recommendation

**Primary target: C++20 modules (.cppm)**

**Secondary targets:**
- Legacy C++ (for compatibility)
- MLIR (for advanced lowering)
- C (for embedded/minimal runtime)

**Why .cppm first:**
- Best semantic match for CPP2
- Clean module boundaries
- Modern C++ evolution path
- Native performance
- Universal compiler support (2025+)

**SON-guided .cppm lowering is the appropriate default target for cppfort.**
