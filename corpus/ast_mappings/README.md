# Cpp2 → C++1 → Clang AST Semantic Mappings

This directory contains **n-way semantic mappings** between:
1. **Cpp2 syntax** (from cppfront regression test corpus)
2. **C++1 code** (transpiled output)
3. **Clang AST** (abstract syntax tree representation)

The mappings enable systematic understanding of how Cpp2 language constructs translate to standard C++ and their underlying AST structure.

## Purpose

- **Semantic Analysis**: Understand how Cpp2 features map to C++ semantics
- **Compiler Development**: Reference for building Cpp2 → MLIR/LLVM pipelines
- **Validation**: Verify transpilation correctness against Clang's AST
- **Documentation**: Comprehensive examples of all Cpp2 language features

## Directory Structure

```
ast_mappings/
├── README.md                          # This file
├── mixed-hello.cpp                    # C++1 translation example
├── mixed-hello.ast.txt                # Clang AST dump (text)
├── mixed-hello.ast.json               # Clang AST dump (JSON)
├── mixed-hello.mapping.md             # Detailed semantic mappings
├── ...                                # More test cases
└── MAPPING_SUMMARY.md                 # Summary of all mappings
```

## Mapping Format

Each mapping file (`.mapping.md`) contains:

### 1. Source References
- Original Cpp2 file path
- Generated C++1 file path
- Clang AST dump paths

### 2. Side-by-Side Syntax Comparison
```cpp2
// Cpp2 Syntax
name: () -> std::string = { ... }
```

```cpp
// C++1 Translation
auto name() -> std::string { ... }
```

### 3. Clang AST Structure
```
FunctionDecl 'std::string ()'
├─CompoundStmt
└─ReturnStmt
```

### 4. Semantic Analysis
- **Syntax transformation rules**
- **Type system mappings**
- **Memory safety guarantees**
- **UFCS (Unified Function Call Syntax) resolution**
- **Definite initialization tracking**

## Key Semantic Mappings

### Function Signatures

| Cpp2 | C++1 | AST Node | Semantic |
|------|------|----------|----------|
| `f: () -> int` | `auto f() -> int` | `FunctionDecl` | Postfix → Trailing return |
| `f: (x: int)` | `auto f(int x) -> void` | `FunctionDecl` + `ParmVarDecl` | Type-after-name → Type-first |

### Variable Declarations

| Cpp2 | C++1 | AST Node | Semantic |
|------|------|----------|----------|
| `x: int = 5` | `int x = 5` | `VarDecl` | Name-first → Type-first |
| `x := 5` | `auto x = 5` | `VarDecl` (deduced) | Type inference |
| `x: int`  | (Error) | - | Cpp2 requires initialization |

### Parameter Qualifiers

| Cpp2 Qualifier | C++1 Equivalent | AST Representation | Semantic Guarantee |
|----------------|-----------------|--------------------|--------------------|
| `in param: T` | `const T& param` or `T param` | `ParmVarDecl` with `const &` | Read-only access |
| `inout param: T` | `T& param` | `ParmVarDecl` with `&` | Mutable reference, definite init |
| `out param: T` | `T& param` | `ParmVarDecl` with `&` | Definite assignment before return |
| `move param: T` | `T&& param` | `ParmVarDecl` with `&&` | Move semantics |
| `forward param: T` | `auto&& param` | `ParmVarDecl` with `&&` | Perfect forwarding |

### UFCS (Unified Function Call Syntax)

| Cpp2 | C++1 | AST | Resolution |
|------|------|-----|-----------|
| `obj.func()` | `obj.func()` | `CXXMemberCallExpr` | Member function |
| `obj.func()` | `func(obj)` | `CallExpr` | Free function via UFCS |
| `func(obj)` | `func(obj)` | `CallExpr` | Regular call |

### Expressions

| Cpp2 | C++1 | AST Node | Notes |
|------|------|----------|-------|
| `x + y` | `x + y` | `BinaryOperator` or `CXXOperatorCallExpr` | Depends on type |
| `x.y` | `x.y` | `MemberExpr` | Member access |
| `x.f()` | `x.f()` or `f(x)` | `CXXMemberCallExpr` or `CallExpr` | UFCS |
| `x[i]` | `x[i]` | `CXXOperatorCallExpr` | Bounds-checked in Cpp2 |

### Control Flow

| Cpp2 | C++1 | AST Node | Semantic Difference |
|------|------|----------|---------------------|
| `if condition { }` | `if (condition) { }` | `IfStmt` | Cpp2 requires boolean, no implicit conversion |
| `while condition { }` | `while (condition) { }` | `WhileStmt` | Same |
| `for x in range { }` | `for (auto&& x : range) { }` | `CXXForRangeStmt` | Cpp2 always uses forwarding reference |

### Type System

| Cpp2 Type | C++1 Type | AST `Type` | Semantic |
|-----------|-----------|------------|----------|
| `int` | `int` | `BuiltinType` | Built-in integer |
| `i32` | `std::int32_t` | `TypedefType` | Fixed-width integer |
| `*T` | `T*` | `PointerType` | Raw pointer (unsafe) |
| `std::unique_ptr<T>` | `std::unique_ptr<T>` | `TemplateSpecializationType` | Owned pointer |
| `std::shared_ptr<T>` | `std::shared_ptr<T>` | `TemplateSpecializationType` | Shared ownership |

## Generating Mappings

### Prerequisites

1. **cppfront compiler** (for Cpp2 → C++1 transpilation)
   ```bash
   # Build cppfront
   cd third_party/cppfront
   cmake -B build
   cmake --build build
   ```

2. **Clang/LLVM** (for AST generation)
   ```bash
   clang++ --version  # Should be 14+
   ```

### Usage

```bash
# Generate mappings for all corpus files
./tools/generate_ast_mappings.py

# Generate for specific pattern
./tools/generate_ast_mappings.py --pattern "mixed-*.cpp2"

# Limit to first N files (for testing)
./tools/generate_ast_mappings.py --limit 5

# Specify cppfront path
./tools/generate_ast_mappings.py --cppfront /path/to/cppfront
```

### Manual Mapping Creation

For files where cppfront isn't available:

1. **Write C++1 equivalent** manually
2. **Generate AST dump**:
   ```bash
   clang++ -std=c++20 -Xclang -ast-dump -fsyntax-only file.cpp > file.ast.txt
   clang++ -std=c++20 -Xclang -ast-dump=json -fsyntax-only file.cpp > file.ast.json
   ```
3. **Document mappings** in `file.mapping.md`

## AST Analysis Tools

### Viewing AST Dumps

**Text format** (human-readable):
```bash
cat mixed-hello.ast.txt | less
```

**JSON format** (machine-readable):
```bash
cat mixed-hello.ast.json | jq . | less
```

### Extracting Patterns

```bash
# Find all function declarations
grep "FunctionDecl" mixed-hello.ast.txt

# Find all variable declarations
grep "VarDecl" mixed-hello.ast.txt

# Find operator calls
grep "CXXOperatorCallExpr" mixed-hello.ast.txt
```

### Python AST Analysis

```python
import json

with open("mixed-hello.ast.json") as f:
    ast = json.load(f)

# Traverse AST
def find_nodes(node, kind):
    results = []
    if node.get("kind") == kind:
        results.append(node)
    for child in node.get("inner", []):
        results.extend(find_nodes(child, kind))
    return results

# Find all function declarations
funcs = find_nodes(ast, "FunctionDecl")
for func in funcs:
    print(func["name"], func["type"]["qualType"])
```

## Integration with Cppfort

These mappings inform the design of:

1. **Lexer**: Tokenization rules for Cpp2 syntax
2. **Parser**: AST construction for Cpp2 programs
3. **Semantic Analyzer**: Type checking and name resolution
4. **Code Generator**:
   - C++1 backend: Use direct mappings
   - MLIR backend: Transform via intermediate representation
5. **Safety Checker**: Definite initialization, bounds checking, lifetime analysis

## Reference Documentation

- **Cpp2 Specification**: https://github.com/hsutter/cppfront
- **Clang AST**: https://clang.llvm.org/docs/IntroductionToTheClangAST.html
- **MLIR Dialects**: https://mlir.llvm.org/docs/Dialects/

## Contributing

When adding new mappings:

1. Use a consistent format (see `mixed-hello.mapping.md`)
2. Include both text and JSON AST dumps
3. Document **all** semantic differences, not just syntax
4. Note any Cpp2 safety guarantees not present in C++1
5. Link to relevant Cpp2 spec sections

## Next Steps

1. **Automated Mapping Extraction**: Build tools to auto-generate mapping rules from AST analysis
2. **MLIR Translation**: Create Cpp2 → MLIR Cpp2Dialect mappings
3. **Verification**: Cross-check cppfort transpiler output against these mappings
4. **Pattern Library**: Extract common patterns into reusable transformation rules

---

**Last Updated**: See MAPPING_SUMMARY.md for generation timestamp
