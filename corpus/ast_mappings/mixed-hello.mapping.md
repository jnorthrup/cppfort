# AST Mapping: mixed-hello.cpp2

## Source Files
- **Cpp2**: `corpus/inputs/mixed-hello.cpp2`
- **C++1**: `corpus/ast_mappings/mixed-hello.cpp`
- **Clang AST**: `corpus/ast_mappings/mixed-hello.ast.txt`

## Semantic Mappings (Cpp2 → C++1 → Clang AST)

### 1. Function Declaration with Trailing Return Type

**Cpp2 Syntax:**
```cpp2
name: () -> std::string = {
    ...
}
```

**C++1 Translation:**
```cpp
auto name() -> std::string {
    ...
}
```

**Clang AST Node:**
```
FunctionDecl <line:8:1, line:16:1> line:8:6 name 'std::string ()'
├─CompoundStmt
└─ReturnStmt
```

**Mapping:**
- Cpp2 postfix function signature `name: () -> std::string` → C++1 trailing return type `auto name() -> std::string`
- Cpp2 `= {` → C++1 `{` (function body)
- Return type placement: postfix → trailing

---

### 2. Local Variable Declaration with Type Inference

**Cpp2 Syntax:**
```cpp2
s: std::string = "world";
```

**C++1 Translation:**
```cpp
std::string s = "world";
```

**Clang AST Node:**
```
VarDecl <line:10:5, col:29> col:17 s 'std::string':'std::basic_string<char>'
└─StringLiteral <col:25, col:29> 'const char[6]' lvalue "world"
```

**Mapping:**
- Cpp2 name-first `s: std::string` → C++1 type-first `std::string s`
- Type annotation position: postfix → prefix
- Initialization syntax identical: `= "world"`

---

### 3. Function Call (UFCS)

**Cpp2 Syntax:**
```cpp2
decorate(s);
```

**C++1 Translation:**
```cpp
decorate(s);
```

**Clang AST Node:**
```
CallExpr <line:13:5, col:16> 'void'
├─ImplicitCastExpr <col:5> 'void (*)(std::string &)' <FunctionToPointerDecay>
│ └─DeclRefExpr <col:5> 'void (std::string &)' lvalue Function 'decorate'
└─DeclRefExpr <col:14> 'std::string':'std::basic_string<char>' lvalue Var 's' 'std::string':'std::basic_string<char>'
```

**Mapping:**
- Cpp2 UFCS call `decorate(s)` → C++1 regular call `decorate(s)`
- Note: In more complex cases, Cpp2 `s.decorate()` would also map to `decorate(s)`
- AST represents as standard CallExpr with function pointer decay

---

### 4. Parameter Declaration with `inout` Qualifier

**Cpp2 Syntax:**
```cpp2
decorate: (inout s: std::string) = { ... }
```

**C++1 Translation:**
```cpp
auto decorate(std::string& s) -> void { ... }
```

**Clang AST Node:**
```
FunctionDecl <line:19:1, line:22:1> line:19:6 decorate 'void (std::string &)'
├─ParmVarDecl <line:19:16, col:29> col:29 s 'std::string &'
└─CompoundStmt
```

**Mapping:**
- Cpp2 `inout` qualifier → C++1 `&` (lvalue reference)
- Cpp2 `(inout s: std::string)` → C++1 `(std::string& s)`
- Parameter semantics: pass-by-mutable-reference
- Cpp2 guarantees definite initialization for `inout` parameters

---

### 5. String Concatenation Expression

**Cpp2 Syntax:**
```cpp2
s = "[" + s + "]";
```

**C++1 Translation:**
```cpp
s = "[" + s + "]";
```

**Clang AST Node:**
```
BinaryOperator <line:21:5, col:23> 'std::string':'std::basic_string<char>' '='
├─DeclRefExpr <col:5> 'std::string':'std::basic_string<char>' lvalue Var 's'
└─CXXOperatorCallExpr <col:9, col:23> 'std::string':'std::basic_string<char>'
  ├─ImplicitCastExpr <col:19> 'std::string (*)(std::string &&, const char (&)[2])'
  ├─CXXOperatorCallExpr <col:9, col:17> 'std::string':'std::basic_string<char>'
  │ ├─ImplicitCastExpr 'std::string (*)(const char (&)[2], const std::string &)'
  │ ├─StringLiteral <col:9> 'const char[2]' lvalue "["
  │ └─ImplicitCastExpr <col:15> 'const std::string':'const std::basic_string<char>' lvalue
  │   └─DeclRefExpr <col:15> 'std::string':'std::basic_string<char>' lvalue Var 's'
  └─StringLiteral <col:21, col:23> 'const char[2]' lvalue "]"
```

**Mapping:**
- Syntax identical in Cpp2 and C++1
- AST reveals operator overloading resolution:
  - Left-associative: `("[" + s) + "]"`
  - Two `CXXOperatorCallExpr` nodes (one nested)
  - Operator overload: `std::string operator+(const char*, const std::string&)`

---

### 6. Main Function

**Cpp2 Syntax:**
```cpp2
auto main() -> int {
    std::cout << "Hello " << name() << "\n";
}
```

**C++1 Translation:**
```cpp
auto main() -> int {
    std::cout << "Hello " << name() << "\n";
}
```

**Clang AST Node:**
```
FunctionDecl <line:24:1, line:26:1> line:24:6 main 'int ()'
└─CompoundStmt <line:24:20, line:26:1>
  └─CXXOperatorCallExpr <line:25:5, col:44> 'std::ostream':'std::basic_ostream<char>' lvalue
    ├─ImplicitCastExpr <col:38> 'std::ostream &(*)(std::ostream &, const char *)'
    ├─CXXOperatorCallExpr <col:5, col:36> 'std::ostream':'std::basic_ostream<char>' lvalue
    │ ├─ImplicitCastExpr <col:32> 'std::ostream &(*)(std::ostream &, std::string &&)'
    │ ├─CXXOperatorCallExpr <col:5, col:28> 'std::ostream':'std::basic_ostream<char>' lvalue
    │ │ ├─ImplicitCastExpr <col:20> 'std::ostream &(*)(std::ostream &, const char *)'
    │ │ ├─DeclRefExpr <col:5, col:10> 'std::ostream':'std::basic_ostream<char>' lvalue Var 'std::cout'
    │ │ └─ImplicitCastExpr <col:15> 'const char *' <ArrayToPointerDecay>
    │ │   └─StringLiteral <col:15, col:19> 'const char[7]' lvalue "Hello "
    │ └─CallExpr <col:25, col:30> 'std::string':'std::basic_string<char>'
    │   └─ImplicitCastExpr <col:25> 'std::string (*)()'
    │     └─DeclRefExpr <col:25> 'std::string ()' lvalue Function 'name'
    └─ImplicitCastExpr <col:40> 'const char *' <ArrayToPointerDecay>
      └─StringLiteral <col:40, col:44> 'const char[3]' lvalue "\\n"
```

**Mapping:**
- Syntax identical (already C++11-style trailing return)
- Stream insertion chain: `(((cout << "Hello ") << name()) << "\n")`
- Each `<<` operator resolves to `CXXOperatorCallExpr`
- Left-to-right evaluation preserved in AST structure

---

## Summary Table

| Cpp2 Feature | C++1 Equivalent | Clang AST Node Type | Semantic Transform |
|--------------|-----------------|---------------------|-------------------|
| `name: () -> T` | `auto name() -> T` | `FunctionDecl` with trailing return | Postfix → Trailing return type |
| `x: Type = val` | `Type x = val` | `VarDecl` | Name-first → Type-first |
| `inout param: T` | `T& param` | `ParmVarDecl` with `&` | `inout` → lvalue reference |
| `f(x)` (UFCS) | `f(x)` | `CallExpr` | Direct mapping (more complex with member syntax) |
| String ops | String ops | `CXXOperatorCallExpr` | Identical syntax, operator resolution |
| `= { body }` | `{ body }` | `CompoundStmt` | Remove `=` token |

---

## Notes

1. **Forward Declarations Required**: Unlike Cpp2 (which has two-phase parsing), C++1 requires forward declaration for `decorate` before use in `name()`.

2. **Type Deduction**: Cpp2's type-after-name syntax doesn't affect AST structure; both produce identical `VarDecl` nodes.

3. **UFCS Limitation**: This simple example doesn't demonstrate full UFCS (e.g., `s.decorate()` → `decorate(s)`). The mapping is trivial here since both use function call syntax.

4. **Return Type Inference**: Cpp2 could infer `-> void` for `decorate`, but explicit annotation is shown for clarity.

5. **Memory Safety**: Cpp2's `inout` provides stronger guarantees than C++ references (definite initialization, no null). AST doesn't capture this semantic difference.
