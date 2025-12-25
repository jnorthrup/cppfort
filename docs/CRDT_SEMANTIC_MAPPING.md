# CRDT Semantic Mapping: Cpp2 ↔ C++ Bidirectional AST Transform

## Problem Statement

The Pijul CRDT integration requires:
1. **SHA256 Merkle Tree hashing** of AST nodes for content-addressed storage
2. **Bidirectional semantic mapping** between cpp2 AST and Clang AST
3. **Semantic equivalence preservation** when transforming between representations

Current state: cpp2 → C++ transpilation works (forward direction)
Missing: C++ (Clang AST) → cpp2 AST reverse mapping with semantic equivalence

## Architecture

```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│   cpp2      │ ◄──────►│   Semantic  │ ◄──────►│  Clang AST  │
│   AST       │         │   Equiv     │         │  (C++ AST)  │
└─────────────┘         └─────────────┘         └─────────────┘
       │                       │                       │
       │                       ▼                       │
       │              ┌─────────────┐                 │
       └─────────────►│  CRDT Graph│◄────────────────┘
                      │  (Pijul)    │
                      └─────────────┘
                            │
                            ▼
                      ┌─────────────┐
                      │  SHA256     │
                      │  Merkle     │
                      │  Tree       │
                      └─────────────┘
```

## Semantic Hash Design

### Hash Computation

Each AST node's hash is computed from its **semantic content**, not textual representation:

```cpp
struct SemanticHash {
    uint64_t node_id;              // Unique ID within translation unit
    std::string sha256_hash;        // SHA256 of semantic content
    std::vector<uint64_t> children; // Child node IDs

    // Merkle tree hash computation
    std::string compute_merkle_hash() const {
        std::string combined = sha256_hash;
        for (auto child : children) {
            combined += child_hash;
        }
        return sha256(combined);
    }
};
```

### Semantic Content for Different Node Types

| Node Type | Semantic Content (for hashing) |
|-----------|-------------------------------|
| FunctionDeclaration | name + parameter_types + return_type |
| VariableDeclaration | name + type + initializer_expr_hash |
| BinaryExpression | operator_hash + left_hash + right_hash |
| CallExpression | callee_hash + argument_hashes |
| ParameterDeclaration | name + type + qualifiers |
| LiteralExpression | type + value |

## Bidirectional Mapping Rules

### 1. Function Signatures

**Cpp2 → C++:**
```cpp2
// Cpp2
name: (inout x: int, move y: std::string) -> std::string
```
```cpp
// C++ (Clang AST)
auto name(int& x, std::string&& y) -> std::string
```

**C++ → Cpp2 (Reverse Mapping):**
```cpp
// Clang AST Pattern:
FunctionDecl
├─ ParmVarDecl 'x' with QualType 'int &'
├─ ParmVarDecl 'y' with QualType 'std::string &&'
└─ ReturnType 'std::string'

// Inferred Cpp2 Semantics:
name: (inout x: int, move y: std::string) -> std::string
```

### 2. Parameter Qualifier Mapping

| Cpp2 Qualifier | C++ Type | Clang AST Pattern | Reverse Rule |
|----------------|----------|-------------------|--------------|
| `inout` | `T&` | `ParmVarDecl` with `LValueReferenceType` | Map `&` to `inout` |
| `out` | `T&` | `ParmVarDecl` with `LValueReferenceType` + definite assignment analysis | Map `&` to `out` if not read before write |
| `move` | `T&&` | `ParmVarDecl` with `RValueReferenceType` | Map `&&` to `move` |
| `forward` | `auto&&` / `T&&` | `ParmVarDecl` with `RValueReferenceType` + template context | Map `&&` to `forward` in template |
| `in` | `const T&` or `T` | `ParmVarDecl` with `ConstQualifier` or non-reference | Map `const T&` or `T` to `in` |

### 3. Variable Declarations

**Cpp2 → C++:**
```cpp2
x: std::string = "hello"
x := "hello"  // type deduced
```
```cpp
std::string x = "hello";
auto x = "hello";
```

**C++ → Cpp2 (Reverse Mapping):**
```cpp
// Clang AST: VarDecl with Type 'std::string'
// Inferred Cpp2: x: std::string = "hello"

// Clang AST: VarDecl with Type 'auto' (deduced)
// Inferred Cpp2: x := "hello"
```

### 4. UFCS (Unified Function Call Syntax)

**Cpp2 → C++:**
```cpp2
obj.method(arg)      // Member call
method(obj, arg)     // UFCS: obj.method() desugars to method(obj)
```
```cpp
obj.method(arg);     // Same
method(obj, arg);    // Same
```

**C++ → Cpp2 (Reverse Mapping):**
```cpp
// Clang AST: CXXMemberCallExpr
// Inferred Cpp2: obj.method(arg)

// Clang AST: CallExpr with first arg matching UFCS candidate
// Inferred Cpp2: method(obj) or obj.method() (preference based on style)
```

## CRDT Integration

### Patch Generation

```cpp
struct CRDTPatch {
    enum class Op {
        InsertNode,    // Insert new AST node
        DeleteNode,    // Delete AST node
        UpdateNode,    // Update node content
        MoveNode       // Move node within tree
    };

    Op operation;
    std::string node_hash;        // Target node's Merkle hash
    std::string parent_hash;      // Parent node's Merkle hash
    uint64_t timestamp;
    std::vector<std::string> dependencies;  // Hashes of depended nodes
};
```

### Merge Algorithm

1. **Compute Merkle hashes** for both AST versions
2. **Find LCA (Lowest Common Ancestor)** of changed nodes
3. **Apply patches** using Pijul's conflict-free merge
4. **Validate semantic consistency** post-merge

## Implementation Plan

### Phase 1: Semantic Hash Computation
- [ ] Add `compute_semantic_hash()` to each AST node type
- [ ] Implement Merkle tree propagation
- [ ] Add SHA256 hashing utilities

### Phase 2: C++ → Cpp2 Reverse Mapping
- [ ] Implement Clang AST visitor
- [ ] Add pattern matching for Cpp2 semantics
- [ ] Handle parameter qualifier inference
- [ ] Handle type inference (`auto` → `:=`)

### Phase 3: CRDT Integration
- [ ] Implement patch generation from AST diffs
- [ ] Add Pijul-style conflict resolution
- [ ] Implement merge algorithm
- [ ] Add semantic validation post-merge

### Phase 4: Testing
- [ ] Unit tests for hash computation
- [ ] Unit tests for bidirectional mapping
- [ ] Integration tests for CRDT merge
- [ ] Regression tests on corpus

## Challenges

### 1. Information Loss in Forward Mapping

Some Cpp2 semantics don't survive transpilation:
- `inout` vs `out` (both become `&`)
- Definite initialization guarantees
- Contract annotations
- UFCS call sites (identical in C++)

**Solution:** Encode Cpp2-specific semantics as attributes in Clang AST or store separately in sidecar metadata.

### 2. Ambiguity in Reverse Mapping

Multiple Cpp2 syntaxes can produce identical C++:
- `x: int = 5` vs `x := 5` (both: `int x = 5;`)
- `inout` vs `out` (both: `T&`)

**Solution:** Use dataflow analysis to infer qualifiers (e.g., `out` if never read before write).

### 3. Whitespace/Formatting Preservation

CRDT merges should not destroy formatting preferences.

**Solution:** Separate semantic AST from formatted output (use persistent trees for formatting).

## References

- Pijul: https://pijul.org/
- Clang AST: https://clang.llvm.org/docs/IntroductionToTheClangAST.html
- Cpp2 Specification: https://github.com/hsutter/cppfront
- Merkle Trees: https://en.wikipedia.org/wiki/Merkle_tree
