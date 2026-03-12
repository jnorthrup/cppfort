# Product

## cppfort - Sea-of-Nodes Cpp2 Transpiler

cppfort is a Cpp2-to-C++ transpiler built around a Sea-of-Nodes (SoN) intermediate representation powered by MLIR. It implements the TrikeShed architectural course correction: **semantic objects first, dense lowered views second**.

### TrikeShed Strategy: Front-End Sugar as Zero-Cost Abstraction Core

**Core Principle**: Front-end sugar IS the abstraction mechanism. The compiler's job is not to "compile away" syntax but to **normalize it to canonical forms** while preserving semantic intent.

**Architecture**:
```
TrikeShed Surface Syntax (front-end sugar)
    ↓ (early normalization - semantic preservation)
Canonical AST (small, repo-owned, zero-cost)
    ↓ (Sea-of-Nodes + constant propagation)
Optimized IR (zero-cost abstractions proven)
    ↓ (MLIR lowering)
Target Code (optimal, no abstraction overhead)
```

**Key Insights**:
1. **Zero-cost means**: Type alias = free hoisted vtable in real-world front IR
2. **Front-end sugar is core**: Operators, underscore patterns, manifold notation are PRIMARY user interface
3. **Normalization is semantic preservation**: Transform surface syntax to canonical AST without losing intent
4. **SoN does the optimization**: Constant propagation, alias analysis, effect recovery happens in MLIR
5. **Manifold applicability**: Unique challenge for SoN compilation and lifecycle memory management without the maths

**cppfront as Temporary Benchmark**:
- cppfront is used as a **benchmark/validator only**, not a build dependency
- No cppfront linking, calling, or inclusion in the build process
- Temporary ride for bootstrap compatibility, removable once native parser lands

### Current Status and Architecture

**Active Architecture**:
- **Front-end sugar as core**: TrikeShed surface syntax is the primary abstraction mechanism
- **Early normalization**: Surface sugar normalizes immediately to canonical AST
- **SoN optimization**: Sea-of-Nodes + constant propagation does the heavy lifting
- **MLIR lowering**: Canonical AST → SoN → target code with zero abstraction overhead

**Implementation Status**:
- **Live restart path**: `selfhost/` is the current source of truth
- **Build target**: `selfhost_bootstrap_smoke` is the authoritative top-level build target
- **cppfront role**: Temporary benchmark/validator only (not a build dependency)

**Manifold Clarification**:
- **Compiler-process guidance**: Charts, atlases, coordinates for program forms
- **Semantic transitions**: Guides normalization and lowering phases
- **NOT**: Model training, learned classification, embeddings, or statistical inference

**I/O Strategy Decision**:
- **stdio/mmap vs. channelized reactor**: For clarity, use simple stdio between build stages
- **Rationale**: Channelized reactor adds complexity for marginal benefit in a compiler pipeline
- **Exception**: Use memory-mapped I/O only for large intermediate representations

## Core Architecture

### Canonical Semantic Layer (Templates)

The foundation is a small set of template-based canonical types that normalize early and feed directly into SoN:

```cpp2
// indexed<I, F> - function from domain I to value type
template<typename I, typename F>
class indexed {
    I domain;
    F at;
public:
    using value_type = decltype(at(declval<I>()));
    operator[](i: I) -> value_type = at(i);
    size() -> I = domain;
};

// series<F> - indexed sequence (indexed<int, F>)
template<typename F>
using series = indexed<int, F>;

// tensor<K, F> - sparse tensor with coordinate keys
template<typename K, typename F>
class tensor {
    series<axis<K>> axes;
    F at;  // coord<K> -> T
};

// dense_tensor<K, T> - dense array with semantic shape + lowered strides
template<typename K, typename T>
class dense_tensor {
    series<axis<K>> axes;    // semantic shape
    series<int> strides;     // lowering
    span<T> data;            // dense backing
    operator[](flat: int) -> T& = data[flat];
};

// atlas<C, Chart> - chart management for manifolds
template<typename C, typename Chart>
class atlas {
    indexed<C, Chart> charts;
    operator()(c: C) -> Chart = charts[c];
};

// manifold<C, Chart> - smooth coordinate space
template<typename C, typename Chart>
using manifold = atlas<C, Chart>;
```

### Gradient Protocol

Differentiable programming support via protocol, not library capture:

```cpp2
template<typename E>
concept grad_expr = requires(E e) {
    { e + e } -> same_as<E>;
    { e - e } -> same_as<E>;
    { e * e } -> same_as<E>;
    { e / e } -> same_as<E>;
};

template<typename E, typename V>
struct grad_backend {
    auto constant(double) -> E;
    auto variable(std::string_view) -> V;
    auto diff(E, V) -> E;
    auto eval(E, bindings) -> double;
};
```

## MLIR SoN Pipeline

### Dialect Operations

The canonical types lower to MLIR SoN operations:

- `cpp2.indexed` - coordinate-based indexing
- `cpp2.series` - indexed sequence
- `cpp2.tensor` - sparse tensor
- `cpp2.dense_tensor` - dense array with contracts
- `cpp2.atlas` - chart management
- `cpp2.manifold` - smooth coordinate spaces
- `cpp2.grad_diff` - automatic differentiation
- `cpp2.jacobian` - Jacobian matrix computation
- `cpp2.matrix_mul` - chain rule aggregation

### SoN Passes

- **SoNConstantProp**: Template parameter folding, dead code elimination
- **GradADLowering**: Protocol-to-arithmetic lowering for AD
- **JacobianMatrixMulLowering**: Manifold chain rule to fused multiply-add

## Parser Architecture

### Public API (cppfort_parser.h)

100% hand-written parser contract:

```cpp
struct ParseResult {
    std::unique_ptr<CanonicalAST> ast;
    std::vector<std::string> errors;
    bool success() const;
};

class Parser {
    static ParseResult parse(std::string_view source, std::string_view filename);
    static ParseResult parse_with_trikeshed(std::string_view source,
                                           bool enable_operators = true,
                                           bool enable_underscores = true);
    static bool can_self_parse();
};
```

### Normalization Flow

```
TrikeShed sugar (text)
    ↓ (LLM transpiler)
pure Cpp2 using canonical types
    ↓ (cppfront bootstrap)
C++23 modules + CAS pool
    ↓
Parser::parse_with_trikeshed
    ↓
CanonicalAST (indexed/series/tensor/dense_tensor/atlas)
    ↓
Sea-of-Nodes + MLIR
    ↓
Dense lowered views
```

## CAS Linker Internments

Java classfile analog for compile-time constant deduplication:

```cpp
[[gnu::section(".cas_pool")]]
constexpr struct CASPool {
    const char* strings[4096];
    // type IDs, series literals, axis constants, grad_expr nodes
} cas_pool = { };
```

## Build System

- CMake-based build with Ninja generator
- MLIR/LLVM integration via llvm-project
- Self-host bootstrap under `selfhost/`
- Archive legacy at `old/cppfort`

## Key Principles

1. **Semantic objects first**: Canonical types represent the mathematical/algorithmic intent
2. **Dense lowered views second**: Separate representation for optimized runtime data
3. **Early normalization**: Surface sugar normalizes immediately to small canonical AST
4. **SoN does the work**: Constant propagation, alias analysis, effect recovery in MLIR
5. **Templates, not constexpr**: Raw generics stay in source; compiler smashes to constants
6. **No safe language arena**: No constexpr factories, reflection gymnastics, or Python/TableGen

---

## Implementation Status (2026-03-12)

### What's Actually Implemented

| Component | Status | Location |
|-----------|--------|----------|
| Parser API contract | **HEADER ONLY** - no implementation | [`cppfort_parser.h`](cppfort_parser.h) |
| Canonical type templates | **DECLARED** - not wired into build | [`selfhost/canonical_types.cpp2`](selfhost/canonical_types.cpp2) |
| Bootstrap tags | **BUILT** - integer constants only | [`selfhost/bootstrap_tags.cpp2`](selfhost/bootstrap_tags.cpp2) |
| MLIR SoN dialect | **TABLEGEN DEFINED** - disabled in build | [`include/Cpp2SONDialect.td`](include/Cpp2SONDialect.td) |
| CAS internment types | **HEADER ONLY** - no implementation | [`cppfort_parser.h:114-132`](cppfort_parser.h:114) |

### Critical Gaps

1. **Parser has no implementation**: `Parser::Impl` has no definition. `src/` directory is empty.
2. **MLIR dialect disabled**: LLVM 21 FieldParser issue blocks SoN dialect compilation.
3. **Bootstrap transpilation limited**: `old/cppfort` binary can only handle trivial declarations.
4. **No canonical → SoN lowering**: No wired path from `canonical_types.cpp2` to MLIR operations.

### Rewrite Merit

A fresh start would:
- Implement parser from `cppfort_parser.h` contract forward
- Fix or bypass LLVM 21 issue to enable MLIR dialect
- Wire `canonical_types.cpp2` into build with actual C++ transpilation
- Start with one working SoN op lowering, not full pipeline
