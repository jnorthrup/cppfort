# Cppfort Architecture

## MLIR Dialect Integration

Cppfort implements a proper MLIR dialect for Cpp2 compilation using TableGen-generated infrastructure.

### Dialect Definition

**File**: `include/Cpp2Dialect.td`

Defines all operations, types, and attributes for the Cpp2 Sea of Nodes IR:

- **Types**: `Int`, `Ptr<T, nullable>`, `Struct<name>`, `Mem<aliasClass>`
- **Control Ops**: `start`, `if`, `region`, `loop`, `return`
- **Data Ops**: `constant`, `add`, `sub`, `mul`, `div`, `phi`
- **Memory Ops**: `new`, `load`, `store`, `cast`
- **Cpp2 Ops**: `ufcs_call`, `contract`, `metafunction`

### Build Process

```bash
cmake -G Ninja -B build
ninja -C build
```

TableGen generates:
- `Cpp2Ops.h.inc` / `Cpp2Ops.cpp.inc` - Operation definitions
- `Cpp2OpsDialect.h.inc` / `Cpp2OpsDialect.cpp.inc` - Dialect infrastructure
- `Cpp2OpsTypes.h.inc` / `Cpp2OpsTypes.cpp.inc` - Type system

### Architecture Components

#### 1. Pijul CRDT Graph (`src/pijul_crdt.cpp`)

Patch-based graph operations with LWW conflict resolution:

```cpp
struct Patch {
    NodeID target;
    Op operation; // AddNode, RemoveNode, AddEdge, RemoveEdge
    variant<Node, pair<NodeID, NodeID>> data;
};
```

Operations are conflict-free and mergeable across distributed development.

#### 2. Sea of Nodes IR (`src/sea_of_nodes_ir.cpp`)

Graph-based IR following Cliff Click's Simple documentation:

- **Control nodes**: Start, Stop, If, Region, Loop
- **Data nodes**: Constant, Phi, Binary ops
- **Memory nodes**: New, Load, Store with equivalence-class aliasing

**Key features**:
- Forward references for recursive types (Chapter 13)
- Alias class system for memory safety (Chapter 10)
- Global Code Motion scheduler (Chapter 11)

#### 3. Combinator Hierarchy (`src/combinator_hierarchy.cpp`)

Category theory-inspired combinators:

```cpp
template<typename A, typename B>
struct Arrow {
    function<B(A)> morphism;
    Arrow<B, C> compose(const Arrow<B, C>& other);
};
```

**Parser combinators**: `seq`, `alt`, `many`, `recursive`
**Graph combinators**: `constant_fold`, `dead_code_elimination`, `optimize`
**Cpp2 combinators**: `ufcs_call`, `contract`, `metafunction`

#### 4. MLIR Bridge (`src/mlir_bridge.cpp`)

Bidirectional conversion between CRDT graph and MLIR operations:

```cpp
// CRDT → MLIR
CRDTToMLIRConverter converter(ctx);
ModuleOp module = converter.convert(crdtGraph);

// MLIR → CRDT
MLIRToCRDTConverter reverseConverter;
CRDTGraph graph = reverseConverter.convert(module);
```

This eliminates custom serialization - all graph state is represented as MLIR operations.

#### 5. Global Code Motion Scheduler

Implements Chapter 11 algorithms:

1. **Schedule Early**: Upward DFS, place nodes at earliest dominated block
2. **Schedule Late**: Downward DFS, move to latest valid position in shallowest loop
3. **Anti-dependencies**: Insert Load→Store ordering constraints

```cpp
Scheduler scheduler(graph);
scheduler.schedule_early();
scheduler.schedule_late();
scheduler.insert_anti_dependencies();
```

### Transpiler Modes

#### Traditional AST Mode (default)

```bash
cppfort input.cpp2 output.cpp
```

Pipeline: Lexer → Parser → AST → Semantic → CodeGen

#### Sea of Nodes Mode

```bash
cppfort --son input.cpp2 output.cpp
```

Pipeline: Lexer → Parser → AST → SoN CRDT → MLIR → GCM → CodeGen

#### Pure SoN Mode

```bash
cppfort-son input.cpp2 output.cpp
```

Pipeline: Lexer → Tokens → SoN CRDT → MLIR → GCM → CodeGen

No AST construction - direct graph building from tokens.

### Memory Aliasing (Chapter 10)

Equivalence-class aliasing ensures:

```cpp
struct Vec2D { int x; int y; }
struct Vec3D { int x; int y; int z; }

// Distinct alias classes:
// Vec2D.x = 1, Vec2D.y = 2
// Vec3D.x = 3, Vec3D.y = 4, Vec3D.z = 5
```

Different alias classes never alias. Same class always aliases.
Represented directly in Sea of Nodes - no side structures needed.

### Type Lattice

```
         TOP
       /  |  \
    Int  Ptr  Struct
       \  |  /
       BOTTOM
```

Pointer types include nullable flag: `Ptr<Struct, null>`

### Optimizations

**Peephole (local)**:
- Constant folding: `add(const(2), const(3))` → `const(5)`
- Load after Store: `load(store(ptr, v), ptr)` → `v`
- Algebraic: `add(x, 0)` → `x`, `mul(x, 1)` → `x`

**Global Code Motion**:
- Loop-invariant code motion
- Common subexpression elimination via graph sharing
- Dead code elimination via reachability

### Distributed Development

Pijul CRDT enables:

1. **Parallel edits**: Multiple developers modify graph concurrently
2. **Automatic merge**: Patches commute without conflicts
3. **Incremental compilation**: Only changed subgraphs recompile
4. **Version control**: Native Pijul patch tracking

### Testing

```bash
ninja -C build test
```

Regression tests from hsutter/cppfront with SHA256 verification.

### Future Work

- Complete MLIR lowering to LLVM dialect
- JIT compilation via MLIR execution engine
- MLIR pass pipeline integration
- Subtyping and inheritance (Chapter 14+)
- Arrays with range checking (Chapter 15+)