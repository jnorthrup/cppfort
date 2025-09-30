# Divergence: Sea of Nodes Simple vs Cppfort N-Way Conversion

**Critical Distinction:** Cppfort's goals diverge from Simple compiler's Sea of Nodes approach.

**Simple's Goal:** Sea of Nodes → Optimize → Emit single target
**Cppfort's Goal:** Sea of Nodes → Pattern-based N-way conversion → Multiple targets

This fundamentally changes the architecture.

---

## Simple Compiler: Pure Sea of Nodes Pipeline

### Architecture

```
Source Code
    ↓
Parse → Sea of Nodes IR
    ↓
Optimize (within SoN)
    ↓
Schedule (GCM)
    ↓
Emit C++ (single target)
```

**Philosophy:**
- Sea of Nodes is the optimization IR
- Stay in SoN as long as possible
- Single-target emission (C++)
- Optimization = graph transformations

**Subsumption Use Case:**
- Query for optimization opportunities
- Pattern matching for peephole rules
- Bulk transformations

---

## Cppfort: N-Way Conversion Architecture

### Architecture

```
CPP2 Source
    ↓
Parse → Sea of Nodes IR (source representation)
    ↓
Optimize (within SoN) ← Band 1-4
    ↓
Pattern-Based Lowering via TableGen ← KEY DIVERGENCE
    ├─→ MLIR Arithmetic Dialect
    ├─→ MLIR MemRef Dialect
    ├─→ MLIR Vector Dialect
    ├─→ MLIR GPU Dialect
    ├─→ C (direct emission)
    ├─→ C++ (direct emission)
    ├─→ Rust (via MLIR)
    └─→ WASM (via MLIR)
```

**Philosophy:**
- Sea of Nodes is **source IR** (input format)
- MLIR dialects are **target IRs** (output formats)
- N-way conversion = declarative pattern matching
- Subsumption = pattern matching engine

**Subsumption Use Case:**
- **PRIMARY:** Pattern-based lowering to multiple targets
- Secondary: Optimization queries

---

## The Divergence Point

### Simple Compiler Approach

**Goal:** Emit optimized C++ from Sea of Nodes.

**Process:**
1. Build Sea of Nodes graph
2. Optimize within graph (peepholes, GCM, etc.)
3. Schedule to concrete order (Band 3)
4. Emit C++ code directly

**No n-way conversion needed.**

### Cppfort Approach

**Goal:** Lower Sea of Nodes to multiple target representations.

**Process:**
1. Build Sea of Nodes graph (universal source IR)
2. Optimize within graph (Band 1-4)
3. **Pattern-match to target dialects** ← DIVERGENCE
4. Lower through MLIR pipeline
5. Emit to multiple targets

**N-way conversion is the core value.**

---

## Why N-Way Conversion Requires Subsumption

### The Problem: Imperative Lowering Doesn't Scale

**Imperative approach (doesn't work for N targets):**

```cpp
void lowerToMLIR(Graph* graph) {
    for (Node* n : graph->nodes()) {
        if (auto* add = dynamic_cast<AddNode*>(n)) {
            if (add->type()->isInteger()) {
                emit_mlir_arith_addi(add);
            } else if (add->type()->isFloat()) {
                emit_mlir_arith_addf(add);
            }
        } else if (auto* mul = dynamic_cast<MulNode*>(n)) {
            if (mul->type()->isInteger()) {
                emit_mlir_arith_muli(mul);
            } else if (mul->type()->isFloat()) {
                emit_mlir_arith_mulf(mul);
            }
        }
        // ... repeat for 100+ node types
    }
}

// Now repeat entire function for:
// - lowerToC()
// - lowerToCpp()
// - lowerToRust()
// - lowerToWASM()
// = 5 copies of 1000+ line lowering functions
```

**Problem:** Unmaintainable explosion of imperative lowering code.

### Declarative Pattern Matching (Scales to N Targets)

**TableGen pattern approach:**

```tablegen
// Define patterns once
def : Pat<
    (AddNode IntegerType:$lhs, IntegerType:$rhs),
    (MLIR_Arith_AddIOp $lhs, $rhs)
>;

def : Pat<
    (AddNode FloatType:$lhs, FloatType:$rhs),
    (MLIR_Arith_AddFOp $lhs, $rhs)
>;

def : Pat<
    (MulNode IntegerType:$lhs, IntegerType:$rhs),
    (MLIR_Arith_MulIOp $lhs, $rhs)
>;

// Pattern matcher applies these declaratively
```

**Subsumption engine role:**
- Match Sea of Nodes patterns
- Check subsumption constraints (type hierarchy, CFG legality)
- Apply rewrite rules
- Generate target IR

**Advantage:** Patterns are data, not code. Easier to generate, verify, maintain.

---

## Subsumption as Pattern Matching Engine

### Core Capability: Structural Pattern Matching

**Query:** "Find all Add(Load(ptr), Const) patterns for MLIR strength reduction"

```cpp
// Subsumption-based pattern matching
Pattern strengthReducePattern = engine.createPattern()
    .match<AddNode>()
    .whereInput(0, subsumes<LoadNode>())    // Structural constraint
    .whereInput(1, subsumes<ConstantNode>()) // Structural constraint
    .whereType(subsumes<IntegerType>())      // Type constraint
    .rewrite([](Match m) {
        // Lower to MLIR affine.load + affine.add
        return lower_to_affine_pattern(m);
    });

// Apply pattern to entire graph
auto matches = strengthReducePattern.apply(graph);
```

**This is TableGen-style pattern matching.**

### Integration with MLIR TableGen

**MLIR uses TableGen for dialect conversions:**

```tablegen
// MLIR's approach (TableGen)
def AddPattern : Pat<
    (TF_AddOp $lhs, $rhs),
    (TOSA_AddOp $lhs, $rhs)
>;
```

**Cppfort's approach (Subsumption + TableGen):**

```tablegen
// Define patterns for Sea of Nodes → MLIR
def SONAddToMLIRArith : Pat<
    (SONAddNode IntType:$lhs, IntType:$rhs),
    (Arith_AddIOp $lhs, $rhs)
>;

def SONLoopToMLIRSCF : Pat<
    (SONLoopNode ControlFlow:$init, ControlFlow:$body),
    (SCF_ForOp $init, $body)
>;
```

**Subsumption engine:** Implements the pattern matcher that evaluates these rules.

---

## N-Way Conversion Targets

### Target 1: MLIR Dialects

**Sea of Nodes → MLIR patterns:**

```cpp
// Arithmetic operations → arith dialect
SONArithmeticOp → MLIR::arith::AddIOp
SONArithmeticOp → MLIR::arith::MulFOp

// Memory operations → memref dialect
SONLoadNode → MLIR::memref::LoadOp
SONStoreNode → MLIR::memref::StoreOp

// Control flow → cf/scf dialects
SONIfNode → MLIR::cf::CondBranchOp
SONLoopNode → MLIR::scf::ForOp

// Vectorizable → vector dialect
SONAddNode (vectorizable) → MLIR::vector::AddOp
```

**Subsumption role:** Pattern match + constraint checking (vectorizable, etc.)

### Target 2: Direct C Emission

**Sea of Nodes → C patterns:**

```cpp
// Simple patterns
SONAddNode → "a + b"
SONMulNode → "a * b"
SONIfNode → "if (cond) { ... }"

// Complex patterns
SONLoopNode → "for (int i = 0; i < n; i++) { ... }"
SONLoadNode → "*ptr"
SONStoreNode → "*ptr = value"
```

**Subsumption role:** Match patterns, emit C syntax.

### Target 3: C++20 Modules (.cppm) - APPROPRIATE TARGET

**Sea of Nodes → C++20 modules patterns:**

```cpp
// Module structure
SONModuleNode → "export module foo;"
SONImportNode → "import bar;"

// Modern C++ patterns
SONNewNode → "std::make_unique<T>()"
SONArrayNode → "std::vector<T>" or "std::span<T>"
SONReferenceNode → "T&" or "T*"

// C++20 concepts integration
SONTypeConstraint → "template<std::integral T>"
```

**Why .cppm is appropriate:**
- Clean module semantics (no header inclusion mess)
- Explicit exports match Sea of Nodes module structure
- Better build performance (modules are cached)
- Natural target for CPP2 semantics
- Alignment with modern C++ evolution

**Legacy C++ emission** (for compatibility) also supported, but .cppm is the preferred modern target.

### Target 4: Rust (via MLIR)

**Sea of Nodes → MLIR → Rust:**

```
SONGraph → MLIR dialects → MLIR-to-Rust → Rust code
```

**Subsumption role:** First-stage lowering to MLIR.

### Target 5: WASM (via MLIR)

**Sea of Nodes → MLIR → WASM:**

```
SONGraph → MLIR dialects → MLIR-to-WASM → WASM binary
```

---

## Alpha Calculation: N-Way vs Imperative

### Imperative Lowering (Without Subsumption)

**Lines of code per target:**
- MLIR lowering: ~2,000 LOC
- C lowering: ~1,500 LOC
- C++ lowering: ~1,800 LOC
- Rust lowering: ~2,000 LOC
- WASM lowering: ~2,500 LOC

**Total imperative code: ~10,000 LOC**

**Maintenance:** Each Sea of Nodes change = update 5 lowering functions.

### Declarative Pattern Matching (With Subsumption)

**Pattern definitions:**
- Patterns: ~500 TableGen entries (~50 LOC each = 2,500 LOC)
- Subsumption engine: ~3,000 LOC (one-time cost)

**Total declarative code: ~5,500 LOC**

**Maintenance:** Each Sea of Nodes change = update pattern file once.

**Alpha:**
- Code reduction: 10,000 → 5,500 = **45% less code**
- Maintenance: 5 updates → 1 update = **80% less maintenance**
- Correctness: Declarative = formally verifiable

---

## Simple Compiler Doesn't Need This

### Why Simple Uses Imperative Lowering

**Simple's goal:** Emit C++ only.

```cpp
void emit_cpp(Node* n) {
    if (auto* add = dynamic_cast<AddNode*>(n)) {
        emit("a + b");
    } else if (auto* mul = dynamic_cast<MulNode*>(n)) {
        emit("a * b");
    }
    // ... ~500 LOC for single target
}
```

**This is fine for one target.**

**No need for:**
- Pattern matching engine
- Subsumption constraints
- TableGen integration
- Multi-target lowering

**Simple's approach is simpler because it's single-target.**

---

## Cppfort's Divergence Justified

### Why Cppfort Needs N-Way Conversion

**Goal:** Universal meta-transpiler (stage0).

**Use cases:**
1. CPP2 → C (for legacy integration)
2. CPP2 → C++ (for cppfront compatibility)
3. CPP2 → Rust (for safety-critical systems)
4. CPP2 → WASM (for web deployment)
5. CPP2 → MLIR → GPU (for acceleration)

**Each target has different semantics:**
- C: No references, manual memory
- C++: References, RAII, templates
- Rust: Ownership, lifetimes, traits
- WASM: Linear memory, stack machine
- GPU: Parallel execution, shared memory

**Imperative lowering doesn't scale to these semantic differences.**

### Subsumption Enables Semantic Constraints

**Example: Rust ownership lowering**

```tablegen
// Pattern: Sea of Nodes owned value → Rust owned value
def SONOwnedToRust : Pat<
    (SONNewNode $type),
    (RustBox $type),
    [(IsOwned $type),          // Constraint: must be owned
     (NoAliasing $type)]       // Constraint: no aliasing
>;

// Pattern: Sea of Nodes borrowed value → Rust borrow
def SONBorrowToRust : Pat<
    (SONReferenceNode $referent),
    (RustBorrow $referent),
    [(IsBorrowed $referent),   // Constraint: must be borrow
     (ValidLifetime $referent)] // Constraint: lifetime valid
>;
```

**Subsumption engine:** Checks semantic constraints before applying patterns.

**Without subsumption:** Must manually check constraints in imperative code (error-prone).

---

## Architecture: Subsumption as Lowering Engine

### Design

```
┌─────────────────────────────────────────┐
│       Sea of Nodes Source IR            │
│  (AddNode, MulNode, LoadNode, etc.)     │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│     Subsumption Pattern Matcher         │
│  - Match structural patterns            │
│  - Check subsumption constraints        │
│  - Apply rewrite rules                  │
└──────────────┬──────────────────────────┘
               │
               ├─────────────────┐
               │                 │
               ▼                 ▼
    ┌──────────────────┐  ┌──────────────────┐
    │ MLIR Dialects    │  │ Direct Emission  │
    │ - arith          │  │ - C              │
    │ - memref         │  │ - C++            │
    │ - scf            │  │                  │
    │ - vector         │  │                  │
    └────────┬─────────┘  └──────────────────┘
             │
             ▼
    ┌──────────────────┐
    │ MLIR Lowering    │
    │ - LLVM IR        │
    │ - Rust           │
    │ - WASM           │
    │ - GPU Kernels    │
    └──────────────────┘
```

### Subsumption Role

**Primary:** Pattern matching for n-way lowering
**Secondary:** Optimization queries

**Key operations:**
1. Structural pattern matching (graph shapes)
2. Constraint checking (type subsumption, CFG legality, lifetime validity)
3. Rewrite rule application (lower to target IR)
4. Multi-target dispatch (choose lowering strategy per target)

---

## Comparison Table

| Aspect | Simple Compiler | Cppfort |
|--------|----------------|---------|
| **Goal** | Optimize + emit C++ | N-way conversion to multiple targets |
| **Sea of Nodes** | Optimization IR | Source IR |
| **Target count** | 1 (C++) | 5+ (C, C++, Rust, WASM, GPU) |
| **Lowering** | Imperative | Declarative (TableGen patterns) |
| **Subsumption** | Optional (optimization only) | Essential (pattern matching) |
| **MLIR** | Not needed | Core integration |
| **Code size** | ~500 LOC (single target) | ~5,500 LOC (N targets) |
| **Maintenance** | Simple (one target) | Complex (N targets, but declarative) |

---

## Implementation Strategy

### Phase 1: Build Pattern Matcher

```cpp
class PatternMatcher {
    struct Pattern {
        std::function<bool(Node*)> match;
        std::vector<Constraint> constraints;
        std::function<TargetIR(Node*)> rewrite;
    };

    std::vector<Pattern> _patterns;

    void addPattern(Pattern p) {
        _patterns.push_back(p);
    }

    std::optional<TargetIR> matchAndRewrite(Node* n) {
        for (auto& p : _patterns) {
            if (p.match(n) && checkConstraints(n, p.constraints)) {
                return p.rewrite(n);
            }
        }
        return std::nullopt;
    }
};
```

### Phase 2: Define TableGen Patterns

```tablegen
// patterns.td
include "mlir/IR/OpBase.td"
include "son/IR/SONOps.td"

def SONAddToArithAdd : Pat<
    (SON_AddNode IntType:$lhs, IntType:$rhs),
    (Arith_AddIOp $lhs, $rhs)
>;

def SONMulToArithMul : Pat<
    (SON_MulNode IntType:$lhs, IntType:$rhs),
    (Arith_MulIOp $lhs, $rhs)
>;
```

### Phase 3: Generate Pattern Matcher from TableGen

```bash
# Generate C++ from TableGen
tablegen -gen-pattern-matcher patterns.td -o patterns_generated.cpp
```

### Phase 4: Integrate with Subsumption Engine

```cpp
class SubsumptionEngine {
    PatternMatcher _matcher;

    void lowerToMLIR(Graph* graph, MLIRContext* ctx) {
        for (Node* n : graph->nodes()) {
            if (auto mlir_op = _matcher.matchAndRewrite(n)) {
                emit_mlir(mlir_op, ctx);
            }
        }
    }
};
```

---

## Why This Divergence Matters

### Simple Compiler Philosophy

**Stay in Sea of Nodes, optimize aggressively, emit once.**

- No multi-target complexity
- No pattern matching overhead
- No TableGen integration
- Simpler codebase

**This is correct for Simple's goals (educational compiler).**

### Cppfort Philosophy

**Use Sea of Nodes as universal source IR, lower to multiple targets declaratively.**

- Multi-target is core goal (not nice-to-have)
- Pattern matching is essential (not optional)
- TableGen enables maintainability
- MLIR integration is requirement

**This is correct for cppfort's goals (production meta-transpiler).**

---

## Conclusion

**Cppfort diverges from Simple at the lowering stage.**

**Simple:**
```
Sea of Nodes → Optimize → Emit C++
```

**Cppfort:**
```
Sea of Nodes → Pattern Match → N-Way Lower → Multiple Targets
```

**Subsumption engine's PRIMARY role:** Enable declarative pattern-based n-way lowering.

**This is not Simple compiler with bells and whistles - this is fundamentally different architecture for fundamentally different goals.**

**The divergence is justified by multi-target requirements that Simple doesn't have.**

**Alpha of subsumption:** Not just query speedup (10-100x), but **enabling n-way lowering that scales** (45% less code, 80% less maintenance).

Without subsumption, n-way lowering becomes unmaintainable imperative code explosion. With subsumption, it's declarative patterns that TableGen can generate and verify.

**This is why cppfort needs subsumption and Simple doesn't.**
