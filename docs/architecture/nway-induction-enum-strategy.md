# N-Way Induction via Enum-Based Pattern Matching

**Status:** Band 5 Core Architecture
**Context:** Builds on Band 4 type enum infrastructure

## The Enum Upgrade Strategy

Band 4 introduced type classification via enums:
- `TypeFloat::Precision` (F32, F64)
- `TypeNarrow::Width` (I8, I16, I32, I64, U8, U16, U32)
- `TypePointer` (nullable/non-nullable states)
- `TypeArray` (fixed/dynamic size)

Band 5 extends this pattern to **node classification**, enabling systematic induction over the entire Sea of Nodes graph for n-way lowering.

## Why Enums Enable N-Way Induction

### The Problem: Dynamic Dispatch Doesn't Scale

Traditional C++ polymorphism using virtual functions and dynamic_cast:

```cpp
// Imperative approach - requires N copies for N targets
void emitC(Node* node) {
    if (auto* add = dynamic_cast<AddNode*>(node)) {
        out << lhs << " + " << rhs;
    } else if (auto* sub = dynamic_cast<SubNode*>(node)) {
        out << lhs << " - " << rhs;
    } // ... 50+ more cases
}

void emitCPP(Node* node) {
    // Duplicate entire function
}

void emitMLIR(Node* node) {
    // Duplicate entire function again
}
```

**Problem:** O(N × M) code where N = targets, M = node types

### The Solution: Enum-Based Dispatch

```cpp
enum class NodeKind {
    ADD, SUB, MUL, DIV,
    FADD, FSUB, FMUL, FDIV,
    // ... all node types
};

// Each node exposes its kind
class Node {
    virtual NodeKind getKind() const = 0;
};

// Pattern matcher operates on kinds
class Pattern {
    bool matches(Node* node) const {
        return _kinds.contains(node->getKind());
    }
};
```

**Advantage:** O(1) dispatch, data-driven pattern matching

## Induction: The Core Mechanism

**Induction** means systematically iterating over a classification to apply transformations. Enums make this explicit and type-safe.

### Mathematical Induction Over Nodes

Traditional mathematical induction:
```
Base case: P(0) holds
Inductive step: P(k) → P(k+1)
Conclusion: P(n) holds for all n
```

Node kind induction:
```
Base case: Pattern matches NodeKind::ADD
Inductive extension: Pattern extends to all arithmetic kinds
Conclusion: Pattern applies to {ADD, SUB, MUL, DIV, ...}
```

### Enum Induction in Practice

```cpp
// Define category as enum range
enum class NodeKind {
    // Arithmetic operations (contiguous range)
    ARITH_START = 100,
    ADD = ARITH_START,
    SUB,
    MUL,
    DIV,
    ARITH_END,

    // Bitwise operations (contiguous range)
    BITWISE_START = 200,
    AND = BITWISE_START,
    OR,
    XOR,
    SHL,
    ASHR,
    LSHR,
    BITWISE_END,

    // Float operations (contiguous range)
    FLOAT_START = 300,
    FADD = FLOAT_START,
    FSUB,
    FMUL,
    FDIV,
    FLOAT_END
};

// Induction via range checking
bool isArithmetic(NodeKind k) {
    return k > NodeKind::ARITH_START && k < NodeKind::ARITH_END;
}

bool isBitwise(NodeKind k) {
    return k > NodeKind::BITWISE_START && k < NodeKind::BITWISE_END;
}

bool isFloatOp(NodeKind k) {
    return k > NodeKind::FLOAT_START && k < NodeKind::FLOAT_END;
}

// Apply transformation inductively
void optimizeArithmetic(Graph* g) {
    for (Node* n : g->nodes()) {
        if (isArithmetic(n->getKind())) {
            // Transformation applies to ALL arithmetic ops
            applyAlgebraicSimplification(n);
        }
    }
}
```

## N-Way Lowering via Induction

Enums enable declarative n-way lowering through inductive pattern specification:

### Pattern Specification (TableGen)

```tablegen
// Define inductive pattern over arithmetic operations
class ArithmeticPattern<NodeKind kind, string op> : Pat<
    (SON_Node kind, $lhs, $rhs),
    [
        // Inductively apply to all targets
        (C_BinOp op, $lhs, $rhs),
        (CPP_BinOp op, $lhs, $rhs),
        (CPP2_BinOp op, $lhs, $rhs),
        (MLIR_Arith_Op op, $lhs, $rhs)
    ]
>;

// Instantiate pattern for each arithmetic kind
def : ArithmeticPattern<NodeKind::ADD, "+">;
def : ArithmeticPattern<NodeKind::SUB, "-">;
def : ArithmeticPattern<NodeKind::MUL, "*">;
def : ArithmeticPattern<NodeKind::DIV, "/">;

// Generated code handles all arithmetic × all targets
```

### Pattern Matcher Implementation

```cpp
class InductivePatternMatcher {
private:
    // Pattern table indexed by kind
    std::unordered_map<NodeKind, std::vector<Pattern>> _patterns;

public:
    // Register pattern for specific kind
    void registerPattern(NodeKind kind, Pattern pattern) {
        _patterns[kind].push_back(std::move(pattern));
    }

    // Register pattern for range (inductive)
    void registerRangePattern(NodeKind start, NodeKind end, Pattern pattern) {
        for (int k = static_cast<int>(start); k < static_cast<int>(end); ++k) {
            _patterns[static_cast<NodeKind>(k)].push_back(pattern);
        }
    }

    // Match and apply pattern inductively
    bool matchAndApply(Node* node, Target target) {
        auto it = _patterns.find(node->getKind());
        if (it == _patterns.end()) return false;

        for (const Pattern& pat : it->second) {
            if (pat.matches(node)) {
                pat.apply(node, target);
                return true;
            }
        }
        return false;
    }
};
```

## Enum Induction Expressions

**Induction expressions** are queries that leverage enum classification to select nodes:

### Basic Induction Expression

```cpp
// Select all arithmetic operations
auto arithOps = graph.selectWhere([](Node* n) {
    return isArithmetic(n->getKind());
});

// Apply transformation inductively
for (Node* op : arithOps) {
    optimize(op);
}
```

### Composed Induction Expression

```cpp
// Select arithmetic ops in loops with numeric types
auto candidates = subsumption.query()
    .whereKind([](NodeKind k) { return isArithmetic(k); })
    .whereCFG([](Node* n) { return n->cfg0()->loopDepth() > 0; })
    .whereType([](Type* t) { return t->isNumeric(); })
    .execute();
```

### Induction with Type Refinement

```cpp
// Leverage Band 4 type enums + Band 5 node enums
auto floatArith = subsumption.query()
    .whereKind([](NodeKind k) { return isArithmetic(k); })
    .whereType([](Type* t) {
        if (auto* ft = dynamic_cast<TypeFloat*>(t)) {
            return ft->precision() == TypeFloat::F32;
        }
        return false;
    })
    .execute();

// Lower to MLIR f32 arithmetic dialect
for (Node* op : floatArith) {
    lowerToMLIRArithF32(op);
}
```

## Cross-Band Induction

Enums enable induction **across band boundaries** (key insight from subsumption architecture):

### Example: Vectorization Induction

```cpp
// Induction spans Bands 2, 3, 4, 5
auto vectorizable = subsumption.query()
    // Band 5: Node kind (arithmetic or float)
    .whereKind([](NodeKind k) {
        return isArithmetic(k) || isFloatOp(k);
    })
    // Band 3: Scheduling (sequential in loop)
    .whereSchedule([](Node* n) {
        return n->isScheduledSequentially();
    })
    // Band 4: Type width (compatible SIMD width)
    .whereType([](Type* t) {
        if (auto* nt = dynamic_cast<TypeNarrow*>(t)) {
            return nt->bitWidth() == 32;  // SSE/NEON compatible
        }
        return false;
    })
    // Band 2: Memory (no aliasing)
    .whereMemory([](Node* n) {
        return !n->hasMemoryDependencies();
    })
    .execute();

// Lower to MLIR vector dialect
lowerToVectorDialect(vectorizable, mlir_ctx);
```

**Key insight:** Single induction query spans 4 bands, enabled by enum-based classification at each level.

## Enum-Based Subsumption Queries

The subsumption engine uses enums to create efficient query indexes:

### Query Index Structure

```cpp
class SubsumptionEngine {
private:
    // Primary index: NodeKind → Nodes
    std::unordered_map<NodeKind, std::vector<Node*>> _kind_index;

    // Secondary index: TypeKind → Nodes
    std::unordered_map<TypeKind, std::vector<Node*>> _type_index;

    // Tertiary index: ScheduleKind → Nodes
    std::unordered_map<ScheduleKind, std::vector<Node*>> _schedule_index;

public:
    // Build indexes (O(N) where N = nodes)
    void buildIndexes(Graph* graph) {
        for (Node* n : graph->nodes()) {
            _kind_index[n->getKind()].push_back(n);
            _type_index[n->type()->getKind()].push_back(n);
            _schedule_index[n->getScheduleKind()].push_back(n);
        }
    }

    // Query by kind (O(1) lookup + O(M) filter where M = matches)
    std::vector<Node*> queryKind(NodeKind kind) {
        return _kind_index[kind];
    }

    // Inductive query over kind range (O(R) where R = range size)
    std::vector<Node*> queryKindRange(NodeKind start, NodeKind end) {
        std::vector<Node*> result;
        for (int k = static_cast<int>(start); k < static_cast<int>(end); ++k) {
            auto& nodes = _kind_index[static_cast<NodeKind>(k)];
            result.insert(result.end(), nodes.begin(), nodes.end());
        }
        return result;
    }
};
```

### Query Performance

**Without enums (dynamic_cast):**
- O(N) scan of entire graph per query
- O(N × M) for M queries
- No caching possible (dynamic_cast is opaque)

**With enum indexes:**
- O(1) lookup per kind
- O(R) for range queries where R = range size
- O(log N) for sorted index queries
- Indexes built once, queried many times

**Alpha:** 100-1000x speedup for repeated queries

## Pattern Language for N-Way Induction

The pattern language leverages enums to express inductive rules:

### Pattern Syntax (Conceptual)

```
Pattern := Match × Constraint × Rewrite
Match := NodeKind | NodeKindRange | NodeKindSet
Constraint := TypeConstraint ∧ CFGConstraint ∧ ScheduleConstraint
Rewrite := Target → Code
```

### Example: Inductive Strength Reduction

```tablegen
// Pattern: Multiply by power of 2 → Shift left
def MulToPowerOf2ToShift : Pat<
    // Match: Any arithmetic operation
    (SON_Node (isArithmetic $kind), $lhs, $rhs),
    // Constraint: RHS is power of 2 constant
    [(isPowerOf2Constant $rhs)],
    // Rewrite inductively to all targets
    [
        (C_Shift "<<" $lhs (log2 $rhs)),
        (CPP_Shift "<<" $lhs (log2 $rhs)),
        (CPP2_ShiftLeft $lhs (log2 $rhs)),
        (MLIR_Arith_ShLI $lhs (log2 $rhs))
    ]
>;
```

### Generated Pattern Matcher

```cpp
// Generated from TableGen pattern
class MulToPowerOf2ToShiftPattern : public Pattern {
public:
    bool matches(Node* node) override {
        // Check node kind (inductive over arithmetic ops)
        if (!isArithmetic(node->getKind())) return false;

        // Check constraint
        Node* rhs = node->in(1);
        if (!rhs->isConstant()) return false;

        long val = static_cast<ConstantNode*>(rhs)->value();
        return isPowerOf2(val);
    }

    void applyC(Node* node, CEmitter* emit) override {
        long val = static_cast<ConstantNode*>(node->in(1))->value();
        int shift = log2(val);
        emit->emit(node->in(0));
        emit->emit(" << ");
        emit->emit(shift);
    }

    void applyCPP(Node* node, CPPEmitter* emit) override {
        // Same as C
        applyC(node, emit);
    }

    void applyMLIR(Node* node, MLIREmitter* emit) override {
        long val = static_cast<ConstantNode*>(node->in(1))->value();
        int shift = log2(val);
        emit->emit("arith.shli ");
        emit->emit(node->in(0));
        emit->emit(", ");
        emit->emit(shift);
    }
};
```

## Integration with Band 4 Type Enums

Band 5 node enums complement Band 4 type enums for precise pattern matching:

### Two-Dimensional Pattern Space

```
                NodeKind (Band 5)
                ↓
        ADD   SUB   MUL   DIV   FADD  FSUB  ...
      ┌─────┬─────┬─────┬─────┬─────┬─────┬
  INT │  ✓  │  ✓  │  ✓  │  ✓  │  ✗  │  ✗  │
      ├─────┼─────┼─────┼─────┼─────┼─────┤
  F32 │  ✗  │  ✗  │  ✗  │  ✗  │  ✓  │  ✓  │ ← TypeKind
      ├─────┼─────┼─────┼─────┼─────┼─────┤    (Band 4)
  F64 │  ✗  │  ✗  │  ✗  │  ✗  │  ✓  │  ✓  │
      ├─────┼─────┼─────┼─────┼─────┼─────┤
  PTR │  ✗  │  ✗  │  ✗  │  ✗  │  ✗  │  ✗  │
      └─────┴─────┴─────┴─────┴─────┴─────┘
```

### Precise Pattern Matching

```cpp
// Pattern: Integer ADD → arith.addi
if (node->getKind() == NodeKind::ADD &&
    node->type()->getKind() == TypeKind::INTEGER) {
    emit_mlir_arith_addi(node);
}

// Pattern: Float ADD (F32) → arith.addf (f32)
if (node->getKind() == NodeKind::FADD &&
    node->type()->getKind() == TypeKind::FLOAT) {
    auto* ft = static_cast<TypeFloat*>(node->type());
    if (ft->precision() == TypeFloat::F32) {
        emit_mlir_arith_addf_f32(node);
    }
}
```

### Inductive Pattern Over Two Dimensions

```cpp
// Inductively match all numeric arithmetic
auto numericArith = subsumption.query()
    .whereKind([](NodeKind k) {
        return isArithmetic(k) || isFloatOp(k);
    })
    .whereType([](TypeKind tk) {
        return tk == TypeKind::INTEGER ||
               tk == TypeKind::FLOAT ||
               tk == TypeKind::NARROW;
    })
    .execute();
```

## Implementation Strategy

### Phase 1: Define NodeKind Enum

```cpp
// src/stage0/node.h
enum class NodeKind : uint16_t {
    // CFG nodes (0-99)
    START = 0,
    STOP = 1,
    RETURN = 2,
    IF = 3,
    REGION = 4,
    LOOP = 5,
    CPROJ = 6,

    // Data nodes (100-199)
    CONSTANT = 100,
    PHI = 101,
    PROJ = 102,

    // Arithmetic (200-299)
    ARITH_START = 200,
    ADD = 200,
    SUB = 201,
    MUL = 202,
    DIV = 203,
    MINUS = 204,
    ARITH_END = 299,

    // Bitwise (300-399)
    BITWISE_START = 300,
    AND = 300,
    OR = 301,
    XOR = 302,
    SHL = 303,
    ASHR = 304,
    LSHR = 305,
    BITWISE_END = 399,

    // Float arithmetic (400-499)
    FLOAT_START = 400,
    FADD = 400,
    FSUB = 401,
    FMUL = 402,
    FDIV = 403,
    FLOAT_END = 499,

    // Memory (500-599)
    MEMORY_START = 500,
    NEW = 500,
    LOAD = 501,
    STORE = 502,
    NEW_ARRAY = 503,
    ARRAY_LOAD = 504,
    ARRAY_STORE = 505,
    ARRAY_LENGTH = 506,
    MEMORY_END = 599,

    // Type conversions (600-699)
    CAST = 600,

    // Comparisons (700-799)
    CMP_START = 700,
    EQ = 700,
    NE = 701,
    LT = 702,
    LE = 703,
    GT = 704,
    GE = 705,
    CMP_END = 799,

    // Boolean ops (800-899)
    BOOL_START = 800,
    BOOL_AND = 800,
    BOOL_OR = 801,
    BOOL_NOT = 802,
    BOOL_END = 899
};
```

### Phase 2: Add getKind() to Node Hierarchy

```cpp
class Node {
public:
    virtual NodeKind getKind() const = 0;
    // ...
};

class AddNode : public Node {
public:
    NodeKind getKind() const override { return NodeKind::ADD; }
    // ...
};
```

### Phase 3: Implement Category Predicates

```cpp
// src/stage0/node_category.h
namespace cppfort::ir {

class NodeCategory {
public:
    static constexpr bool isArithmetic(NodeKind k) {
        return k > NodeKind::ARITH_START && k < NodeKind::ARITH_END;
    }

    static constexpr bool isBitwise(NodeKind k) {
        return k > NodeKind::BITWISE_START && k < NodeKind::BITWISE_END;
    }

    static constexpr bool isFloatOp(NodeKind k) {
        return k > NodeKind::FLOAT_START && k < NodeKind::FLOAT_END;
    }

    static constexpr bool isMemoryOp(NodeKind k) {
        return k > NodeKind::MEMORY_START && k < NodeKind::MEMORY_END;
    }

    static constexpr bool isCFG(NodeKind k) {
        return k >= NodeKind::START && k <= NodeKind::CPROJ;
    }

    static constexpr bool isComparison(NodeKind k) {
        return k > NodeKind::CMP_START && k < NodeKind::CMP_END;
    }

    static constexpr bool isBoolOp(NodeKind k) {
        return k > NodeKind::BOOL_START && k < NodeKind::BOOL_END;
    }
};

} // namespace cppfort::ir
```

### Phase 4: Build Subsumption Indexes

```cpp
// src/stage0/subsumption_engine.cpp
void SubsumptionEngine::buildIndexes(Graph* graph) {
    _kind_index.clear();
    _type_index.clear();

    for (Node* node : graph->nodes()) {
        // Index by node kind
        _kind_index[node->getKind()].push_back(node);

        // Index by type kind
        if (node->type()) {
            _type_index[node->type()->getKind()].push_back(node);
        }
    }
}
```

### Phase 5: Implement Pattern Matcher

```cpp
// src/stage0/pattern_matcher.cpp
bool PatternMatcher::match(Node* node, Target target) {
    // Lookup patterns by kind (O(1))
    auto it = _patterns.find(node->getKind());
    if (it == _patterns.end()) return false;

    // Try each pattern
    for (const Pattern& pat : it->second) {
        if (pat.matches(node)) {
            pat.apply(node, target);
            return true;
        }
    }

    return false;
}
```

## Testing Enum Induction

```cpp
TEST(Band5, EnumInductionBasic) {
    // Test arithmetic category
    EXPECT_TRUE(isArithmetic(NodeKind::ADD));
    EXPECT_TRUE(isArithmetic(NodeKind::SUB));
    EXPECT_FALSE(isArithmetic(NodeKind::AND));
}

TEST(Band5, EnumInductionQuery) {
    Graph graph;
    auto* a = graph.add<AddNode>();
    auto* b = graph.add<SubNode>();
    auto* c = graph.add<AndNode>();

    // Query arithmetic ops
    auto arith = queryKind([](NodeKind k) { return isArithmetic(k); });
    EXPECT_EQ(arith.size(), 2);  // a, b
}

TEST(Band5, EnumInductionPattern) {
    Graph graph;
    auto* add = graph.add<AddNode>(makeInt(5), makeInt(3));

    // Pattern should match ADD
    Pattern pat(NodeKind::ADD, [](Node* n, Target t) {
        // emit code
    });

    EXPECT_TRUE(pat.matches(add));
}
```

## Conclusion

Enum-based induction provides the foundation for n-way meta-transpilation:

1. **Classification:** NodeKind enum classifies all nodes systematically
2. **Induction:** Range predicates enable inductive pattern matching
3. **Subsumption:** Enum indexes enable O(1) query performance
4. **N-Way:** Single pattern definition applies to multiple targets
5. **Maintainability:** Declarative patterns, not imperative code explosion

Band 5's enum upgrade transforms cppfort from a simple compiler into a production meta-transpiler capable of targeting multiple languages from a unified Sea of Nodes IR.
