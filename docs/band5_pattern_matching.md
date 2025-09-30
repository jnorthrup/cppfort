# Band 5: N-Way Pattern Matching Infrastructure (Chapters 16-18)

## Strategic Convergence Point

Band 5 represents the architectural fulcrum where cppfort diverges from Simple compiler's single-target approach to enable n-way meta-transpilation. This band leverages Band 4's type enum infrastructure to build a comprehensive pattern matching system for multi-target lowering.

## Architectural Vision

**Simple Compiler:** Sea of Nodes → Optimize → Emit C++
**Cppfort Stage0:** Sea of Nodes → Pattern Match → N-Way Lower → {C, C++, CPP2, MLIR, ...}

Band 5 implements the pattern matching layer that makes n-way lowering declarative, maintainable, and formally verifiable.

## Core Components

### 1. NodeKind Enum Classification

Building on Band 4's type enums (TypeFloat::Precision, TypeNarrow::Width), Band 5 introduces comprehensive node classification:

```cpp
// Band 4 established type classification via enums
enum class TypeKind {
    INTEGER,
    FLOAT,
    POINTER,
    NARROW,
    ARRAY,
    BOTTOM,
    TOP
};

// Band 5 extends this to NODE classification
enum class NodeKind {
    // Control Flow (Band 1-2)
    START,
    RETURN,
    IF,
    REGION,
    LOOP,
    CPROJ,
    STOP,

    // Data Operations (Band 1)
    CONSTANT,
    ADD,
    SUB,
    MUL,
    DIV,
    MINUS,

    // Variables (Band 1)
    PHI,
    SCOPE,

    // Memory Operations (Band 2)
    NEW,
    LOAD,
    STORE,
    PROJ,

    // Band 4: Type-specific operations
    FADD,    // Float add
    FSUB,    // Float subtract
    FMUL,    // Float multiply
    FDIV,    // Float divide

    // Band 4: Array operations
    NEW_ARRAY,
    ARRAY_LOAD,
    ARRAY_STORE,
    ARRAY_LENGTH,

    // Band 4: Type conversions
    CAST,

    // Band 5: Bitwise operations (Chapter 16)
    AND,     // Bitwise AND
    OR,      // Bitwise OR
    XOR,     // Bitwise XOR
    SHL,     // Shift left
    ASHR,    // Arithmetic shift right
    LSHR,    // Logical shift right

    // Band 5: Comparison operations
    EQ,      // Equal
    NE,      // Not equal
    LT,      // Less than
    LE,      // Less or equal
    GT,      // Greater than
    GE,      // Greater or equal

    // Band 5: Boolean operations
    BOOL_AND,  // Logical AND
    BOOL_OR,   // Logical OR
    BOOL_NOT   // Logical NOT
};
```

### 2. Pattern Matching Architecture

The pattern matching system operates via **subsumption queries** - a powerful abstraction that enables declarative multi-target lowering.

#### Subsumption Query Model

```cpp
// Query builder API - composable filters across band boundaries
class SubsumptionQuery {
public:
    // Type-based filtering (Band 4)
    SubsumptionQuery& whereType(TypePredicate pred);

    // Node kind filtering (Band 5)
    SubsumptionQuery& whereKind(NodeKind kind);
    SubsumptionQuery& whereKind(std::vector<NodeKind> kinds);

    // CFG-based filtering (Band 2-3)
    SubsumptionQuery& whereCFG(CFGPredicate pred);

    // Scheduling constraints (Band 3)
    SubsumptionQuery& whereSchedule(SchedulePredicate pred);

    // Memory constraints (Band 2)
    SubsumptionQuery& whereMemory(MemoryPredicate pred);

    // Execute query and return matching nodes
    std::vector<Node*> execute();
};
```

#### Example: MLIR Arithmetic Dialect Lowering

```cpp
// Find all integer arithmetic operations suitable for MLIR arith dialect
auto intArithOps = subsumption.query()
    .whereKind({NodeKind::ADD, NodeKind::SUB, NodeKind::MUL, NodeKind::DIV})
    .whereType([](Type* t) { return t->isInteger(); })
    .execute();

// Lower each to MLIR arith.addi, arith.subi, etc.
for (Node* op : intArithOps) {
    switch (op->kind()) {
        case NodeKind::ADD:  emit_mlir_arith_addi(op); break;
        case NodeKind::SUB:  emit_mlir_arith_subi(op); break;
        case NodeKind::MUL:  emit_mlir_arith_muli(op); break;
        case NodeKind::DIV:  emit_mlir_arith_divsi(op); break;
    }
}
```

### 3. TableGen N-Way Patterns

TableGen specifications define declarative patterns for n-way lowering:

```tablegen
// patterns/nway_lowering.td

// ============================================================================
// Integer Arithmetic Patterns
// ============================================================================

// Pattern: SON AddNode → Multiple targets
def SONAddInt : Pat<
    (SON_AddNode IntType:$lhs, IntType:$rhs),
    [
        // C target
        (C_BinOp "+" $lhs $rhs),

        // C++ target
        (CPP_BinOp "+" $lhs $rhs),

        // CPP2 target
        (CPP2_BinOp "+" $lhs $rhs),

        // MLIR target
        (MLIR_Arith_AddIOp $lhs $rhs)
    ]
>;

// Pattern: SON MulNode → Multiple targets
def SONMulInt : Pat<
    (SON_MulNode IntType:$lhs, IntType:$rhs),
    [
        (C_BinOp "*" $lhs $rhs),
        (CPP_BinOp "*" $lhs $rhs),
        (CPP2_BinOp "*" $lhs $rhs),
        (MLIR_Arith_MulIOp $lhs $rhs)
    ]
>;

// ============================================================================
// Float Arithmetic Patterns
// ============================================================================

// Pattern: SON AddNode with float type → Multiple targets
def SONAddFloat : Pat<
    (SON_AddNode FloatType:$lhs, FloatType:$rhs),
    [
        (C_BinOp "+" $lhs $rhs),
        (CPP_BinOp "+" $lhs $rhs),
        (CPP2_BinOp "+" $lhs $rhs),
        (MLIR_Arith_AddFOp $lhs $rhs)
    ]
>;

// ============================================================================
// Bitwise Operation Patterns (Chapter 16)
// ============================================================================

def SONBitAnd : Pat<
    (SON_AndNode IntType:$lhs, IntType:$rhs),
    [
        (C_BinOp "&" $lhs $rhs),
        (CPP_BinOp "&" $lhs $rhs),
        (CPP2_BitwiseAnd $lhs $rhs),
        (MLIR_Arith_AndIOp $lhs $rhs)
    ]
>;

def SONBitOr : Pat<
    (SON_OrNode IntType:$lhs, IntType:$rhs),
    [
        (C_BinOp "|" $lhs $rhs),
        (CPP_BinOp "|" $lhs $rhs),
        (CPP2_BitwiseOr $lhs $rhs),
        (MLIR_Arith_OrIOp $lhs $rhs)
    ]
>;

def SONShiftLeft : Pat<
    (SON_ShlNode IntType:$val, IntType:$shift),
    [
        (C_BinOp "<<" $val $shift),
        (CPP_BinOp "<<" $val $shift),
        (CPP2_ShiftLeft $val $shift),
        (MLIR_Arith_ShLIOp $val $shift)
    ]
>;

// ============================================================================
// Array Operation Patterns (Band 4 integration)
// ============================================================================

def SONArrayLoad : Pat<
    (SON_ArrayLoadNode MemType:$mem, ArrayType:$array, IntType:$index),
    [
        // C: array[index]
        (C_ArrayAccess $array $index),

        // C++: array[index] or vector.at(index)
        (CPP_ArrayAccess $array $index),

        // CPP2: array[index] with bounds checking
        (CPP2_ArrayAccess $array $index),

        // MLIR: memref.load %array[%index]
        (MLIR_MemRef_LoadOp $array $index)
    ]
>;

// ============================================================================
// Control Flow Patterns
// ============================================================================

def SONIfNode : Pat<
    (SON_IfNode BoolType:$cond, CFGNode:$true_branch, CFGNode:$false_branch),
    [
        // C: if (cond) { ... } else { ... }
        (C_IfStatement $cond $true_branch $false_branch),

        // C++: Same as C
        (CPP_IfStatement $cond $true_branch $false_branch),

        // CPP2: if (cond) { ... } else { ... }
        (CPP2_IfStatement $cond $true_branch $false_branch),

        // MLIR: scf.if %cond { ... } else { ... }
        (MLIR_SCF_IfOp $cond $true_branch $false_branch)
    ]
>;

def SONLoopNode : Pat<
    (SON_LoopNode CFGNode:$entry, PhiNode:$phi, CFGNode:$body),
    [
        // C: while (condition) { ... }
        (C_WhileLoop $entry $phi $body),

        // C++: Same as C
        (CPP_WhileLoop $entry $phi $body),

        // CPP2: while (condition) { ... }
        (CPP2_WhileLoop $entry $phi $body),

        // MLIR: scf.while (%phi) { ... }
        (MLIR_SCF_WhileOp $entry $phi $body)
    ]
>;
```

### 4. Enum-Based Pattern Dispatcher

The dispatcher uses NodeKind enums to route nodes to appropriate lowering patterns:

```cpp
class PatternDispatcher {
private:
    // Pattern tables indexed by NodeKind
    std::unordered_map<NodeKind, std::vector<Pattern>> _patterns;

    // Target-specific emitters
    std::unique_ptr<CEmitter> _c_emitter;
    std::unique_ptr<CPPEmitter> _cpp_emitter;
    std::unique_ptr<CPP2Emitter> _cpp2_emitter;
    std::unique_ptr<MLIREmitter> _mlir_emitter;

public:
    // Register pattern for specific node kind
    void registerPattern(NodeKind kind, Pattern pattern) {
        _patterns[kind].push_back(std::move(pattern));
    }

    // Dispatch node to appropriate pattern based on target
    void dispatch(Node* node, TargetLanguage target) {
        NodeKind kind = node->getKind();

        auto it = _patterns.find(kind);
        if (it == _patterns.end()) {
            throw std::runtime_error("No pattern for NodeKind");
        }

        // Find pattern matching constraints
        for (const Pattern& pat : it->second) {
            if (pat.matches(node)) {
                switch (target) {
                    case TargetLanguage::C:
                        pat.emitC(node, _c_emitter.get());
                        break;
                    case TargetLanguage::CPP:
                        pat.emitCPP(node, _cpp_emitter.get());
                        break;
                    case TargetLanguage::CPP2:
                        pat.emitCPP2(node, _cpp2_emitter.get());
                        break;
                    case TargetLanguage::MLIR:
                        pat.emitMLIR(node, _mlir_emitter.get());
                        break;
                }
                return;
            }
        }

        throw std::runtime_error("No matching pattern");
    }
};
```

### 5. Induction Over Node Categories

Enums enable systematic induction over node categories:

```cpp
// Category classification via enum ranges
class NodeCategory {
public:
    static bool isArithmetic(NodeKind kind) {
        return kind >= NodeKind::ADD && kind <= NodeKind::DIV;
    }

    static bool isFloatArithmetic(NodeKind kind) {
        return kind >= NodeKind::FADD && kind <= NodeKind::FDIV;
    }

    static bool isBitwise(NodeKind kind) {
        return kind >= NodeKind::AND && kind <= NodeKind::LSHR;
    }

    static bool isComparison(NodeKind kind) {
        return kind >= NodeKind::EQ && kind <= NodeKind::GE;
    }

    static bool isMemoryOp(NodeKind kind) {
        return kind >= NodeKind::NEW && kind <= NodeKind::PROJ;
    }

    static bool isCFG(NodeKind kind) {
        return kind >= NodeKind::START && kind <= NodeKind::STOP;
    }
};

// Inductive pattern matching over categories
void optimizeArithmeticOps(Graph* graph) {
    for (Node* node : graph->nodes()) {
        if (NodeCategory::isArithmetic(node->getKind())) {
            // Apply arithmetic-specific optimizations
            applyConstantFolding(node);
            applyAlgebraicSimplification(node);
        }
    }
}

// N-way lowering via category induction
void lowerToMLIR(Graph* graph, MLIRContext* ctx) {
    for (Node* node : graph->nodes()) {
        if (NodeCategory::isArithmetic(node->getKind())) {
            lowerToArithDialect(node, ctx);
        } else if (NodeCategory::isMemoryOp(node->getKind())) {
            lowerToMemRefDialect(node, ctx);
        } else if (NodeCategory::isCFG(node->getKind())) {
            lowerToSCFDialect(node, ctx);
        }
    }
}
```

## Chapter 16 Integration: Bitwise Operations

Chapter 16 introduces bitwise operations, which Band 5 adds to the node taxonomy:

### New Node Types

```cpp
// Bitwise operations - Chapter 16
class AndNode : public Node {
public:
    NodeKind getKind() const override { return NodeKind::AND; }
    std::string label() const override { return "&"; }
    Type* compute() override;
    Node* peephole() override;
};

class OrNode : public Node {
public:
    NodeKind getKind() const override { return NodeKind::OR; }
    std::string label() const override { return "|"; }
    Type* compute() override;
    Node* peephole() override;
};

class XorNode : public Node {
public:
    NodeKind getKind() const override { return NodeKind::XOR; }
    std::string label() const override { return "^"; }
    Type* compute() override;
    Node* peephole() override;
};

class ShlNode : public Node {
public:
    NodeKind getKind() const override { return NodeKind::SHL; }
    std::string label() const override { return "<<"; }
    Type* compute() override;
    Node* peephole() override;
};

class AShrNode : public Node {
public:
    NodeKind getKind() const override { return NodeKind::ASHR; }
    std::string label() const override { return ">>"; }
    Type* compute() override;
    Node* peephole() override;
};

class LShrNode : public Node {
public:
    NodeKind getKind() const override { return NodeKind::LSHR; }
    std::string label() const override { return ">>>"; }
    Type* compute() override;
    Node* peephole() override;
};
```

### Bitwise Peephole Optimizations

```cpp
// Peephole patterns for bitwise operations
Node* AndNode::peephole() {
    // x & 0 → 0
    if (in(1)->isConstant() && static_cast<ConstantNode*>(in(1))->value() == 0) {
        return in(1);
    }

    // x & -1 → x
    if (in(1)->isConstant() && static_cast<ConstantNode*>(in(1))->value() == -1) {
        return in(0);
    }

    // x & x → x
    if (in(0) == in(1)) {
        return in(0);
    }

    return this;
}

Node* OrNode::peephole() {
    // x | 0 → x
    if (in(1)->isConstant() && static_cast<ConstantNode*>(in(1))->value() == 0) {
        return in(0);
    }

    // x | -1 → -1
    if (in(1)->isConstant() && static_cast<ConstantNode*>(in(1))->value() == -1) {
        return in(1);
    }

    // x | x → x
    if (in(0) == in(1)) {
        return in(0);
    }

    return this;
}

Node* XorNode::peephole() {
    // x ^ 0 → x
    if (in(1)->isConstant() && static_cast<ConstantNode*>(in(1))->value() == 0) {
        return in(0);
    }

    // x ^ x → 0
    if (in(0) == in(1)) {
        return ConstantNode::make(0);
    }

    return this;
}
```

## Integration with Subsumption Engine

The subsumption engine enables cross-band queries using NodeKind enums:

```cpp
// Example: Find all operations that can be vectorized
auto vectorizable = subsumption.query()
    .whereKind({NodeKind::ADD, NodeKind::MUL, NodeKind::FADD, NodeKind::FMUL})
    .whereCFG([](Node* n) { return n->cfg0()->loopDepth() > 0; })  // Band 3
    .whereType([](Type* t) { return t->isNumeric(); })              // Band 4
    .whereMemory([](Node* n) { return !n->hasMemoryDep(); })        // Band 2
    .execute();

// Lower vectorizable ops to MLIR vector dialect
for (Node* op : vectorizable) {
    lowerToVectorDialect(op, mlir_ctx);
}
```

## N-Way Conversion Alpha

Enum-based pattern matching dramatically reduces maintenance burden:

### Imperative Approach (Without Enums)
```cpp
// Must manually check every node type
void lowerToC(Node* node) {
    if (auto* add = dynamic_cast<AddNode*>(node)) {
        // emit C code
    } else if (auto* sub = dynamic_cast<SubNode*>(node)) {
        // emit C code
    } else if (auto* mul = dynamic_cast<MulNode*>(node)) {
        // emit C code
    }
    // ... 50+ more types
}

// Repeat for C++, CPP2, MLIR, Rust, WASM...
// Total: 6 targets × 1000 LOC each = 6000 LOC
```

### Declarative Approach (With Enums + Patterns)
```tablegen
// Define patterns once
def SONAdd : Pat<
    (SON_AddNode $lhs, $rhs),
    [(C_Add $lhs $rhs), (CPP_Add $lhs $rhs), ...]
>;

// Pattern matcher applies declaratively
// Total: ~500 patterns × 10 LOC = 5000 LOC + 2000 LOC engine
//      = 7000 LOC (but declarative, verifiable, maintainable)
```

**Alpha calculation:**
- **Imperative:** 6000 LOC × N targets (grows linearly)
- **Declarative:** 7000 LOC total (fixed cost)
- **Crossover:** At 2+ targets, declarative wins
- **Maintenance:** 1 pattern update vs N function updates (6x reduction)

## Testing Strategy

Band 5 requires comprehensive pattern matching tests:

### Test Categories

1. **Enum Classification Tests**
   - Verify NodeKind enum covers all node types
   - Test category predicates (isArithmetic, isBitwise, etc.)
   - Validate enum-based dispatch

2. **Pattern Matching Tests**
   - Verify patterns match intended nodes
   - Test constraint checking (type, CFG, schedule)
   - Validate pattern priority/ordering

3. **N-Way Lowering Tests**
   - Generate C, C++, CPP2, MLIR from same graph
   - Verify semantic equivalence across targets
   - Test edge cases (nulls, overflows, etc.)

4. **Bitwise Operation Tests**
   - Verify peephole optimizations
   - Test constant folding for bitwise ops
   - Validate shift operations (arithmetic vs logical)

5. **Subsumption Query Tests**
   - Test cross-band queries
   - Verify query composition
   - Validate result correctness

### Example Test: N-Way Lowering

```cpp
TEST(Band5, NWayArithmeticLowering) {
    // Build simple arithmetic graph
    Graph graph;
    auto* start = graph.addNode<StartNode>();
    auto* a = graph.addNode<ConstantNode>(5);
    auto* b = graph.addNode<ConstantNode>(3);
    auto* add = graph.addNode<AddNode>(a, b);
    auto* ret = graph.addNode<ReturnNode>(start, add);

    // Lower to C
    CEmitter c_emit;
    c_emit.emit(&graph);
    EXPECT_EQ(c_emit.code(), "return 5 + 3;");

    // Lower to C++
    CPPEmitter cpp_emit;
    cpp_emit.emit(&graph);
    EXPECT_EQ(cpp_emit.code(), "return 5 + 3;");

    // Lower to MLIR
    MLIREmitter mlir_emit;
    mlir_emit.emit(&graph);
    EXPECT_THAT(mlir_emit.code(), HasSubstr("arith.addi"));
}
```

## Implementation Phases

### Phase 1: NodeKind Enum (Week 1)
- Add NodeKind enum to Node base class
- Implement getKind() for all existing nodes
- Add NodeCategory helper class
- Update Band 1-4 nodes with kind classification

### Phase 2: Pattern Infrastructure (Week 2)
- Implement SubsumptionQuery builder API
- Create Pattern base class
- Build PatternDispatcher
- Add basic pattern matching tests

### Phase 3: Chapter 16 Bitwise Ops (Week 3)
- Implement AndNode, OrNode, XorNode
- Add ShlNode, AShrNode, LShrNode
- Implement bitwise peephole optimizations
- Add parser support for bitwise operators
- Create bitwise operation tests

### Phase 4: TableGen N-Way Patterns (Week 4)
- Create nway_lowering.td specification
- Generate pattern matcher from TableGen
- Implement C/C++/CPP2/MLIR emitters
- Add n-way lowering tests

### Phase 5: Integration (Week 5)
- Integrate pattern matcher with existing pipeline
- Update CMakeLists.txt for TableGen generation
- Create comprehensive test suite
- Document pattern matching API

## Conclusion

Band 5 transforms cppfort from a simple compiler into a true meta-transpiler. By leveraging enums for pattern matching and TableGen for declarative lowering, we achieve:

1. **Maintainability:** Patterns defined once, applied to N targets
2. **Correctness:** Declarative patterns are formally verifiable
3. **Extensibility:** New targets added by defining new patterns
4. **Performance:** Enum-based dispatch is O(1)

With Bands 1-5 complete, stage0 becomes a production-capable meta-transpiler foundation capable of targeting multiple languages from a single Sea of Nodes IR.

**Next:** Band 6+ will add advanced optimizations (escape analysis, inlining, borrow checking) building on this pattern matching infrastructure.
