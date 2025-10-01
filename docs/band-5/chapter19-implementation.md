# Chapter 19 Implementation: Sea of Nodes IR Foundation

## Overview

Chapter 19 establishes the foundational **Sea of Nodes** intermediate representation (IR) that serves as the unifying graph language for n-way transpilation. This IR enables perfect move/copy elision and lifetime rescoping by representing program semantics as a directed acyclic graph (DAG) where data flow and control flow are explicitly separated.

## Core Principles

### 1. Sea of Nodes Architecture

Unlike traditional CFG-based IRs, Sea of Nodes represents:
- **Data dependencies** as directed edges between value-producing nodes
- **Control dependencies** as a separate dimension orthogonal to data flow
- **Memory effects** explicitly tracked through state edges

**Key Insight:** By separating data and control flow, the IR exposes optimization opportunities invisible to traditional representations.

### 2. Node Classification Hierarchy

Building on Band 4's type enum infrastructure, Chapter 19 introduces comprehensive node classification:

```cpp
enum class NodeKind : uint16_t {
    // Control Flow Nodes (0-99)
    CFG_START = 0,
    START = 0,          // Program entry point
    STOP = 1,           // Program exit
    RETURN = 2,         // Function return
    IF = 3,             // Conditional branch
    REGION = 4,         // Control flow merge
    LOOP = 5,           // Loop header
    CFG_END = 99,

    // Data Flow Nodes (100-199)
    DATA_START = 100,
    CONSTANT = 100,     // Literal values
    PARAMETER = 101,    // Function parameters
    PHI = 102,          // SSA phi node
    PROJECTION = 103,   // Tuple projection
    DATA_END = 199,

    // Arithmetic Nodes (200-299)
    ARITH_START = 200,
    ADD = 200,          // Integer/float addition
    SUB = 201,          // Subtraction
    MUL = 202,          // Multiplication
    DIV = 203,          // Division
    MOD = 204,          // Modulo
    NEG = 205,          // Negation
    ARITH_END = 299,

    // Memory Nodes (300-399)
    MEM_START = 300,
    LOAD = 300,         // Memory read
    STORE = 301,        // Memory write
    ALLOC = 302,        // Allocation
    MEMBAR = 303,       // Memory barrier
    MEM_END = 399,

    // Comparison Nodes (400-499)
    CMP_START = 400,
    EQ = 400,           // Equal
    NE = 401,           // Not equal
    LT = 402,           // Less than
    LE = 403,           // Less or equal
    GT = 404,           // Greater than
    GE = 405,           // Greater or equal
    CMP_END = 499,
};
```

### 3. Node Base Class Design

The `Node` class provides the foundation for all IR nodes:

```cpp
class Node {
protected:
    NodeKind _kind;
    uint32_t _nodeId;           // Unique identifier
    Type* _type;                // Result type
    std::vector<Node*> _inputs; // Input edges
    std::vector<Node*> _outputs;// Output edges (uses)
    
public:
    // Core interface
    NodeKind kind() const { return _kind; }
    Type* type() const { return _type; }
    
    // Graph manipulation
    Node* in(size_t i) const { return _inputs[i]; }
    void setIn(size_t i, Node* n);
    void addInput(Node* n);
    void replaceWith(Node* replacement);
    
    // Category queries (using enum ranges)
    bool isCFG() const { 
        return _kind >= NodeKind::CFG_START && _kind <= NodeKind::CFG_END; 
    }
    bool isArithmetic() const {
        return _kind >= NodeKind::ARITH_START && _kind <= NodeKind::ARITH_END;
    }
    bool isMemory() const {
        return _kind >= NodeKind::MEM_START && _kind <= NodeKind::MEM_END;
    }
    
    // Virtual dispatch for language-specific operations
    virtual std::string emitC() const = 0;
    virtual std::string emitCpp() const = 0;
    virtual std::string emitCpp2() const = 0;
    virtual std::string emitMLIR() const = 0;
    
    // Graph analysis
    bool dominates(Node* other) const;
    std::vector<Node*> reachableFrom() const;
    
    virtual ~Node() = default;
};
```

## Stage0 Integration

### Graph Builder API

The `GraphBuilder` class constructs Sea of Nodes IR from AST:

```cpp
class GraphBuilder {
    Node* _start;               // Entry node
    Node* _stop;                // Exit node
    std::vector<Node*> _nodes;  // All nodes
    ScopeStack _scopes;         // Variable bindings
    
public:
    GraphBuilder();
    
    // Node creation
    Node* makeConstant(int64_t value, Type* type);
    Node* makeParameter(int index, Type* type);
    Node* makeAdd(Node* lhs, Node* rhs);
    Node* makeSub(Node* lhs, Node* rhs);
    Node* makeLoad(Node* address, Node* memory);
    Node* makeStore(Node* address, Node* value, Node* memory);
    
    // Control flow
    Node* makeReturn(Node* value, Node* control);
    Node* makeIf(Node* condition, Node* control);
    Node* makeRegion(std::vector<Node*> controls);
    Node* makeLoop(Node* entry);
    
    // AST translation
    Node* buildFromAST(const TranslationUnit& unit);
    Node* buildExpression(const Expression& expr);
    Node* buildStatement(const Statement& stmt);
    
    // Graph queries
    std::vector<Node*> topologicalSort() const;
    std::vector<Node*> reversePostOrder() const;
};
```

### Example: Building IR from AST

```cpp
// Source: int x = a + b; return x * 2;
Node* buildExample(GraphBuilder& builder) {
    // Parameters
    Node* param_a = builder.makeParameter(0, TypeInteger::INT);
    Node* param_b = builder.makeParameter(1, TypeInteger::INT);
    
    // Add operation: x = a + b
    Node* add = builder.makeAdd(param_a, param_b);
    
    // Constant 2
    Node* const_2 = builder.makeConstant(2, TypeInteger::INT);
    
    // Multiply: x * 2
    Node* mul = builder.makeMul(add, const_2);
    
    // Return
    Node* ret = builder.makeReturn(mul, builder.start());
    
    return ret;
}
```

## Key Optimizations Enabled

### 1. Global Value Numbering (GVN)

Sea of Nodes naturally supports GVN because:
- Equivalent expressions produce identical nodes
- Hash-consing eliminates duplicates automatically
- No need for separate CSE pass

```cpp
// Both x and y point to the same node
Node* x = builder.makeAdd(a, b);
Node* y = builder.makeAdd(a, b);  // Returns existing node
assert(x == y);
```

### 2. Dead Code Elimination

Unreachable nodes have no outputs and can be pruned:

```cpp
void GraphBuilder::eliminateDeadCode() {
    // Mark all nodes reachable from stop node
    std::unordered_set<Node*> live;
    std::queue<Node*> worklist;
    worklist.push(_stop);
    
    while (!worklist.empty()) {
        Node* node = worklist.front();
        worklist.pop();
        
        if (live.insert(node).second) {
            for (Node* input : node->inputs()) {
                worklist.push(input);
            }
        }
    }
    
    // Remove dead nodes
    _nodes.erase(
        std::remove_if(_nodes.begin(), _nodes.end(),
            [&](Node* n) { return !live.count(n); }),
        _nodes.end()
    );
}
```

### 3. Algebraic Simplification

Pattern matching on node structure enables rewrites:

```cpp
Node* simplifyAdd(Node* add) {
    assert(add->kind() == NodeKind::ADD);
    
    Node* lhs = add->in(0);
    Node* rhs = add->in(1);
    
    // x + 0 → x
    if (rhs->kind() == NodeKind::CONSTANT && rhs->asConstant()->value() == 0) {
        return lhs;
    }
    
    // 0 + x → x
    if (lhs->kind() == NodeKind::CONSTANT && lhs->asConstant()->value() == 0) {
        return rhs;
    }
    
    // (x + c1) + c2 → x + (c1 + c2)
    if (lhs->kind() == NodeKind::ADD && rhs->kind() == NodeKind::CONSTANT) {
        Node* inner_rhs = lhs->in(1);
        if (inner_rhs->kind() == NodeKind::CONSTANT) {
            int64_t sum = inner_rhs->asConstant()->value() + 
                         rhs->asConstant()->value();
            return makeAdd(lhs->in(0), makeConstant(sum));
        }
    }
    
    return add;
}
```

## N-Way Transpiler Foundation

### Emission Strategy

Each node implements target-specific emission:

```cpp
class AddNode : public Node {
public:
    std::string emitC() const override {
        return "(" + in(0)->emitC() + " + " + in(1)->emitC() + ")";
    }
    
    std::string emitCpp() const override {
        // Same as C for simple addition
        return emitC();
    }
    
    std::string emitCpp2() const override {
        // cpp2 uses more type-safe operations
        return "cpp2::add(" + in(0)->emitCpp2() + ", " + 
               in(1)->emitCpp2() + ")";
    }
    
    std::string emitMLIR() const override {
        return "%v" + std::to_string(_nodeId) + " = arith.addi " +
               in(0)->emitMLIR() + ", " + in(1)->emitMLIR();
    }
};
```

### Pattern-Based Lowering

For complex patterns, use declarative rules:

```cpp
struct LoweringPattern {
    NodeKind kind;
    std::function<bool(Node*)> matches;
    std::function<std::string(Node*, Target)> emit;
};

std::vector<LoweringPattern> patterns = {
    // Pattern: a * 2 → a << 1 (for C/C++)
    {NodeKind::MUL, 
     [](Node* n) { 
         return n->in(1)->kind() == NodeKind::CONSTANT && 
                isPowerOfTwo(n->in(1)->asConstant()->value());
     },
     [](Node* n, Target t) {
         if (t == Target::C || t == Target::Cpp) {
             int shift = log2(n->in(1)->asConstant()->value());
             return "(" + n->in(0)->emit(t) + " << " + 
                    std::to_string(shift) + ")";
         }
         return n->defaultEmit(t);
     }
    }
};
```

## Integration with Stage0

### AST to Sea of Nodes Translation

```cpp
Node* GraphBuilder::buildFromAST(const TranslationUnit& unit) {
    // Create start node
    _start = new StartNode();
    
    // Build parameter nodes for each function parameter
    for (const auto& fn : unit.functions) {
        for (size_t i = 0; i < fn.parameters.size(); ++i) {
            Node* param = makeParameter(i, 
                resolveType(fn.parameters[i].type));
            _scopes.bind(fn.parameters[i].name, param);
        }
        
        // Build function body
        Node* body = buildStatement(fn.body);
        
        // Create return/stop
        _stop = makeReturn(body, _start);
    }
    
    return _stop;
}
```

### Preserving Source Location

```cpp
class Node {
    SourceLocation _location;  // Original source position
    
public:
    void setLocation(const SourceLocation& loc) { _location = loc; }
    const SourceLocation& location() const { return _location; }
    
    // Emit debug info for target languages
    std::string emitDebugInfo(Target t) const;
};
```

## Testing Strategy

### Graph Invariant Checks

```cpp
void verifyGraphInvariants(Node* root) {
    std::unordered_set<Node*> visited;
    std::queue<Node*> worklist;
    worklist.push(root);
    
    while (!worklist.empty()) {
        Node* node = worklist.front();
        worklist.pop();
        
        if (!visited.insert(node).second) continue;
        
        // Check: All inputs are non-null
        for (size_t i = 0; i < node->numInputs(); ++i) {
            assert(node->in(i) != nullptr);
        }
        
        // Check: Type consistency
        if (node->isArithmetic()) {
            Type* lhs_type = node->in(0)->type();
            Type* rhs_type = node->in(1)->type();
            assert(lhs_type->meet(rhs_type) != TypeBottom::BOTTOM);
        }
        
        // Continue traversal
        for (Node* input : node->inputs()) {
            worklist.push(input);
        }
    }
}
```

### Roundtrip Testing

Verify that AST → Sea of Nodes → Target language produces correct semantics:

```cpp
void testRoundtrip(const std::string& source) {
    // Parse to AST
    auto ast = parse(source);
    
    // Build Sea of Nodes
    GraphBuilder builder;
    Node* graph = builder.buildFromAST(ast);
    
    // Emit to C++
    std::string cpp = graph->emitCpp();
    
    // Compile and run both versions
    int expected = execute(source);
    int actual = execute(cpp);
    
    assert(expected == actual);
}
```

## Next Steps

Chapter 19 establishes the IR foundation. Subsequent chapters build on this:

- **Chapter 20:** N-way transpiler architecture and target abstraction
- **Chapter 21:** Copy/move elision through SSA-based analysis
- **Chapter 22:** Lifetime analysis and automatic rescoping
- **Chapter 23:** Tblgen-based declarative node specifications
- **Chapter 24:** Integration, optimization pipeline, and performance validation

## References

- Sea of Nodes paper: Cliff Click's PhD thesis
- Band 5 Architecture Summary (`docs/BAND5_ARCHITECTURE_SUMMARY.md`)
- Stage0 AST definitions (`src/stage0/ast.h`)
- Type system implementation (`src/stage0/type.h`)
