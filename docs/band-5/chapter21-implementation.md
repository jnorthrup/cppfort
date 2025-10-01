# Chapter 21 Implementation: Perfect Copy and Move Elision via SSA

## Overview

Chapter 21 implements **perfect copy and move elision** by leveraging the Single Static Assignment (SSA) properties of the Sea of Nodes IR. By analyzing def-use chains and lifetime relationships, the compiler can eliminate unnecessary copies and moves, generating optimal C++ code that matches hand-optimized implementations.

## Core Principles

### 1. SSA Form and Value Semantics

In Sea of Nodes IR, every value is defined exactly once:

```cpp
// Traditional code with multiple definitions
int x = 5;
x = x + 1;  // Redefines x
x = x * 2;  // Redefines x again

// SSA form with unique definitions
int x₀ = 5;
int x₁ = x₀ + 1;  // New name
int x₂ = x₁ * 2;  // New name
```

**Key insight:** SSA eliminates aliasing confusion, making it trivial to track when values are last used.

### 2. Last Use Analysis

A value can be moved (rather than copied) if:
1. This use is the **last use** of the value
2. The value is **moveable** (not const, not trivially copyable where copy is cheaper)
3. The target context **accepts rvalue references**

```cpp
class LastUseAnalyzer {
    std::unordered_map<Node*, int> _useCount;
    std::unordered_map<Node*, std::unordered_set<Node*>> _uses;
    
public:
    void analyze(Node* root) {
        // Count uses for each node
        std::function<void(Node*)> visit = [&](Node* node) {
            for (Node* input : node->inputs()) {
                _useCount[input]++;
                _uses[input].insert(node);
                visit(input);
            }
        };
        visit(root);
    }
    
    bool isLastUse(Node* value, Node* use) const {
        // Check if this is the only remaining use
        auto it = _uses.find(value);
        if (it == _uses.end()) return false;
        
        return it->second.size() == 1 && 
               it->second.count(use) > 0;
    }
    
    int useCount(Node* value) const {
        auto it = _useCount.find(value);
        return it != _useCount.end() ? it->second : 0;
    }
};
```

### 3. Move Eligibility

Not all values benefit from moving:

```cpp
class MoveEligibilityChecker {
public:
    bool canMove(Type* type) const {
        // Primitive types: copy is as cheap as move
        if (type->isPrimitive()) {
            return false;
        }
        
        // Pointers: copy is trivial
        if (dynamic_cast<TypePointer*>(type)) {
            return false;
        }
        
        // Small types (≤16 bytes): copy is usually faster
        if (type->size() <= 16) {
            return false;
        }
        
        // Non-copyable types: must move
        if (type->hasUniqueOwnership()) {
            return true;
        }
        
        // Large aggregates: move beneficial
        if (type->size() > 64) {
            return true;
        }
        
        // Check for move constructor
        if (auto* struct_type = dynamic_cast<TypeStruct*>(type)) {
            return struct_type->hasMoveConstructor();
        }
        
        return false;
    }
    
    bool mustMove(Type* type) const {
        // Unique ownership types (e.g., unique_ptr)
        if (type->hasUniqueOwnership()) {
            return true;
        }
        
        // Move-only types
        if (auto* struct_type = dynamic_cast<TypeStruct*>(type)) {
            return !struct_type->isCopyable() && 
                   struct_type->isMoveable();
        }
        
        return false;
    }
};
```

## Copy Elision Patterns

### Pattern 1: Return Value Optimization (RVO)

Detect when a local variable is returned directly:

```cpp
class RVODetector {
public:
    struct RVOOpportunity {
        Node* allocation;   // Local variable allocation
        Node* returnNode;   // Return statement
        bool eligible;      // Can apply RVO
    };
    
    std::vector<RVOOpportunity> findOpportunities(Node* function) {
        std::vector<RVOOpportunity> opportunities;
        
        // Find return nodes
        for (Node* node : function->reachable()) {
            if (node->kind() != NodeKind::RETURN) continue;
            
            Node* returnValue = node->in(0);
            
            // Check if returning a local allocation
            if (returnValue->kind() == NodeKind::ALLOC) {
                // Check if allocation doesn't escape
                if (!escapes(returnValue, node)) {
                    opportunities.push_back({
                        returnValue,
                        node,
                        true
                    });
                }
            }
            
            // Check if returning result of constructor
            if (returnValue->kind() == NodeKind::CONSTRUCT) {
                Node* target = returnValue->in(0);
                if (target->kind() == NodeKind::ALLOC && 
                    !escapes(target, node)) {
                    opportunities.push_back({
                        target,
                        node,
                        true
                    });
                }
            }
        }
        
        return opportunities;
    }
    
private:
    bool escapes(Node* allocation, Node* boundary) const {
        // Check if allocation is used outside its scope
        for (Node* use : allocation->outputs()) {
            if (use == boundary) continue;
            
            // Check if use is reachable without going through boundary
            if (canReach(use, boundary)) {
                return true;  // Escapes
            }
        }
        return false;
    }
};
```

### Pattern 2: Named Return Value Optimization (NRVO)

Optimize when a named variable is constructed and returned:

```cpp
// Source code:
std::string foo() {
    std::string result;
    result = "Hello";
    result += " World";
    return result;  // NRVO candidate
}

// Without NRVO (copy):
std::string foo() {
    std::string result;
    // ... operations on result ...
    std::string __return_value = result;  // Copy!
    return __return_value;
}

// With NRVO (elision):
void foo(std::string* __return_slot) {
    std::string* result = __return_slot;  // Alias return slot
    // ... operations directly on return slot ...
    // No copy needed!
}
```

Implementation:

```cpp
class NRVOTransform {
public:
    void apply(Node* function, const RVOOpportunity& opp) {
        // Transform function to use return slot
        
        // 1. Add return slot parameter
        Node* returnSlot = function->addParameter(
            "__return_slot",
            makePointerType(opp.allocation->type())
        );
        
        // 2. Replace allocation with return slot
        opp.allocation->replaceWith(returnSlot);
        
        // 3. Remove the return value (now void return)
        opp.returnNode->setInput(0, nullptr);
        
        // 4. Change return type to void
        function->setReturnType(TypeVoid::VOID);
    }
};
```

### Pattern 3: Copy Elision in Pass-by-Value

```cpp
// Source: pass temporary by value
void processData(std::vector<int> data);  // Pass by value

void caller() {
    processData(createVector());  // Temporary
}

// Optimized: elide copy by passing prvalue directly
// The temporary is constructed directly in callee's parameter
```

Detection:

```cpp
class PassByValueOptimizer {
public:
    bool canElide(Node* callSite, int paramIndex) {
        Node* arg = callSite->in(paramIndex);
        
        // Must be a temporary (prvalue)
        if (!isTemporary(arg)) return false;
        
        // Must be last use
        LastUseAnalyzer lua;
        lua.analyze(callSite);
        if (!lua.isLastUse(arg, callSite)) return false;
        
        // Parameter must accept by value
        FunctionType* fnType = callSite->functionType();
        if (fnType->parameter(paramIndex).kind != ParameterKind::Copy) {
            return false;
        }
        
        return true;
    }
    
private:
    bool isTemporary(Node* node) const {
        // Temporaries: results of constructors, operators, etc.
        return node->kind() == NodeKind::CONSTRUCT ||
               node->kind() == NodeKind::ADD ||
               node->kind() == NodeKind::CALL;
    }
};
```

## Move Insertion

### Automatic Move at Last Use

```cpp
class MoveInserter {
    LastUseAnalyzer _lastUse;
    MoveEligibilityChecker _eligibility;
    
public:
    void insertMoves(Node* root) {
        _lastUse.analyze(root);
        
        for (Node* node : root->reachable()) {
            for (size_t i = 0; i < node->numInputs(); ++i) {
                Node* input = node->in(i);
                
                // Check if this is last use and should be moved
                if (_lastUse.isLastUse(input, node) &&
                    _eligibility.canMove(input->type())) {
                    
                    // Insert move node
                    Node* moveNode = new MoveNode(input);
                    node->setInput(i, moveNode);
                }
            }
        }
    }
};
```

### Move Semantics in Emission

```cpp
class MoveNode : public Node {
public:
    explicit MoveNode(Node* value) : Node(NodeKind::MOVE) {
        addInput(value);
        _type = value->type();
    }
    
    std::string emitCpp() const override {
        return "std::move(" + in(0)->emitCpp() + ")";
    }
    
    std::string emitCpp2() const override {
        // cpp2 uses 'move' parameter kind instead
        return in(0)->emitCpp2();  // Handled by parameter
    }
    
    std::string emitC() const override {
        // C doesn't have move semantics
        return in(0)->emitC();  // Regular copy
    }
    
    std::string emitMLIR() const override {
        // MLIR represents move as special attribute
        return in(0)->emitMLIR() + " { move = true }";
    }
};
```

## Integration with cpp2 Parameter Kinds

### Mapping to cpp2::impl Templates

The cpp2 language has explicit parameter kinds:

```cpp
// cpp2 syntax:
func: (in x: int, copy y: string, move z: vector) = { ... }

// Generated C++:
auto func(
    cpp2::impl::in<int> x,           // const int&
    cpp2::impl::copy<std::string> y, // std::string (by value)
    cpp2::impl::move<std::vector> z  // std::vector&& (rvalue ref)
) { ... }
```

Implementation in stage0:

```cpp
class ParameterKindResolver {
public:
    ParameterKind resolve(Node* arg, const FunctionDecl& fn, int index) {
        // If explicitly specified in source, use that
        if (fn.parameters[index].kind != ParameterKind::Default) {
            return fn.parameters[index].kind;
        }
        
        // Otherwise, infer from usage
        LastUseAnalyzer lua;
        lua.analyze(arg);
        
        // Small types: pass by value (implicit copy)
        if (arg->type()->size() <= 8) {
            return ParameterKind::In;
        }
        
        // Last use of large type: move
        if (lua.isLastUse(arg, /* call node */) &&
            _eligibility.canMove(arg->type())) {
            return ParameterKind::Move;
        }
        
        // Multiple uses of large type: const ref
        if (lua.useCount(arg) > 1) {
            return ParameterKind::In;
        }
        
        // Single use, will be copied anyway: pass by value
        return ParameterKind::Copy;
    }
    
private:
    MoveEligibilityChecker _eligibility;
};
```

## Escape Analysis Integration

### Computing Escape Information

```cpp
class EscapeAnalyzer {
public:
    enum class EscapeKind {
        NoEscape,      // Value doesn't escape
        ScopeEscape,   // Escapes current scope but not function
        ReturnEscape,  // Returned from function
        GlobalEscape,  // Stored in global/heap
    };
    
    EscapeKind analyze(Node* allocation) {
        EscapeKind maxEscape = EscapeKind::NoEscape;
        
        for (Node* use : allocation->outputs()) {
            EscapeKind useEscape = analyzeUse(allocation, use);
            maxEscape = std::max(maxEscape, useEscape);
        }
        
        return maxEscape;
    }
    
private:
    EscapeKind analyzeUse(Node* allocation, Node* use) {
        switch (use->kind()) {
            case NodeKind::RETURN:
                return EscapeKind::ReturnEscape;
                
            case NodeKind::STORE:
                // Storing to global/heap?
                if (isGlobalOrHeap(use->in(0))) {
                    return EscapeKind::GlobalEscape;
                }
                return EscapeKind::ScopeEscape;
                
            case NodeKind::CALL:
                // Escapes if parameter has out/forward kind
                return analyzeCallEscape(allocation, use);
                
            default:
                // Local use only
                return EscapeKind::NoEscape;
        }
    }
    
    EscapeKind analyzeCallEscape(Node* allocation, Node* call) {
        // Find which parameter this allocation goes to
        for (size_t i = 0; i < call->numInputs(); ++i) {
            if (call->in(i) == allocation) {
                ParameterKind kind = call->functionType()
                    ->parameter(i).kind;
                    
                switch (kind) {
                    case ParameterKind::In:
                    case ParameterKind::Copy:
                        return EscapeKind::NoEscape;
                        
                    case ParameterKind::Out:
                    case ParameterKind::InOut:
                        return EscapeKind::ScopeEscape;
                        
                    case ParameterKind::Forward:
                        // Could escape, conservative
                        return EscapeKind::GlobalEscape;
                        
                    default:
                        return EscapeKind::ScopeEscape;
                }
            }
        }
        return EscapeKind::NoEscape;
    }
};
```

## Complete Optimization Pipeline

```cpp
class CopyMoveOptimizer {
    LastUseAnalyzer _lastUse;
    MoveEligibilityChecker _eligibility;
    EscapeAnalyzer _escape;
    RVODetector _rvo;
    
public:
    void optimize(Node* function) {
        // 1. Analyze last uses
        _lastUse.analyze(function);
        
        // 2. Detect RVO opportunities
        auto rvoOpps = _rvo.findOpportunities(function);
        
        // 3. Apply RVO/NRVO
        NRVOTransform nrvo;
        for (const auto& opp : rvoOpps) {
            if (opp.eligible) {
                nrvo.apply(function, opp);
            }
        }
        
        // 4. Insert moves at last uses
        MoveInserter inserter;
        inserter.insertMoves(function);
        
        // 5. Eliminate unnecessary copies
        eliminateDeadCopies(function);
    }
    
private:
    void eliminateDeadCopies(Node* function) {
        // Remove copy nodes that are immediately moved
        for (Node* node : function->reachable()) {
            if (node->kind() == NodeKind::COPY) {
                // Check if result is immediately moved
                if (node->outputs().size() == 1 &&
                    node->outputs()[0]->kind() == NodeKind::MOVE) {
                    // Bypass the copy
                    node->outputs()[0]->replaceWith(node->in(0));
                }
            }
        }
    }
};
```

## Validation and Testing

### Verify Move Correctness

```cpp
void verifyMoveCorrectness(Node* function) {
    for (Node* node : function->reachable()) {
        if (node->kind() == NodeKind::MOVE) {
            Node* value = node->in(0);
            
            // Verify this is indeed the last use
            int usesAfter = countUsesAfter(value, node);
            assert(usesAfter == 0 && 
                   "Move inserted but value used after move!");
        }
    }
}
```

### Performance Testing

```cpp
void benchmarkCopyElision() {
    std::string source = R"(
        vector: (n: int) -> vector = {
            v: vector;
            for (i: 0..n) v.push_back(i);
            return v;  // Should use NRVO
        }
    )";
    
    // Compile with and without optimization
    auto unoptimized = compile(source, {.copyElision = false});
    auto optimized = compile(source, {.copyElision = true});
    
    // Measure performance
    auto t1 = benchmark(unoptimized, 1000);
    auto t2 = benchmark(optimized, 1000);
    
    std::cout << "Speedup: " << (t1 / t2) << "x\n";
    // Expected: 2-5x faster for large vectors
}
```

## Next Steps

- **Chapter 22:** Lifetime analysis and automatic scope management
- **Chapter 23:** Declarative tblgen specifications for optimization patterns
- **Chapter 24:** Complete integration and whole-program optimization

## References

- C++ copy elision: [P0135R1](https://wg21.link/p0135r1)
- SSA form: Cooper & Torczon, "Engineering a Compiler"
- Stage2 escape analysis: `docs/stage2.md`
