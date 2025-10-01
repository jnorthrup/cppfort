# Chapter 22 Implementation: Lifetime Analysis and Automatic Rescoping

## Overview

Chapter 22 implements **lifetime analysis and automatic rescoping** to enable optimal memory management without garbage collection. By computing precise lifetime regions in the Sea of Nodes IR and applying constraint-based analysis, the compiler can:

1. Stack-allocate values that don't escape
2. Extend lifetimes minimally when needed
3. Insert destructors at optimal points
4. Detect and prevent use-after-free errors at compile time

This chapter builds on Chapter 21's escape analysis and integrates with `docs/stage2.md`'s borrow checking framework.

## Lifetime Regions

### Representing Lifetimes

```cpp
class Lifetime {
public:
    enum class Kind {
        Temporary,   // Expression temporaries
        Local,       // Local variable
        Parameter,   // Function parameter
        Static,      // Static/global lifetime
        Heap,        // Heap-allocated
    };
    
private:
    Kind _kind;
    Node* _allocation;      // Allocation site
    Node* _scope;           // Scope node (region/function)
    std::unordered_set<Node*> _uses;  // All uses
    Node* _lastUse;         // Last use in program order
    
public:
    Lifetime(Node* alloc, Node* scope) 
        : _kind(Kind::Local), _allocation(alloc), _scope(scope), 
          _lastUse(nullptr) {}
    
    void addUse(Node* use) {
        _uses.insert(use);
    }
    
    void computeLastUse() {
        // Find last use in post-order traversal
        auto postOrder = _scope->reversePostOrder();
        for (auto it = postOrder.rbegin(); it != postOrder.rend(); ++it) {
            if (_uses.count(*it)) {
                _lastUse = *it;
                break;
            }
        }
    }
    
    bool contains(Node* use) const {
        return _uses.count(use) > 0;
    }
    
    bool outlives(const Lifetime& other) const {
        // Check if this lifetime extends beyond other
        return _scope->dominates(other._scope) &&
               (!_lastUse || !other._lastUse ||
                dominatesInOrder(_lastUse, other._lastUse));
    }
    
    // Merge two lifetimes (for phi nodes)
    static Lifetime* merge(Lifetime* a, Lifetime* b);
};
```

### Lifetime Variables and Constraints

Following `docs/stage2.md`, we model lifetimes as lattice variables:

```cpp
class LifetimeVar {
    std::string _name;           // L_a, L_b, etc.
    Node* _allocation;           // Source allocation
    Lifetime::Kind _kind;
    
    // Constraints
    std::vector<LifetimeVar*> _mustOutlive;  // L_a ⊒ L_b
    std::vector<LifetimeVar*> _mustBeOutlived; // L_a ⊑ L_b
    
public:
    void addConstraint(LifetimeVar* other, bool mustOutlive) {
        if (mustOutlive) {
            _mustOutlive.push_back(other);
        } else {
            _mustBeOutlived.push_back(other);
        }
    }
    
    // Check if constraints are satisfiable
    bool isSatisfiable() const;
    
    // Compute minimal lifetime that satisfies constraints
    Lifetime* solve() const;
};

class LifetimeConstraintSystem {
    std::vector<LifetimeVar*> _variables;
    std::vector<std::pair<LifetimeVar*, LifetimeVar*>> _constraints;
    
public:
    LifetimeVar* createVar(Node* allocation) {
        auto* var = new LifetimeVar(allocation);
        _variables.push_back(var);
        return var;
    }
    
    void addConstraint(LifetimeVar* a, LifetimeVar* b, bool aOutlivesB) {
        if (aOutlivesB) {
            a->addConstraint(b, true);
        } else {
            b->addConstraint(a, true);
        }
        _constraints.push_back({a, b});
    }
    
    // Solve constraints using subsumption lattice
    bool solve();
    
    // Verify solution satisfies all constraints
    bool verify() const;
};
```

## Lifetime Analysis Algorithm

### Phase 1: Collect Allocation Sites

```cpp
class LifetimeAnalyzer {
    std::unordered_map<Node*, Lifetime*> _lifetimes;
    LifetimeConstraintSystem _constraints;
    
public:
    void analyze(Node* function) {
        // 1. Identify all allocations
        collectAllocations(function);
        
        // 2. Build use-def chains
        buildUseDefChains(function);
        
        // 3. Generate constraints
        generateConstraints(function);
        
        // 4. Solve constraint system
        if (!_constraints.solve()) {
            reportError("Unsatisfiable lifetime constraints");
            return;
        }
        
        // 5. Compute optimal scopes
        computeOptimalScopes(function);
        
        // 6. Insert destructors
        insertDestructors(function);
    }
    
private:
    void collectAllocations(Node* function) {
        for (Node* node : function->reachable()) {
            if (node->kind() == NodeKind::ALLOC) {
                Node* scope = findScope(node);
                auto* lifetime = new Lifetime(node, scope);
                _lifetimes[node] = lifetime;
            }
        }
    }
    
    Node* findScope(Node* node) {
        // Find enclosing region or function
        Node* current = node;
        while (current) {
            if (current->kind() == NodeKind::REGION ||
                current->kind() == NodeKind::START) {
                return current;
            }
            current = current->parent();
        }
        return nullptr;
    }
};
```

### Phase 2: Generate Constraints from Uses

```cpp
void LifetimeAnalyzer::generateConstraints(Node* function) {
    for (auto& [alloc, lifetime] : _lifetimes) {
        LifetimeVar* allocVar = _constraints.createVar(alloc);
        
        // For each use of this allocation
        for (Node* use : alloc->outputs()) {
            generateUseConstraints(allocVar, use, lifetime);
        }
    }
}

void LifetimeAnalyzer::generateUseConstraints(
    LifetimeVar* allocVar, 
    Node* use,
    Lifetime* allocLifetime
) {
    switch (use->kind()) {
        case NodeKind::LOAD:
        case NodeKind::STORE:
            // Memory access: allocation must outlive use
            // L_alloc ⊒ L_use
            allocVar->addConstraint(
                _constraints.getUseVar(use), 
                true  // allocVar must outlive useVar
            );
            break;
            
        case NodeKind::CALL:
            // Function call: depends on parameter kind
            generateCallConstraints(allocVar, use);
            break;
            
        case NodeKind::RETURN:
            // Return: allocation escapes function
            // L_alloc ⊒ L_function
            allocVar->addConstraint(
                _constraints.getFunctionVar(use->function()),
                true
            );
            allocLifetime->setKind(Lifetime::Kind::Heap);
            break;
            
        case NodeKind::PHI:
            // Merge point: join lifetimes
            generatePhiConstraints(allocVar, use);
            break;
            
        default:
            // Default: allocation must outlive any use
            allocVar->addConstraint(
                _constraints.getUseVar(use),
                true
            );
            break;
    }
}

void LifetimeAnalyzer::generateCallConstraints(
    LifetimeVar* allocVar,
    Node* call
) {
    // Find which parameter this allocation goes to
    for (size_t i = 0; i < call->numInputs(); ++i) {
        if (call->in(i) == allocVar->allocation()) {
            ParameterKind kind = call->functionType()
                ->parameter(i).kind;
            
            switch (kind) {
                case ParameterKind::In:
                    // Borrowed: L_alloc ⊒ L_call
                    allocVar->addConstraint(
                        _constraints.getCallVar(call),
                        true
                    );
                    break;
                    
                case ParameterKind::Copy:
                    // Copied: no constraint (independent lifetime)
                    break;
                    
                case ParameterKind::Move:
                    // Moved: ownership transferred
                    // L_alloc ⊒ L_call (must live until call)
                    // After call, original binding is dead
                    allocVar->addConstraint(
                        _constraints.getCallVar(call),
                        true
                    );
                    allocVar->allocation()->markMovedAt(call);
                    break;
                    
                case ParameterKind::Out:
                case ParameterKind::InOut:
                    // Mutated: L_alloc ⊒ L_call ⊒ L_subsequent_uses
                    allocVar->addConstraint(
                        _constraints.getCallVar(call),
                        true
                    );
                    break;
                    
                default:
                    // Conservative: assume escape
                    allocVar->allocation()->markEscaped();
                    break;
            }
        }
    }
}
```

### Phase 3: Solve Constraint System

Using subsumption lattice from `docs/stage2.md`:

```cpp
bool LifetimeConstraintSystem::solve() {
    // Initialize: all lifetimes at bottom (minimal)
    for (LifetimeVar* var : _variables) {
        var->setLevel(0);  // Bottom of lattice
    }
    
    // Fixed-point iteration
    bool changed = true;
    int iterations = 0;
    const int MAX_ITERATIONS = 1000;
    
    while (changed && iterations < MAX_ITERATIONS) {
        changed = false;
        iterations++;
        
        // For each constraint L_a ⊒ L_b (a must outlive b)
        for (auto& [a, b] : _constraints) {
            // If a doesn't outlive b, extend a's lifetime
            if (a->level() <= b->level()) {
                int newLevel = b->level() + 1;
                a->setLevel(newLevel);
                changed = true;
            }
        }
    }
    
    if (iterations >= MAX_ITERATIONS) {
        // Cyclic dependency detected
        return false;
    }
    
    return verify();
}

bool LifetimeConstraintSystem::verify() const {
    // Check all constraints are satisfied
    for (const auto& [a, b] : _constraints) {
        if (a->level() <= b->level()) {
            return false;  // Constraint violated
        }
    }
    return true;
}
```

## Automatic Scope Extension

### Computing Minimal Scopes

```cpp
class ScopeExtender {
public:
    void extendScopes(Node* function, 
                      const std::unordered_map<Node*, Lifetime*>& lifetimes) {
        for (auto& [alloc, lifetime] : lifetimes) {
            Node* minimalScope = computeMinimalScope(lifetime);
            
            if (minimalScope != lifetime->scope()) {
                // Need to extend lifetime
                extendToScope(alloc, minimalScope);
            }
        }
    }
    
private:
    Node* computeMinimalScope(Lifetime* lifetime) {
        // Find smallest scope that contains all uses
        Node* minScope = lifetime->scope();
        
        for (Node* use : lifetime->uses()) {
            Node* useScope = findScope(use);
            minScope = findCommonAncestor(minScope, useScope);
        }
        
        return minScope;
    }
    
    Node* findCommonAncestor(Node* a, Node* b) {
        // Find lowest common ancestor in dominator tree
        std::unordered_set<Node*> aAncestors;
        Node* current = a;
        while (current) {
            aAncestors.insert(current);
            current = current->dominator();
        }
        
        current = b;
        while (current) {
            if (aAncestors.count(current)) {
                return current;
            }
            current = current->dominator();
        }
        
        return nullptr;  // Should not happen in valid CFG
    }
    
    void extendToScope(Node* allocation, Node* newScope) {
        // Move allocation to new scope
        allocation->setScope(newScope);
        
        // May need to convert to heap allocation if scope is too large
        if (needsHeapAllocation(allocation, newScope)) {
            convertToHeapAllocation(allocation);
        }
    }
    
    bool needsHeapAllocation(Node* allocation, Node* scope) const {
        // Check if allocation escapes function
        if (scope->kind() == NodeKind::START) {
            return true;  // Function scope
        }
        
        // Check if returned
        for (Node* use : allocation->outputs()) {
            if (use->kind() == NodeKind::RETURN) {
                return true;
            }
        }
        
        return false;
    }
};
```

### Heap Allocation Conversion

When stack allocation isn't possible:

```cpp
class HeapConverter {
public:
    void convertToHeap(Node* allocation) {
        Type* elementType = allocation->type();
        Type* pointerType = TypePointer::make(elementType, false);
        
        // Replace stack allocation with heap allocation
        Node* heapAlloc = new HeapAllocNode(elementType);
        
        // Update all uses to dereference pointer
        for (Node* use : allocation->outputs()) {
            if (use->kind() == NodeKind::LOAD ||
                use->kind() == NodeKind::STORE) {
                // Already operates on pointers, just update type
                use->setType(pointerType);
            } else {
                // Insert dereference
                Node* deref = new LoadNode(heapAlloc);
                use->replaceInput(allocation, deref);
            }
        }
        
        // Insert deallocation at last use
        Node* lastUse = findLastUse(allocation);
        insertAfter(lastUse, new DeallocNode(heapAlloc));
        
        allocation->replaceWith(heapAlloc);
    }
};
```

## Destructor Insertion

### Automatic Destructor Placement

```cpp
class DestructorInserter {
public:
    void insertDestructors(Node* function,
                          const std::unordered_map<Node*, Lifetime*>& lifetimes) {
        for (auto& [alloc, lifetime] : lifetimes) {
            if (needsDestructor(alloc->type())) {
                insertDestructor(alloc, lifetime);
            }
        }
    }
    
private:
    bool needsDestructor(Type* type) const {
        if (auto* struct_type = dynamic_cast<TypeStruct*>(type)) {
            return struct_type->hasDestructor();
        }
        return false;
    }
    
    void insertDestructor(Node* allocation, Lifetime* lifetime) {
        // Find optimal insertion point: after last use
        Node* lastUse = lifetime->lastUse();
        
        if (!lastUse) {
            // Never used, insert at end of scope
            lastUse = lifetime->scope()->exit();
        }
        
        // Insert destructor call
        Node* destructor = new DestructorNode(allocation);
        insertAfter(lastUse, destructor);
        
        // Handle exceptional control flow
        insertExceptionDestructors(allocation, lifetime);
    }
    
    void insertExceptionDestructors(Node* allocation, Lifetime* lifetime) {
        // Find all exceptional exit points in lifetime scope
        for (Node* node : lifetime->scope()->reachable()) {
            if (node->kind() == NodeKind::THROW ||
                node->kind() == NodeKind::CALL && 
                node->mayThrow()) {
                
                // Insert destructor on exceptional path
                Node* cleanup = node->exceptionPath();
                if (cleanup && cleanup->dominates(node)) {
                    Node* destructor = new DestructorNode(allocation);
                    cleanup->prependCleanup(destructor);
                }
            }
        }
    }
};
```

## Integration with cpp2 Syntax

### Lifetime Annotations in cpp2

```cpp
// cpp2 syntax with lifetime annotations:
func: (data: vector^) -> int = {  // ^ means non-null
    local: vector = data;          // Local copy
    return local.size();
}  // local destroyed here automatically

// Generated C++ with optimal lifetimes:
auto func(std::vector<int> const* data) -> int {
    std::vector<int> local = *data;
    int result = local.size();
    local.~vector();  // Explicit destructor call
    return result;
}
```

### Emitting Lifetime-Aware Code

```cpp
class LifetimeAwareEmitter {
public:
    std::string emitCpp(Node* function,
                       const std::unordered_map<Node*, Lifetime*>& lifetimes) {
        std::string output;
        
        for (Node* node : function->schedule()) {
            switch (node->kind()) {
                case NodeKind::ALLOC: {
                    Lifetime* lt = lifetimes.at(node);
                    if (lt->kind() == Lifetime::Kind::Local) {
                        // Stack allocation
                        output += emitStackAlloc(node);
                    } else {
                        // Heap allocation
                        output += emitHeapAlloc(node);
                    }
                    break;
                }
                
                case NodeKind::DESTRUCTOR: {
                    output += emitDestructor(node);
                    break;
                }
                
                default:
                    output += node->emitCpp();
                    break;
            }
        }
        
        return output;
    }
    
private:
    std::string emitStackAlloc(Node* node) const {
        Type* type = node->type();
        return type->name() + " " + node->varName() + ";\n";
    }
    
    std::string emitHeapAlloc(Node* node) const {
        Type* type = node->type();
        return "auto " + node->varName() + 
               " = std::make_unique<" + type->name() + ">();\n";
    }
    
    std::string emitDestructor(Node* node) const {
        Node* target = node->in(0);
        return target->varName() + ".~" + 
               target->type()->name() + "();\n";
    }
};
```

## Validation and Safety

### Use-After-Free Detection

```cpp
class UseAfterFreeDetector {
public:
    std::vector<std::string> detect(Node* function,
        const std::unordered_map<Node*, Lifetime*>& lifetimes) {
        
        std::vector<std::string> errors;
        
        for (auto& [alloc, lifetime] : lifetimes) {
            // Find destructor insertion point
            Node* destructor = findDestructor(alloc);
            if (!destructor) continue;
            
            // Check for uses after destructor
            for (Node* use : alloc->outputs()) {
                if (use == destructor) continue;
                
                if (dominatesInOrder(destructor, use)) {
                    errors.push_back(
                        "Use-after-free: " + 
                        alloc->varName() + 
                        " used after destruction at " +
                        use->location().toString()
                    );
                }
            }
        }
        
        return errors;
    }
};
```

### Constraint Visualization

```cpp
void visualizeConstraints(const LifetimeConstraintSystem& system) {
    std::cout << "digraph Lifetimes {\n";
    
    for (const auto& [a, b] : system.constraints()) {
        std::cout << "  " << a->name() 
                 << " -> " << b->name()
                 << " [label=\"outlives\"];\n";
    }
    
    for (const auto* var : system.variables()) {
        std::cout << "  " << var->name()
                 << " [label=\"" << var->name() 
                 << "\\nlevel=" << var->level() << "\"];\n";
    }
    
    std::cout << "}\n";
}
```

## Performance Considerations

### Caching Lifetime Information

```cpp
class LifetimeCache {
    std::unordered_map<Node*, Lifetime*> _cache;
    
public:
    Lifetime* get(Node* allocation) {
        auto it = _cache.find(allocation);
        if (it != _cache.end()) {
            return it->second;
        }
        
        // Compute and cache
        auto* lifetime = computeLifetime(allocation);
        _cache[allocation] = lifetime;
        return lifetime;
    }
    
    void invalidate() {
        _cache.clear();
    }
};
```

## Next Steps

- **Chapter 23:** Tblgen-based declarative specifications for lifetime rules
- **Chapter 24:** Integration with complete optimization pipeline

## References

- Rust borrow checker: [RFC 2094](https://rust-lang.github.io/rfcs/2094-nll.html)
- C++ object lifetime: ISO C++20 [basic.life]
- Stage2 escape analysis: `docs/stage2.md`
- SSA and dominance: Cooper & Torczon, Ch. 9
