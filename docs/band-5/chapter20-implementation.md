# Chapter 20 Implementation: N-Way Transpiler Architecture

## Overview

Chapter 20 implements the **n-way transpiler architecture** that enables simultaneous code generation for multiple target languages (C, C++, CPP2, MLIR, disassembly) from a single Sea of Nodes IR. This architecture leverages tblgen-style declarative specifications and pattern matching to achieve O(1) dispatch complexity per node.

## Architectural Pattern

### The Scaling Problem

Traditional compiler architectures scale linearly with targets:

```
Complexity = O(N × M)
where N = number of target languages
      M = number of IR node types
```

**Example:** 50 node types × 5 targets = 250 emission functions to maintain.

### The N-Way Solution

Using enum-based pattern matching and declarative lowering:

```
Complexity = O(M + N × P)
where M = node types (fixed)
      N = targets (extensible)
      P = target-specific patterns (typically P << M)
```

**Key insight:** Most nodes have identical lowering across targets; only special cases need customization.

## Target Abstraction Layer

### Target Enum and Capabilities

```cpp
enum class Target : uint8_t {
    C,          // ISO C11
    Cpp,        // ISO C++20
    Cpp2,       // Cpp2 syntax
    MLIR,       // MLIR dialect
    Disasm,     // Assembly-like representation
};

struct TargetCapabilities {
    bool supportsExceptions;
    bool supportsRTTI;
    bool supportsTemplates;
    bool supportsModules;
    bool supportsCoroutines;
    bool supportsRanges;
    int pointerWidth;  // 32 or 64 bits
    
    static const TargetCapabilities& get(Target t);
};
```

### Target Context

```cpp
class TargetContext {
    Target _target;
    const TargetCapabilities& _caps;
    std::unordered_map<std::string, std::string> _includes;
    std::vector<std::string> _preamble;
    int _indentLevel;
    
public:
    explicit TargetContext(Target t) 
        : _target(t), _caps(TargetCapabilities::get(t)), _indentLevel(0) {}
    
    // Code generation helpers
    std::string indent() const;
    void pushIndent() { _indentLevel++; }
    void popIndent() { _indentLevel--; }
    
    // Include management
    void requireInclude(const std::string& header);
    std::string emitIncludes() const;
    
    // Target queries
    Target target() const { return _target; }
    bool hasCapability(const std::string& cap) const;
    
    // Type mapping
    std::string mapType(Type* type) const;
    std::string mapOperator(NodeKind op) const;
};
```

## Pattern-Based Emission

### Pattern Registry

```cpp
struct EmissionPattern {
    NodeKind kind;
    Target target;
    
    // Pattern matching predicate
    std::function<bool(Node*, const TargetContext&)> matches;
    
    // Code generation function
    std::function<std::string(Node*, TargetContext&)> emit;
    
    // Priority for pattern selection (higher = more specific)
    int priority;
};

class PatternRegistry {
    std::multimap<std::pair<NodeKind, Target>, EmissionPattern> _patterns;
    
public:
    void registerPattern(EmissionPattern pattern);
    
    std::string emit(Node* node, TargetContext& ctx) const {
        auto key = std::make_pair(node->kind(), ctx.target());
        auto range = _patterns.equal_range(key);
        
        // Try patterns in priority order
        std::vector<EmissionPattern> candidates;
        for (auto it = range.first; it != range.second; ++it) {
            candidates.push_back(it->second);
        }
        
        std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.priority > b.priority; });
        
        for (const auto& pattern : candidates) {
            if (pattern.matches(node, ctx)) {
                return pattern.emit(node, ctx);
            }
        }
        
        // Fallback to default emission
        return defaultEmit(node, ctx);
    }
    
private:
    std::string defaultEmit(Node* node, TargetContext& ctx) const;
};
```

### Example Patterns

#### Pattern 1: Integer Addition (Common to C/C++/Cpp2)

```cpp
void registerCommonPatterns(PatternRegistry& registry) {
    // Addition for C, C++, Cpp2 (same syntax)
    EmissionPattern addPattern = {
        .kind = NodeKind::ADD,
        .target = Target::C,  // Will register for C, Cpp, Cpp2
        .matches = [](Node* n, const TargetContext& ctx) { 
            return n->type()->isInteger(); 
        },
        .emit = [](Node* n, TargetContext& ctx) {
            return "(" + emit(n->in(0), ctx) + " + " + 
                   emit(n->in(1), ctx) + ")";
        },
        .priority = 0
    };
    
    registry.registerPattern(addPattern);
    // Also register for Cpp and Cpp2
    addPattern.target = Target::Cpp;
    registry.registerPattern(addPattern);
    addPattern.target = Target::Cpp2;
    registry.registerPattern(addPattern);
}
```

#### Pattern 2: Power-of-Two Multiplication (C/C++ optimization)

```cpp
EmissionPattern mulShiftPattern = {
    .kind = NodeKind::MUL,
    .target = Target::C,
    .matches = [](Node* n, const TargetContext& ctx) {
        // Match: x * (power of 2)
        if (n->in(1)->kind() != NodeKind::CONSTANT) return false;
        int64_t val = n->in(1)->asConstant()->value();
        return val > 0 && (val & (val - 1)) == 0;
    },
    .emit = [](Node* n, TargetContext& ctx) {
        int64_t val = n->in(1)->asConstant()->value();
        int shift = __builtin_ctzll(val);
        return "(" + emit(n->in(0), ctx) + " << " + 
               std::to_string(shift) + ")";
    },
    .priority = 10  // Higher priority than default multiplication
};
```

#### Pattern 3: MLIR Arithmetic

```cpp
EmissionPattern mlirAddPattern = {
    .kind = NodeKind::ADD,
    .target = Target::MLIR,
    .matches = [](Node* n, const TargetContext& ctx) { return true; },
    .emit = [](Node* n, TargetContext& ctx) {
        std::string lhs_type = ctx.mapType(n->in(0)->type());
        return "%v" + std::to_string(n->id()) + " = arith.addi " +
               emit(n->in(0), ctx) + ", " + emit(n->in(1), ctx) + 
               " : " + lhs_type;
    },
    .priority = 0
};
```

## Type Mapping Across Targets

### Type Translation Table

```cpp
class TypeMapper {
    struct TypeMapping {
        std::string cName;
        std::string cppName;
        std::string cpp2Name;
        std::string mlirName;
    };
    
    std::unordered_map<Type*, TypeMapping> _mappings;
    
public:
    void registerStandardTypes() {
        // Integer types
        registerType(TypeInteger::I8, 
            {"int8_t", "std::int8_t", "i8", "i8"});
        registerType(TypeInteger::I16,
            {"int16_t", "std::int16_t", "i16", "i16"});
        registerType(TypeInteger::I32,
            {"int32_t", "std::int32_t", "i32", "i32"});
        registerType(TypeInteger::I64,
            {"int64_t", "std::int64_t", "i64", "i64"});
            
        // Float types
        registerType(TypeFloat::F32,
            {"float", "float", "f32", "f32"});
        registerType(TypeFloat::F64,
            {"double", "double", "f64", "f64"});
            
        // Pointer types handled dynamically
    }
    
    std::string map(Type* type, Target target) const {
        if (auto* ptr = dynamic_cast<TypePointer*>(type)) {
            return mapPointer(ptr, target);
        }
        
        auto it = _mappings.find(type);
        if (it == _mappings.end()) {
            return "void";  // Fallback
        }
        
        switch (target) {
            case Target::C: return it->second.cName;
            case Target::Cpp: return it->second.cppName;
            case Target::Cpp2: return it->second.cpp2Name;
            case Target::MLIR: return it->second.mlirName;
            default: return it->second.cppName;
        }
    }
    
private:
    std::string mapPointer(TypePointer* ptr, Target target) const {
        std::string base = map(ptr->pointeeType(), target);
        
        switch (target) {
            case Target::C:
            case Target::Cpp:
                return base + "*";
                
            case Target::Cpp2:
                return ptr->isNullable() ? 
                    base + "*" : 
                    base + "^";  // Non-null pointer syntax
                    
            case Target::MLIR:
                return "!llvm.ptr<" + base + ">";
                
            default:
                return base + "*";
        }
    }
};
```

## Emission Orchestration

### The Emitter Class

```cpp
class NWayEmitter {
    PatternRegistry _patterns;
    TypeMapper _typeMapper;
    
public:
    NWayEmitter() {
        registerBuiltinPatterns();
        _typeMapper.registerStandardTypes();
    }
    
    std::string emit(Node* root, Target target) {
        TargetContext ctx(target);
        
        // Emit preamble
        std::string output = emitPreamble(ctx);
        
        // Emit nodes in scheduled order
        auto schedule = computeSchedule(root);
        for (Node* node : schedule) {
            if (node->isVisible()) {  // Skip internal nodes
                std::string code = emitNode(node, ctx);
                output += ctx.indent() + code + "\n";
            }
        }
        
        // Emit postamble
        output += emitPostamble(ctx);
        
        return output;
    }
    
private:
    std::string emitPreamble(TargetContext& ctx) {
        std::string output;
        
        // Emit includes
        output += ctx.emitIncludes();
        
        // Target-specific preamble
        switch (ctx.target()) {
            case Target::Cpp2:
                output += "// Generated from Sea of Nodes IR\n";
                output += "#include <cpp2.h>\n\n";
                break;
                
            case Target::MLIR:
                output += "module {\n";
                ctx.pushIndent();
                break;
                
            default:
                break;
        }
        
        return output;
    }
    
    std::string emitNode(Node* node, TargetContext& ctx) {
        return _patterns.emit(node, ctx);
    }
    
    std::string emitPostamble(TargetContext& ctx) {
        std::string output;
        
        switch (ctx.target()) {
            case Target::MLIR:
                ctx.popIndent();
                output += "}\n";
                break;
                
            default:
                break;
        }
        
        return output;
    }
    
    std::vector<Node*> computeSchedule(Node* root);
};
```

### Scheduling Algorithm

```cpp
std::vector<Node*> NWayEmitter::computeSchedule(Node* root) {
    // Reverse post-order traversal for data flow
    std::vector<Node*> schedule;
    std::unordered_set<Node*> visited;
    
    std::function<void(Node*)> visit = [&](Node* node) {
        if (visited.insert(node).second) {
            // Visit dependencies first
            for (Node* input : node->inputs()) {
                if (!input->isCFG()) {  // Don't schedule control flow
                    visit(input);
                }
            }
            schedule.push_back(node);
        }
    };
    
    visit(root);
    return schedule;
}
```

## Integration with Existing Stage0

### Bidirectional Transpiler Extension

```cpp
class BidirectionalTranspiler {
    NWayEmitter _emitter;
    GraphBuilder _graphBuilder;
    
public:
    // Existing interface remains unchanged
    std::string emit_cpp(const TranslationUnit& unit, 
                        const TransformOptions& options);
    
    // New n-way emission interface
    std::string emit_to_target(const TranslationUnit& unit,
                              Target target,
                              const TransformOptions& options) {
        // Build Sea of Nodes from AST
        Node* graph = _graphBuilder.buildFromAST(unit);
        
        // Optimize graph
        optimizeGraph(graph);
        
        // Emit to target
        return _emitter.emit(graph, target);
    }
    
    // Emit to all targets simultaneously
    std::unordered_map<Target, std::string> 
    emit_all_targets(const TranslationUnit& unit) {
        Node* graph = _graphBuilder.buildFromAST(unit);
        optimizeGraph(graph);
        
        std::unordered_map<Target, std::string> results;
        for (Target t : {Target::C, Target::Cpp, Target::Cpp2, Target::MLIR}) {
            results[t] = _emitter.emit(graph, t);
        }
        return results;
    }
};
```

## Testing Strategy

### Cross-Target Validation

```cpp
void testCrossTargetConsistency(const std::string& source) {
    BidirectionalTranspiler transpiler;
    auto ast = transpiler.parse_cpp2(source, "test.cpp2");
    
    // Generate all targets
    auto targets = transpiler.emit_all_targets(ast);
    
    // Compile and execute each
    std::vector<int> results;
    for (const auto& [target, code] : targets) {
        if (target == Target::MLIR) continue;  // Can't execute MLIR directly
        
        std::string compiled = compile(code, target);
        int result = execute(compiled);
        results.push_back(result);
    }
    
    // All results should match
    for (size_t i = 1; i < results.size(); ++i) {
        assert(results[i] == results[0]);
    }
}
```

### Pattern Coverage Analysis

```cpp
void analyzePatternCoverage(const PatternRegistry& registry) {
    // Count patterns per (NodeKind, Target) pair
    std::map<std::pair<NodeKind, Target>, int> coverage;
    
    for (auto kind : allNodeKinds()) {
        for (auto target : allTargets()) {
            int count = registry.countPatterns(kind, target);
            coverage[{kind, target}] = count;
            
            // Warn if no patterns exist
            if (count == 0) {
                std::cerr << "Warning: No patterns for " 
                         << toString(kind) << " on " 
                         << toString(target) << "\n";
            }
        }
    }
}
```

## Performance Considerations

### Pattern Selection Optimization

```cpp
// Cache pattern lookup results
class PatternCache {
    std::unordered_map<
        std::pair<NodeKind, Target>,
        const EmissionPattern*
    > _cache;
    
public:
    const EmissionPattern* lookup(Node* node, Target target,
                                  const PatternRegistry& registry) {
        auto key = std::make_pair(node->kind(), target);
        
        auto it = _cache.find(key);
        if (it != _cache.end()) {
            return it->second;
        }
        
        // Find and cache best pattern
        const EmissionPattern* best = registry.findBest(node, target);
        _cache[key] = best;
        return best;
    }
};
```

## Next Steps

Chapter 20 establishes the n-way emission architecture. Subsequent chapters optimize:

- **Chapter 21:** Copy/move elision through dataflow analysis
- **Chapter 22:** Lifetime inference and automatic rescoping
- **Chapter 23:** Tblgen-based declarative specifications
- **Chapter 24:** Complete integration and performance tuning

## References

- Pattern matching in compilers: Dragon Book Ch. 9
- MLIR documentation: mlir.llvm.org
- Band 5 architecture: `docs/BAND5_ARCHITECTURE_SUMMARY.md`
