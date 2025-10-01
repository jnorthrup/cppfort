# Chapter 23 Implementation: Tblgen-Based Declarative DAG Specifications

## Overview

Chapter 23 implements **tablegen-style declarative specifications** for defining Sea of Nodes IR patterns, transformations, and lowering rules. This approach, inspired by LLVM's TableGen and MLIR's operation definitions, enables:

1. Declarative node type definitions
2. Pattern-based optimization rules
3. Multi-target lowering specifications
4. Automatic code generation for IR infrastructure

The goal is to reduce manual C++ boilerplate and make the n-way transpiler extensible through data files rather than code changes.

## Tblgen Design Philosophy

### Why Tablegen?

Traditional compiler implementations require extensive C++ boilerplate:

```cpp
// Without tblgen: 50+ lines per node type
class AddNode : public Node {
public:
    AddNode(Node* lhs, Node* rhs);
    std::string emitC() const override { /* ... */ }
    std::string emitCpp() const override { /* ... */ }
    std::string emitCpp2() const override { /* ... */ }
    std::string emitMLIR() const override { /* ... */ }
    bool isCommutative() const override { return true; }
    Node* fold() const override { /* ... */ }
    // ... more methods
};
```

With tblgen, the same node becomes:

```tablegen
// With tblgen: 10 lines of declarative spec
def AddNode : BinaryArithmeticNode<"add"> {
    let properties = [Commutative, Associative];
    let cEmit = "$lhs + $rhs";
    let cppEmit = "$lhs + $rhs";
    let cpp2Emit = "$lhs + $rhs";
    let mlirEmit = "arith.addi $lhs, $rhs : $type";
    let foldPattern = [(add $x, 0) -> $x,
                       (add 0, $x) -> $x];
}
```

## Tblgen Language Design

### Node Definition Syntax

```tablegen
// Base class for all nodes
class Node<string opcode> {
    string kind = opcode;
    list<Type> inputTypes = [];
    Type outputType;
    list<NodeProperty> properties = [];
}

// Properties that can be attached to nodes
def Commutative : NodeProperty;
def Associative : NodeProperty;
def Idempotent : NodeProperty;
def Pure : NodeProperty;          // No side effects
def Foldable : NodeProperty;      // Can constant-fold
def Moveable : NodeProperty;      // Can reorder with other ops

// Arithmetic operations
class BinaryArithmeticNode<string op> : Node<op> {
    let inputTypes = [AnyInteger, AnyInteger];
    let outputType = SameAsInput<0>;
    let properties = [Pure, Foldable];
}

def Add : BinaryArithmeticNode<"add"> {
    let properties = !listconcat(properties, [Commutative, Associative]);
}

def Sub : BinaryArithmeticNode<"sub">;

def Mul : BinaryArithmeticNode<"mul"> {
    let properties = !listconcat(properties, [Commutative, Associative]);
}

def Div : BinaryArithmeticNode<"div"> {
    let inputTypes = [AnyInteger, NonZeroInteger];
}

// Memory operations
def Load : Node<"load"> {
    let inputTypes = [AnyPointer, MemoryState];
    let outputType = PointeeType<0>;
    let properties = [];  // Not pure - has side effects
}

def Store : Node<"store"> {
    let inputTypes = [AnyPointer, AnyType, MemoryState];
    let outputType = MemoryState;
    let properties = [];
}

// Control flow
def If : Node<"if"> {
    let inputTypes = [BooleanType, ControlFlow];
    let outputType = ControlFlow;
    let properties = [];
}

def Region : Node<"region"> {
    let inputTypes = !listsplat(ControlFlow, VariadicCount);
    let outputType = ControlFlow;
    let properties = [];
}
```

### Target-Specific Emission

```tablegen
// Emission patterns for different targets
class EmitPattern<string target, string pattern> {
    string targetName = target;
    string codePattern = pattern;
}

def Add : BinaryArithmeticNode<"add"> {
    let emitPatterns = [
        EmitPattern<"C", "($lhs + $rhs)">,
        EmitPattern<"Cpp", "($lhs + $rhs)">,
        EmitPattern<"Cpp2", "($lhs + $rhs)">,
        EmitPattern<"MLIR", "%$result = arith.addi $lhs, $rhs : $type">
    ];
}

def Mul : BinaryArithmeticNode<"mul"> {
    let emitPatterns = [
        EmitPattern<"C", "($lhs * $rhs)">,
        EmitPattern<"Cpp", "($lhs * $rhs)">,
        EmitPattern<"Cpp2", "($lhs * $rhs)">,
        EmitPattern<"MLIR", "%$result = arith.muli $lhs, $rhs : $type">
    ];
    
    // Special case: power-of-two optimization
    let optimizedEmit = [
        EmitPattern<"C", "($lhs << $log2rhs)", 
                   Predicate="isPowerOfTwo($rhs)">,
        EmitPattern<"Cpp", "($lhs << $log2rhs)",
                   Predicate="isPowerOfTwo($rhs)">
    ];
}
```

### Pattern Matching and Rewriting

```tablegen
// Pattern-based optimization rules
class Pattern<dag source, dag target> {
    dag sourcePattern = source;
    dag targetPattern = target;
}

// Algebraic simplifications
def : Pattern<(add $x, 0), $x>;                    // x + 0 → x
def : Pattern<(add 0, $x), $x>;                    // 0 + x → x
def : Pattern<(mul $x, 1), $x>;                    // x * 1 → x
def : Pattern<(mul $x, 0), 0>;                     // x * 0 → 0
def : Pattern<(sub $x, 0), $x>;                    // x - 0 → x
def : Pattern<(div $x, 1), $x>;                    // x / 1 → x

// Strength reduction
def : Pattern<(mul $x, (const 2)), (shl $x, 1)>;   // x * 2 → x << 1
def : Pattern<(div $x, (const 2)), (ashr $x, 1)>;  // x / 2 → x >> 1

// Reassociation
def : Pattern<(add (add $x, $c1), $c2),            // (x + c1) + c2 
              (add $x, (const (evalAdd $c1, $c2)))>; // → x + (c1+c2)

// Conditional optimizations
def : Pattern<(if (not $x), $then, $else),
              (if $x, $else, $then)>;              // Invert condition

// Load forwarding
def : Pattern<(load $addr, (store $addr, $val, $mem)),
              $val>;                               // Store then load → value
```

## Tblgen Code Generation

### C++ Code Generator

```cpp
class TblgenCodeGenerator {
public:
    void generateFromSpecs(const std::string& tblgenFile) {
        // Parse tblgen file
        TblgenParser parser;
        auto specs = parser.parse(tblgenFile);
        
        // Generate node classes
        generateNodeClasses(specs.nodes);
        
        // Generate pattern matcher
        generatePatternMatcher(specs.patterns);
        
        // Generate emission code
        generateEmitters(specs.emitPatterns);
        
        // Generate factory methods
        generateFactories(specs.nodes);
    }
    
private:
    void generateNodeClasses(const std::vector<NodeSpec>& nodes) {
        std::ofstream header("generated/nodes.h");
        std::ofstream impl("generated/nodes.cpp");
        
        header << "#pragma once\n"
               << "#include \"node.h\"\n\n"
               << "namespace cppfort::stage0::generated {\n\n";
        
        for (const auto& nodeSpec : nodes) {
            generateNodeClass(nodeSpec, header, impl);
        }
        
        header << "} // namespace\n";
    }
    
    void generateNodeClass(const NodeSpec& spec,
                          std::ofstream& header,
                          std::ofstream& impl) {
        // Generate class declaration
        header << "class " << spec.name << "Node : public Node {\n"
               << "public:\n";
        
        // Constructor
        header << "    " << spec.name << "Node(";
        for (size_t i = 0; i < spec.inputTypes.size(); ++i) {
            if (i > 0) header << ", ";
            header << "Node* in" << i;
        }
        header << ");\n";
        
        // Emit methods
        for (const auto& pattern : spec.emitPatterns) {
            header << "    std::string emit" 
                   << pattern.targetName 
                   << "() const override;\n";
        }
        
        // Properties as methods
        if (spec.hasProperty("Commutative")) {
            header << "    bool isCommutative() const override "
                   << "{ return true; }\n";
        }
        if (spec.hasProperty("Pure")) {
            header << "    bool isPure() const override "
                   << "{ return true; }\n";
        }
        
        header << "};\n\n";
        
        // Generate implementation
        generateNodeImpl(spec, impl);
    }
    
    void generateNodeImpl(const NodeSpec& spec, std::ofstream& impl) {
        impl << spec.name << "Node::" << spec.name << "Node(";
        for (size_t i = 0; i < spec.inputTypes.size(); ++i) {
            if (i > 0) impl << ", ";
            impl << "Node* in" << i;
        }
        impl << ") : Node(NodeKind::" << toUpperCase(spec.name) << ") {\n";
        
        for (size_t i = 0; i < spec.inputTypes.size(); ++i) {
            impl << "    addInput(in" << i << ");\n";
        }
        
        impl << "}\n\n";
        
        // Generate emit methods
        for (const auto& pattern : spec.emitPatterns) {
            impl << "std::string " << spec.name << "Node::emit"
                 << pattern.targetName << "() const {\n";
            impl << "    return " << generateEmitCode(pattern) << ";\n";
            impl << "}\n\n";
        }
    }
    
    std::string generateEmitCode(const EmitPattern& pattern) const {
        std::string code = "\"" + pattern.codePattern + "\"";
        
        // Replace placeholders
        code = replaceAll(code, "$lhs", "\" + in(0)->emit" + 
                         pattern.targetName + "() + \"");
        code = replaceAll(code, "$rhs", "\" + in(1)->emit" + 
                         pattern.targetName + "() + \"");
        code = replaceAll(code, "$result", 
                         "\" + std::to_string(id()) + \"");
        code = replaceAll(code, "$type", 
                         "\" + type()->name() + \"");
        
        return code;
    }
};
```

### Pattern Matcher Generator

```cpp
class PatternMatcherGenerator {
public:
    void generate(const std::vector<Pattern>& patterns) {
        std::ofstream out("generated/pattern_matcher.cpp");
        
        out << "#include \"pattern_matcher.h\"\n\n"
            << "namespace cppfort::stage0::generated {\n\n"
            << "Node* PatternMatcher::match(Node* node) {\n"
            << "    // Try each pattern in order\n";
        
        for (size_t i = 0; i < patterns.size(); ++i) {
            generatePatternMatch(patterns[i], i, out);
        }
        
        out << "    return nullptr;  // No match\n"
            << "}\n\n"
            << "} // namespace\n";
    }
    
private:
    void generatePatternMatch(const Pattern& pattern, 
                             size_t index,
                             std::ofstream& out) {
        out << "    // Pattern " << index << ": "
            << pattern.description() << "\n";
        out << "    if (";
        
        generateMatchCondition(pattern.source, "node", out);
        
        out << ") {\n";
        generateRewrite(pattern.target, out);
        out << "    }\n\n";
    }
    
    void generateMatchCondition(const DAGPattern& pattern,
                               const std::string& nodeVar,
                               std::ofstream& out) {
        out << nodeVar << "->kind() == NodeKind::" 
            << toUpperCase(pattern.op);
        
        // Match operands recursively
        for (size_t i = 0; i < pattern.operands.size(); ++i) {
            out << " && ";
            
            if (pattern.operands[i].isConstant()) {
                out << nodeVar << "->in(" << i << ")->kind() == "
                    << "NodeKind::CONSTANT && "
                    << nodeVar << "->in(" << i << ")->asConstant()->value() == "
                    << pattern.operands[i].constantValue;
            } else if (pattern.operands[i].isWildcard()) {
                // Always matches
                out << "true";
            } else {
                // Nested pattern
                std::string childVar = nodeVar + "_child" + std::to_string(i);
                out << "(Node* " << childVar << " = " << nodeVar 
                    << "->in(" << i << ")) && ";
                generateMatchCondition(pattern.operands[i], childVar, out);
            }
        }
    }
    
    void generateRewrite(const DAGPattern& pattern, std::ofstream& out) {
        out << "        // Rewrite to: " << pattern.toString() << "\n";
        
        if (pattern.isVariable()) {
            // Identity: just return the matched variable
            out << "        return " << pattern.variableName() << ";\n";
        } else if (pattern.isConstant()) {
            out << "        return new ConstantNode(" 
                << pattern.constantValue << ");\n";
        } else {
            // Create new node
            out << "        auto* result = new " 
                << pattern.op << "Node(";
            
            for (size_t i = 0; i < pattern.operands.size(); ++i) {
                if (i > 0) out << ", ";
                generateRewrite(pattern.operands[i], out);
            }
            
            out << ");\n"
                << "        return result;\n";
        }
    }
};
```

## Tblgen Runtime System

### Pattern Matching Engine

```cpp
class PatternEngine {
    std::vector<CompiledPattern> _patterns;
    
public:
    void loadPatterns(const std::string& tblgenFile) {
        TblgenParser parser;
        auto specs = parser.parse(tblgenFile);
        
        // Compile patterns into efficient matchers
        for (const auto& pattern : specs.patterns) {
            _patterns.push_back(compilePattern(pattern));
        }
        
        // Sort by specificity (more specific first)
        std::sort(_patterns.begin(), _patterns.end(),
            [](const auto& a, const auto& b) {
                return a.specificity > b.specificity;
            });
    }
    
    Node* applyPatterns(Node* node) {
        // Try each pattern until one matches
        for (const auto& pattern : _patterns) {
            if (auto* rewritten = pattern.tryMatch(node)) {
                return rewritten;
            }
        }
        return node;  // No pattern matched
    }
    
private:
    CompiledPattern compilePattern(const Pattern& pattern) {
        // Compile pattern into fast executable form
        CompiledPattern compiled;
        compiled.specificity = computeSpecificity(pattern);
        compiled.matcher = generateMatcher(pattern.source);
        compiled.rewriter = generateRewriter(pattern.target);
        return compiled;
    }
    
    int computeSpecificity(const Pattern& pattern) {
        // More specific patterns have higher scores
        int score = 0;
        score += pattern.source.countConstants() * 10;
        score += pattern.source.countSpecificTypes() * 5;
        score += pattern.source.depth() * 2;
        return score;
    }
};
```

### Type Constraint System

```tablegen
// Type constraint definitions
class TypeConstraint<string desc> {
    string description = desc;
}

def AnyType : TypeConstraint<"matches any type">;
def AnyInteger : TypeConstraint<"matches any integer type">;
def AnyFloat : TypeConstraint<"matches any floating-point type">;
def BooleanType : TypeConstraint<"matches boolean type">;
def AnyPointer : TypeConstraint<"matches any pointer type">;

// Derived constraints
def NonZeroInteger : TypeConstraint<"integer constant != 0"> {
    code constraint = [{
        return node->kind() == NodeKind::CONSTANT &&
               node->asConstant()->value() != 0;
    }];
}

def PowerOfTwo : TypeConstraint<"integer constant that is power of 2"> {
    code constraint = [{
        int64_t val = node->asConstant()->value();
        return val > 0 && (val & (val - 1)) == 0;
    }];
}

def SameAsInput<int n> : TypeConstraint<"same type as input " # n> {
    code constraint = [{
        return outputType == inputTypes[n];
    }];
}
```

## Integration with Existing Stage0

### Bridging Generated and Manual Code

```cpp
// In src/stage0/node_factory.h
class NodeFactory {
    // Hand-written factory methods for complex nodes
    Node* createComplexNode(/* ... */);
    
    // Generated factory methods from tblgen
    #include "generated/node_factories.inc"
    
public:
    Node* create(NodeKind kind, const std::vector<Node*>& inputs) {
        // Dispatch to generated or manual factory
        switch (kind) {
            #include "generated/node_factory_dispatch.inc"
            
            default:
                return createManual(kind, inputs);
        }
    }
};
```

### Extending with Custom Patterns

```cpp
// Allow mixing tblgen and manual patterns
class HybridPatternMatcher {
    PatternEngine _tblgenPatterns;
    std::vector<ManualPattern> _manualPatterns;
    
public:
    void loadTblgen(const std::string& file) {
        _tblgenPatterns.loadPatterns(file);
    }
    
    void registerManual(ManualPattern pattern) {
        _manualPatterns.push_back(std::move(pattern));
    }
    
    Node* match(Node* node) {
        // Try manual patterns first (more specific)
        for (const auto& pattern : _manualPatterns) {
            if (auto* result = pattern.tryMatch(node)) {
                return result;
            }
        }
        
        // Fall back to tblgen patterns
        return _tblgenPatterns.applyPatterns(node);
    }
};
```

## Build System Integration

### CMake Integration

```cmake
# Find tblgen tool
find_program(TBLGEN_EXECUTABLE
    NAMES cppfort-tblgen llvm-tblgen
    PATHS ${CMAKE_BINARY_DIR}/tools)

# Function to generate code from tblgen files
function(add_tblgen_target name input output)
    add_custom_command(
        OUTPUT ${output}
        COMMAND ${TBLGEN_EXECUTABLE} 
                -I ${CMAKE_SOURCE_DIR}/include
                ${input} 
                -o ${output}
        DEPENDS ${input}
        COMMENT "Generating ${output} from ${input}"
    )
    
    add_custom_target(${name} DEPENDS ${output})
endfunction()

# Generate node definitions
add_tblgen_target(
    gen_nodes
    ${CMAKE_SOURCE_DIR}/specs/nodes.td
    ${CMAKE_BINARY_DIR}/generated/nodes.inc
)

# Generate patterns
add_tblgen_target(
    gen_patterns
    ${CMAKE_SOURCE_DIR}/specs/patterns.td
    ${CMAKE_BINARY_DIR}/generated/patterns.inc
)

# Main library depends on generated code
add_library(stage0 ${STAGE0_SOURCES})
add_dependencies(stage0 gen_nodes gen_patterns)
target_include_directories(stage0 PRIVATE ${CMAKE_BINARY_DIR}/generated)
```

## Testing Generated Code

### Validating Tblgen Output

```cpp
void testGeneratedPatterns() {
    // Load patterns
    PatternEngine engine;
    engine.loadPatterns("specs/patterns.td");
    
    // Test each pattern
    {
        // Test: (add $x, 0) → $x
        Node* x = new ParameterNode(0, TypeInteger::I32);
        Node* zero = new ConstantNode(0, TypeInteger::I32);
        Node* add = new AddNode(x, zero);
        
        Node* result = engine.applyPatterns(add);
        assert(result == x);  // Should eliminate addition
    }
    
    {
        // Test: (mul $x, 2) → (shl $x, 1)
        Node* x = new ParameterNode(0, TypeInteger::I32);
        Node* two = new ConstantNode(2, TypeInteger::I32);
        Node* mul = new MulNode(x, two);
        
        Node* result = engine.applyPatterns(mul);
        assert(result->kind() == NodeKind::SHL);
        assert(result->in(0) == x);
        assert(result->in(1)->asConstant()->value() == 1);
    }
}
```

## Next Steps

- **Chapter 24:** Complete integration, optimization pipeline, and performance validation

## References

- LLVM TableGen: llvm.org/docs/TableGen
- MLIR ODS: mlir.llvm.org/docs/OpDefinitions
- Pattern matching: Dragon Book Ch. 8
