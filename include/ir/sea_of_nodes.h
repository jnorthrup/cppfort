#pragma once

#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <unordered_map>
#include <any>

// Mock IR interfaces for Sea of Nodes
// These provide the contracts that stages can use while real IR is developed

namespace cppfort::ir {

// Forward declarations
class Node;
class Graph;
class PatternMatcher;
class LoweringPass;

//==============================================================================
// Core Node Types (Sea of Nodes)
//==============================================================================

enum class NodeType {
    Constant,
    BinaryOp,
    UnaryOp,
    Phi,
    Control,
    Memory,
    Call,
    Load,
    Store
};

enum class BinaryOpType {
    Add, Sub, Mul, Div,
    And, Or, Xor,
    Shl, Shr,
    Eq, Ne, Lt, Le, Gt, Ge
};

enum class UnaryOpType {
    Neg, Not, Load, Store
};

//==============================================================================
// Node Interface
//==============================================================================

class Node {
public:
    virtual ~Node() = default;

    virtual NodeType type() const = 0;
    virtual std::string name() const = 0;
    virtual std::vector<Node*> inputs() const = 0;
    virtual std::vector<Node*> outputs() const = 0;

    // Mock implementation - returns empty
    virtual std::any value() const { return {}; }

    // Pattern matching support
    virtual bool matches(const PatternMatcher& pattern) const { return false; }
};

//==============================================================================
// Graph Interface (Sea of Nodes)
//==============================================================================

class Graph {
public:
    virtual ~Graph() = default;

    // Node management
    virtual Node* createConstant(int64_t value) = 0;
    virtual Node* createBinaryOp(BinaryOpType op, Node* lhs, Node* rhs) = 0;
    virtual Node* createUnaryOp(UnaryOpType op, Node* input) = 0;
    virtual Node* createPhi(std::vector<Node*> inputs) = 0;
    virtual Node* createControl() = 0;

    // Graph queries
    virtual std::vector<Node*> nodes() const = 0;
    virtual std::vector<Node*> roots() const = 0;

    // Optimization passes
    virtual void optimize() = 0;
    virtual bool applyPass(LoweringPass* pass) = 0;
};

//==============================================================================
// Pattern Matching Interface
//==============================================================================

class PatternMatcher {
public:
    virtual ~PatternMatcher() = default;

    virtual bool matches(const Node* node) const = 0;
    virtual std::vector<Node*> findMatches(const Graph* graph) const = 0;
    virtual std::function<void(Node*)> replacement() const = 0;
};

//==============================================================================
// Lowering Pass Interface
//==============================================================================

class LoweringPass {
public:
    virtual ~LoweringPass() = default;

    virtual std::string name() const = 0;
    virtual bool runOnGraph(Graph* graph) = 0;
    virtual std::vector<PatternMatcher*> patterns() const = 0;
};

//==============================================================================
// Target-Specific Lowering Interfaces
//==============================================================================

enum class Target {
    Cpp, MLIR, LLVM, WASM, Rust
};

class TargetLowering {
public:
    virtual ~TargetLowering() = default;

    virtual Target target() const = 0;
    virtual std::string emit(const Graph* graph) const = 0;
    virtual std::vector<LoweringPass*> passes() const = 0;
};

//==============================================================================
// Factory Functions (Mock Implementations)
//==============================================================================

// Create a mock graph implementation
std::unique_ptr<Graph> createMockGraph();

// Create target-specific lowering
std::unique_ptr<TargetLowering> createTargetLowering(Target target);

// Create common optimization passes
std::unique_ptr<LoweringPass> createConstantFoldingPass();
std::unique_ptr<LoweringPass> createDeadCodeEliminationPass();
std::unique_ptr<LoweringPass> createCommonSubexpressionEliminationPass();

//==============================================================================
// Utility Functions
//==============================================================================

// Convert AST to IR graph (mock implementation)
std::unique_ptr<Graph> astToIR(const std::any& ast);

// Validate IR graph structure
bool validateGraph(const Graph* graph);

// Debug utilities
void dumpGraph(const Graph* graph, std::ostream& out);
std::string graphToDot(const Graph* graph);

} // namespace cppfort::ir