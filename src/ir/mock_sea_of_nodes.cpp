#include "sea_of_nodes.h"
#include <iostream>
#include <sstream>
#include <unordered_set>

// Mock IR implementation for Sea of Nodes
// Provides stub implementations that stages can use during development

namespace cppfort::ir {

//==============================================================================
// Mock Node Implementation
//==============================================================================

class MockNode : public Node {
public:
    MockNode(NodeType t, const std::string& n) : node_type_(t), node_name_(n) {}

    NodeType type() const override { return node_type_; }
    std::string name() const override { return node_name_; }

    std::vector<Node*> inputs() const override { return inputs_; }
    std::vector<Node*> outputs() const override { return outputs_; }

    void addInput(Node* node) { inputs_.push_back(node); }
    void addOutput(Node* node) { outputs_.push_back(node); }

private:
    NodeType node_type_;
    std::string node_name_;
    std::vector<Node*> inputs_;
    std::vector<Node*> outputs_;
};

//==============================================================================
// Mock Graph Implementation
//==============================================================================

class MockGraph : public Graph {
public:
    ~MockGraph() {
        for (auto* node : nodes_) {
            delete node;
        }
    }

    Node* createConstant(int64_t value) override {
        auto* node = new MockNode(NodeType::Constant, "const_" + std::to_string(value));
        nodes_.push_back(node);
        return node;
    }

    Node* createBinaryOp(BinaryOpType op, Node* lhs, Node* rhs) override {
        std::string op_name;
        switch (op) {
            case BinaryOpType::Add: op_name = "add"; break;
            case BinaryOpType::Sub: op_name = "sub"; break;
            case BinaryOpType::Mul: op_name = "mul"; break;
            case BinaryOpType::Div: op_name = "div"; break;
            default: op_name = "binop"; break;
        }

        auto* node = new MockNode(NodeType::BinaryOp, op_name);
        if (lhs) static_cast<MockNode*>(node)->addInput(lhs);
        if (rhs) static_cast<MockNode*>(node)->addInput(rhs);
        nodes_.push_back(node);
        return node;
    }

    Node* createUnaryOp(UnaryOpType op, Node* input) override {
        std::string op_name;
        switch (op) {
            case UnaryOpType::Neg: op_name = "neg"; break;
            case UnaryOpType::Not: op_name = "not"; break;
            default: op_name = "unop"; break;
        }

        auto* node = new MockNode(NodeType::UnaryOp, op_name);
        if (input) static_cast<MockNode*>(node)->addInput(input);
        nodes_.push_back(node);
        return node;
    }

    Node* createPhi(std::vector<Node*> inputs) override {
        auto* node = new MockNode(NodeType::Phi, "phi");
        for (auto* input : inputs) {
            if (input) static_cast<MockNode*>(node)->addInput(input);
        }
        nodes_.push_back(node);
        return node;
    }

    Node* createControl() override {
        auto* node = new MockNode(NodeType::Control, "control");
        nodes_.push_back(node);
        return node;
    }

    std::vector<Node*> nodes() const override { return nodes_; }
    std::vector<Node*> roots() const override { return {nodes_.empty() ? nullptr : nodes_[0]}; }

    void optimize() override {
        // Mock optimization - just log
        std::cout << "MockGraph: Running optimization passes\n";
    }

    bool applyPass(LoweringPass* pass) override {
        // Mock pass application
        std::cout << "MockGraph: Applying pass " << pass->name() << "\n";
        return true;
    }

private:
    std::vector<Node*> nodes_;
};

//==============================================================================
// Mock Pattern Matcher
//==============================================================================

class MockPatternMatcher : public PatternMatcher {
public:
    MockPatternMatcher(const std::string& pattern_name) : name_(pattern_name) {}

    bool matches(const Node* node) const override {
        // Mock matching - always return false for now
        return false;
    }

    std::vector<Node*> findMatches(const Graph* graph) const override {
        // Mock - return empty matches
        return {};
    }

    std::function<void(Node*)> replacement() const override {
        // Mock replacement - do nothing
        return [](Node*) {};
    }

private:
    std::string name_;
};

//==============================================================================
// Mock Lowering Pass
//==============================================================================

class MockLoweringPass : public LoweringPass {
public:
    MockLoweringPass(const std::string& pass_name) : name_(pass_name) {}

    std::string name() const override { return name_; }

    bool runOnGraph(Graph* graph) override {
        std::cout << "MockLoweringPass: Running " << name_ << "\n";
        return true;
    }

    std::vector<PatternMatcher*> patterns() const override {
        return {&mock_pattern_};
    }

private:
    std::string name_;
    mutable MockPatternMatcher mock_pattern_{name_ + "_pattern"};
};

//==============================================================================
// Mock Target Lowering
//==============================================================================

class MockTargetLowering : public TargetLowering {
public:
    MockTargetLowering(Target t) : target_(t) {}

    Target target() const override { return target_; }

    std::string emit(const Graph* graph) const override {
        std::ostringstream out;
        out << "// Mock emission for target " << static_cast<int>(target_) << "\n";
        out << "// Graph has " << graph->nodes().size() << " nodes\n";

        // Generate mock C++ code
        if (target_ == Target::Cpp) {
            out << "int mock_function() {\n";
            out << "    return 42; // Mock implementation\n";
            out << "}\n";
        }

        return out.str();
    }

    std::vector<LoweringPass*> passes() const override {
        return {&constant_fold_, &dce_};
    }

private:
    Target target_;
    mutable MockLoweringPass constant_fold_{"constant-folding"};
    mutable MockLoweringPass dce_{"dead-code-elimination"};
};

//==============================================================================
// Factory Implementations
//==============================================================================

std::unique_ptr<Graph> createMockGraph() {
    return std::make_unique<MockGraph>();
}

std::unique_ptr<TargetLowering> createTargetLowering(Target target) {
    return std::make_unique<MockTargetLowering>(target);
}

std::unique_ptr<LoweringPass> createConstantFoldingPass() {
    return std::make_unique<MockLoweringPass>("constant-folding");
}

std::unique_ptr<LoweringPass> createDeadCodeEliminationPass() {
    return std::make_unique<MockLoweringPass>("dead-code-elimination");
}

std::unique_ptr<LoweringPass> createCommonSubexpressionEliminationPass() {
    return std::make_unique<MockLoweringPass>("cse");
}

//==============================================================================
// Utility Functions
//==============================================================================

std::unique_ptr<Graph> astToIR(const std::any& ast) {
    // Mock AST to IR conversion
    std::cout << "Mock IR: Converting AST to Sea of Nodes graph\n";
    auto graph = createMockGraph();

    // Create some mock nodes
    auto* const1 = graph->createConstant(42);
    auto* const2 = graph->createConstant(24);
    auto* add = graph->createBinaryOp(BinaryOpType::Add, const1, const2);

    return graph;
}

bool validateGraph(const Graph* graph) {
    // Mock validation - always pass
    std::cout << "Mock IR: Validating graph structure\n";
    return true;
}

void dumpGraph(const Graph* graph, std::ostream& out) {
    out << "Mock IR Graph Dump:\n";
    out << "Nodes: " << graph->nodes().size() << "\n";
    for (const auto* node : graph->nodes()) {
        out << "  " << node->name() << " (" << static_cast<int>(node->type()) << ")\n";
    }
}

std::string graphToDot(const Graph* graph) {
    std::ostringstream out;
    out << "digraph IR {\n";
    for (const auto* node : graph->nodes()) {
        out << "  " << node->name() << ";\n";
    }
    out << "}\n";
    return out.str();
}

} // namespace cppfort::ir