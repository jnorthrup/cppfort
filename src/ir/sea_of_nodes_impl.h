#pragma once

#include "sea_of_nodes.h"
#include <set>
#include <queue>
#include <map>
#include <algorithm>

namespace cppfort::ir {

//==============================================================================
// Real Node Implementation
//==============================================================================

class NodeImpl : public Node {
public:
    NodeImpl(NodeType t, const std::string& n, int id);
    virtual ~NodeImpl() = default;

    NodeType type() const override { return node_type_; }
    std::string name() const override { return node_name_; }
    std::vector<Node*> inputs() const override;
    std::vector<Node*> outputs() const override;
    std::any value() const override { return value_; }

    // Edge management
    void addInput(NodeImpl* node);
    void addOutput(NodeImpl* node);
    void removeInput(NodeImpl* node);
    void removeOutput(NodeImpl* node);
    void replaceInput(NodeImpl* old_node, NodeImpl* new_node);

    // Control flow
    bool isControl() const { return node_type_ == NodeType::Control; }
    bool isPhi() const { return node_type_ == NodeType::Phi; }

    // Data flow
    NodeImpl* getInput(size_t index) const;
    size_t inputCount() const { return inputs_.size(); }
    size_t outputCount() const { return outputs_.size(); }

    // Node identification
    int id() const { return id_; }

    // Value management
    void setValue(const std::any& val) { value_ = val; }

    // Dead code detection
    bool isDead() const { return outputs_.empty() && !isControl(); }

    // Pattern matching support
    bool matches(const PatternMatcher& pattern) const override;

protected:
    NodeType node_type_;
    std::string node_name_;
    int id_;
    std::any value_;

    // Edges: inputs are operands, outputs are uses
    std::vector<NodeImpl*> inputs_;
    std::vector<NodeImpl*> outputs_;
};

//==============================================================================
// Specialized Node Types
//==============================================================================

class ConstantNode : public NodeImpl {
public:
    ConstantNode(int64_t val, int id);
    int64_t getConstant() const;
};

class BinaryOpNode : public NodeImpl {
public:
    BinaryOpNode(BinaryOpType op, NodeImpl* lhs, NodeImpl* rhs, int id);
    BinaryOpType getOp() const { return op_; }

private:
    BinaryOpType op_;
};

class UnaryOpNode : public NodeImpl {
public:
    UnaryOpNode(UnaryOpType op, NodeImpl* input, int id);
    UnaryOpType getOp() const { return op_; }

private:
    UnaryOpType op_;
};

class PhiNode : public NodeImpl {
public:
    PhiNode(const std::vector<NodeImpl*>& inputs, int id);
};

class ControlNode : public NodeImpl {
public:
    ControlNode(const std::string& name, int id);
};

class RegionNode : public ControlNode {
public:
    RegionNode(int id);
    void addPredecessor(NodeImpl* pred);
};

class ProjectionNode : public NodeImpl {
public:
    ProjectionNode(NodeImpl* input, int index, int id);
    int getIndex() const { return proj_index_; }

private:
    int proj_index_;
};

//==============================================================================
// Real Graph Implementation
//==============================================================================

class GraphImpl : public Graph {
public:
    GraphImpl();
    ~GraphImpl() override;

    // Node creation (Graph interface)
    Node* createConstant(int64_t value) override;
    Node* createBinaryOp(BinaryOpType op, Node* lhs, Node* rhs) override;
    Node* createUnaryOp(UnaryOpType op, Node* input) override;
    Node* createPhi(std::vector<Node*> inputs) override;
    Node* createControl() override;

    // Extended node creation
    NodeImpl* createRegion();
    NodeImpl* createProjection(NodeImpl* input, int index);

    // Graph queries
    std::vector<Node*> nodes() const override;
    std::vector<Node*> roots() const override;

    // Optimization
    void optimize() override;
    bool applyPass(LoweringPass* pass) override;

    // Graph analysis
    void computeDominance();
    bool dominates(NodeImpl* a, NodeImpl* b) const;
    NodeImpl* immediateDominator(NodeImpl* node) const;

    // Scheduling
    void schedule();
    std::vector<NodeImpl*> getScheduledNodes() const { return scheduled_nodes_; }

    // Dead code elimination
    void eliminateDeadCode();

    // Constant folding
    bool foldConstants();

    // Common subexpression elimination
    bool eliminateCommonSubexpressions();

    // Validation
    bool validate() const;

    // Debug
    void dump(std::ostream& out) const;
    std::string toDot() const;

private:
    std::vector<NodeImpl*> all_nodes_;
    std::map<int, NodeImpl*> node_map_;
    int next_id_ = 0;

    // Analysis results
    std::map<NodeImpl*, NodeImpl*> idom_; // Immediate dominators
    std::map<NodeImpl*, std::set<NodeImpl*>> dominance_tree_;
    std::vector<NodeImpl*> scheduled_nodes_;

    // Internal helpers
    int allocateId() { return next_id_++; }
    void addNode(NodeImpl* node);
    void removeNode(NodeImpl* node);

    // Analysis helpers
    void buildDominanceTree();
    NodeImpl* findLCA(NodeImpl* a, NodeImpl* b) const; // Lowest common ancestor in dom tree

    // Scheduling helpers
    void scheduleEarly();
    void scheduleLate();
    int computeDepth(NodeImpl* node);

    // Optimization helpers
    NodeImpl* tryFoldBinaryOp(BinaryOpNode* node);
    bool areEquivalent(NodeImpl* a, NodeImpl* b) const;
};

//==============================================================================
// Pattern Matcher Implementation
//==============================================================================

class PatternMatcherImpl : public PatternMatcher {
public:
    PatternMatcherImpl(const std::string& name,
                       std::function<bool(const Node*)> matcher,
                       std::function<void(Node*)> replacer);

    bool matches(const Node* node) const override;
    std::vector<Node*> findMatches(const Graph* graph) const override;
    std::function<void(Node*)> replacement() const override { return replacer_; }

private:
    std::string name_;
    std::function<bool(const Node*)> matcher_;
    std::function<void(Node*)> replacer_;
};

//==============================================================================
// Lowering Pass Implementation
//==============================================================================

class LoweringPassImpl : public LoweringPass {
public:
    LoweringPassImpl(const std::string& name,
                     std::vector<std::unique_ptr<PatternMatcher>> patterns);

    std::string name() const override { return name_; }
    bool runOnGraph(Graph* graph) override;
    std::vector<PatternMatcher*> patterns() const override;

private:
    std::string name_;
    std::vector<std::unique_ptr<PatternMatcher>> patterns_;
};

//==============================================================================
// Target Lowering Implementation
//==============================================================================

class TargetLoweringImpl : public TargetLowering {
public:
    TargetLoweringImpl(Target t);

    Target target() const override { return target_; }
    std::string emit(const Graph* graph) const override;
    std::vector<LoweringPass*> passes() const override;

private:
    Target target_;
    std::vector<std::unique_ptr<LoweringPass>> passes_;

    // Emission helpers
    std::string emitCpp(const GraphImpl* graph) const;
    std::string emitMLIR(const GraphImpl* graph) const;
};

} // namespace cppfort::ir
