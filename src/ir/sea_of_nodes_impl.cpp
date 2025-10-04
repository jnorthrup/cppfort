#include "sea_of_nodes_impl.h"
#include <sstream>
#include <unordered_set>
#include <algorithm>
#include <iostream>

namespace cppfort::ir {

//==============================================================================
// NodeImpl Implementation
//==============================================================================

NodeImpl::NodeImpl(NodeType t, const std::string& n, int id)
    : node_type_(t), node_name_(n), id_(id) {}

std::vector<Node*> NodeImpl::inputs() const {
    std::vector<Node*> result;
    result.reserve(inputs_.size());
    for (auto* node : inputs_) {
        result.push_back(node);
    }
    return result;
}

std::vector<Node*> NodeImpl::outputs() const {
    std::vector<Node*> result;
    result.reserve(outputs_.size());
    for (auto* node : outputs_) {
        result.push_back(node);
    }
    return result;
}

void NodeImpl::addInput(NodeImpl* node) {
    if (node) {
        inputs_.push_back(node);
        node->outputs_.push_back(this);
    }
}

void NodeImpl::addOutput(NodeImpl* node) {
    if (node) {
        outputs_.push_back(node);
    }
}

void NodeImpl::removeInput(NodeImpl* node) {
    auto it = std::find(inputs_.begin(), inputs_.end(), node);
    if (it != inputs_.end()) {
        inputs_.erase(it);
        // Remove this from node's outputs
        auto out_it = std::find(node->outputs_.begin(), node->outputs_.end(), this);
        if (out_it != node->outputs_.end()) {
            node->outputs_.erase(out_it);
        }
    }
}

void NodeImpl::removeOutput(NodeImpl* node) {
    auto it = std::find(outputs_.begin(), outputs_.end(), node);
    if (it != outputs_.end()) {
        outputs_.erase(it);
    }
}

void NodeImpl::replaceInput(NodeImpl* old_node, NodeImpl* new_node) {
    for (size_t i = 0; i < inputs_.size(); i++) {
        if (inputs_[i] == old_node) {
            inputs_[i] = new_node;
            // Update output edges
            old_node->removeOutput(this);
            if (new_node) {
                new_node->addOutput(this);
            }
        }
    }
}

NodeImpl* NodeImpl::getInput(size_t index) const {
    if (index < inputs_.size()) {
        return inputs_[index];
    }
    return nullptr;
}

bool NodeImpl::matches(const PatternMatcher& pattern) const {
    return pattern.matches(this);
}

//==============================================================================
// Specialized Node Implementations
//==============================================================================

ConstantNode::ConstantNode(int64_t val, int id)
    : NodeImpl(NodeType::Constant, "const_" + std::to_string(val), id) {
    setValue(val);
}

int64_t ConstantNode::getConstant() const {
    return std::any_cast<int64_t>(value_);
}

BinaryOpNode::BinaryOpNode(BinaryOpType op, NodeImpl* lhs, NodeImpl* rhs, int id)
    : NodeImpl(NodeType::BinaryOp, "binop", id), op_(op) {
    addInput(lhs);
    addInput(rhs);
}

UnaryOpNode::UnaryOpNode(UnaryOpType op, NodeImpl* input, int id)
    : NodeImpl(NodeType::UnaryOp, "unop", id), op_(op) {
    addInput(input);
}

PhiNode::PhiNode(const std::vector<NodeImpl*>& inputs, int id)
    : NodeImpl(NodeType::Phi, "phi", id) {
    for (auto* input : inputs) {
        addInput(input);
    }
}

ControlNode::ControlNode(const std::string& name, int id)
    : NodeImpl(NodeType::Control, name, id) {}

RegionNode::RegionNode(int id)
    : ControlNode("region", id) {}

void RegionNode::addPredecessor(NodeImpl* pred) {
    addInput(pred);
}

ProjectionNode::ProjectionNode(NodeImpl* input, int index, int id)
    : NodeImpl(NodeType::Control, "proj_" + std::to_string(index), id),
      proj_index_(index) {
    addInput(input);
}

//==============================================================================
// GraphImpl Implementation
//==============================================================================

GraphImpl::GraphImpl() {
    // Create start node as root of control flow
    auto* start = new ControlNode("start", allocateId());
    addNode(start);
}

GraphImpl::~GraphImpl() {
    for (auto* node : all_nodes_) {
        delete node;
    }
}

void GraphImpl::addNode(NodeImpl* node) {
    all_nodes_.push_back(node);
    node_map_[node->id()] = node;
}

void GraphImpl::removeNode(NodeImpl* node) {
    // Remove from all_nodes_
    auto it = std::find(all_nodes_.begin(), all_nodes_.end(), node);
    if (it != all_nodes_.end()) {
        all_nodes_.erase(it);
    }

    // Remove from node_map_
    node_map_.erase(node->id());

    // Clean up edges
    for (auto* input : node->inputs()) {
        static_cast<NodeImpl*>(input)->removeOutput(node);
    }

    delete node;
}

Node* GraphImpl::createConstant(int64_t value) {
    // Check if constant already exists (CSE for constants)
    for (auto* node : all_nodes_) {
        if (auto* const_node = dynamic_cast<ConstantNode*>(node)) {
            if (const_node->getConstant() == value) {
                return const_node;
            }
        }
    }

    auto* node = new ConstantNode(value, allocateId());
    addNode(node);
    return node;
}

Node* GraphImpl::createBinaryOp(BinaryOpType op, Node* lhs, Node* rhs) {
    auto* node = new BinaryOpNode(op, static_cast<NodeImpl*>(lhs),
                                   static_cast<NodeImpl*>(rhs), allocateId());
    addNode(node);
    return node;
}

Node* GraphImpl::createUnaryOp(UnaryOpType op, Node* input) {
    auto* node = new UnaryOpNode(op, static_cast<NodeImpl*>(input), allocateId());
    addNode(node);
    return node;
}

Node* GraphImpl::createPhi(std::vector<Node*> inputs) {
    std::vector<NodeImpl*> impl_inputs;
    for (auto* input : inputs) {
        impl_inputs.push_back(static_cast<NodeImpl*>(input));
    }
    auto* node = new PhiNode(impl_inputs, allocateId());
    addNode(node);
    return node;
}

Node* GraphImpl::createControl() {
    auto* node = new ControlNode("control", allocateId());
    addNode(node);
    return node;
}

NodeImpl* GraphImpl::createRegion() {
    auto* node = new RegionNode(allocateId());
    addNode(node);
    return node;
}

NodeImpl* GraphImpl::createProjection(NodeImpl* input, int index) {
    auto* node = new ProjectionNode(input, index, allocateId());
    addNode(node);
    return node;
}

std::vector<Node*> GraphImpl::nodes() const {
    std::vector<Node*> result;
    result.reserve(all_nodes_.size());
    for (auto* node : all_nodes_) {
        result.push_back(node);
    }
    return result;
}

std::vector<Node*> GraphImpl::roots() const {
    std::vector<Node*> result;
    for (auto* node : all_nodes_) {
        if (node->inputCount() == 0 || node->name() == "start") {
            result.push_back(node);
        }
    }
    return result;
}

void GraphImpl::optimize() {
    bool changed = true;
    int iterations = 0;
    const int max_iterations = 100;

    while (changed && iterations < max_iterations) {
        changed = false;

        // Constant folding
        if (foldConstants()) {
            changed = true;
        }

        // Common subexpression elimination
        if (eliminateCommonSubexpressions()) {
            changed = true;
        }

        // Dead code elimination
        eliminateDeadCode();

        iterations++;
    }
}

bool GraphImpl::applyPass(LoweringPass* pass) {
    return pass->runOnGraph(this);
}

//==============================================================================
// Dominance Analysis
//==============================================================================

void GraphImpl::computeDominance() {
    idom_.clear();
    dominance_tree_.clear();

    // Get all control nodes
    std::vector<NodeImpl*> control_nodes;
    for (auto* node : all_nodes_) {
        if (node->isControl()) {
            control_nodes.push_back(node);
        }
    }

    if (control_nodes.empty()) {
        return;
    }

    // Find start node
    NodeImpl* start = nullptr;
    for (auto* node : control_nodes) {
        if (node->name() == "start") {
            start = node;
            break;
        }
    }

    if (!start) {
        return;
    }

    // Compute immediate dominators using iterative algorithm
    // Initialize: start dominates itself, others unknown
    std::map<NodeImpl*, std::set<NodeImpl*>> doms;
    doms[start].insert(start);

    for (auto* node : control_nodes) {
        if (node != start) {
            // Initially, all nodes dominate this node
            for (auto* n : control_nodes) {
                doms[node].insert(n);
            }
        }
    }

    // Iterate until fixed point
    bool changed = true;
    while (changed) {
        changed = false;

        for (auto* node : control_nodes) {
            if (node == start) continue;

            // New dominators = {node} ∪ (∩ doms[pred] for all preds)
            std::set<NodeImpl*> new_doms;
            new_doms.insert(node);

            bool first = true;
            for (auto* pred : node->inputs()) {
                if (!static_cast<NodeImpl*>(pred)->isControl()) continue;

                if (first) {
                    new_doms.insert(doms[static_cast<NodeImpl*>(pred)].begin(),
                                    doms[static_cast<NodeImpl*>(pred)].end());
                    first = false;
                } else {
                    std::set<NodeImpl*> intersection;
                    std::set_intersection(new_doms.begin(), new_doms.end(),
                                        doms[static_cast<NodeImpl*>(pred)].begin(),
                                        doms[static_cast<NodeImpl*>(pred)].end(),
                                        std::inserter(intersection, intersection.begin()));
                    new_doms = intersection;
                    new_doms.insert(node);
                }
            }

            if (new_doms != doms[node]) {
                doms[node] = new_doms;
                changed = true;
            }
        }
    }

    // Compute immediate dominators
    for (auto* node : control_nodes) {
        if (node == start) continue;

        auto& dom_set = doms[node];
        dom_set.erase(node); // Remove self

        // Find the dominator closest to node
        NodeImpl* idom = nullptr;
        for (auto* d : dom_set) {
            bool is_idom = true;
            for (auto* other : dom_set) {
                if (other != d && doms[d].count(other)) {
                    is_idom = false;
                    break;
                }
            }
            if (is_idom) {
                idom = d;
                break;
            }
        }

        if (idom) {
            idom_[node] = idom;
        }
    }

    buildDominanceTree();
}

void GraphImpl::buildDominanceTree() {
    dominance_tree_.clear();
    for (const auto& [node, idom] : idom_) {
        dominance_tree_[idom].insert(node);
    }
}

bool GraphImpl::dominates(NodeImpl* a, NodeImpl* b) const {
    if (a == b) return true;

    NodeImpl* curr = b;
    while (curr) {
        auto it = idom_.find(curr);
        if (it == idom_.end()) break;
        curr = it->second;
        if (curr == a) return true;
    }

    return false;
}

NodeImpl* GraphImpl::immediateDominator(NodeImpl* node) const {
    auto it = idom_.find(node);
    return it != idom_.end() ? it->second : nullptr;
}

NodeImpl* GraphImpl::findLCA(NodeImpl* a, NodeImpl* b) const {
    // Find lowest common ancestor in dominance tree
    std::set<NodeImpl*> a_doms;
    NodeImpl* curr = a;
    while (curr) {
        a_doms.insert(curr);
        curr = immediateDominator(curr);
    }

    curr = b;
    while (curr) {
        if (a_doms.count(curr)) {
            return curr;
        }
        curr = immediateDominator(curr);
    }

    return nullptr;
}

//==============================================================================
// Scheduling
//==============================================================================

void GraphImpl::schedule() {
    scheduled_nodes_.clear();

    // Compute early schedule (as early as possible)
    scheduleEarly();

    // Compute late schedule (as late as possible, but before uses)
    scheduleLate();

    // For now, use early schedule
    // In a full implementation, we'd choose optimal placement
    std::map<NodeImpl*, int> schedule_depth;
    for (auto* node : all_nodes_) {
        schedule_depth[node] = computeDepth(node);
    }

    // Sort by depth
    scheduled_nodes_ = all_nodes_;
    std::sort(scheduled_nodes_.begin(), scheduled_nodes_.end(),
              [&](NodeImpl* a, NodeImpl* b) {
                  return schedule_depth[a] < schedule_depth[b];
              });
}

void GraphImpl::scheduleEarly() {
    // Schedule each node as early as possible (closest to inputs)
    // This is a forward pass
    for (auto* node : all_nodes_) {
        // Node must be scheduled after all inputs
        // For now, depth is just max depth of inputs + 1
        computeDepth(node);
    }
}

void GraphImpl::scheduleLate() {
    // Schedule each node as late as possible (closest to uses)
    // This is a backward pass from uses
    // For now, we keep early schedule
}

int GraphImpl::computeDepth(NodeImpl* node) {
    if (node->inputCount() == 0) {
        return 0;
    }

    int max_depth = 0;
    for (size_t i = 0; i < node->inputCount(); i++) {
        auto* input = node->getInput(i);
        if (input) {
            max_depth = std::max(max_depth, computeDepth(input) + 1);
        }
    }

    return max_depth;
}

//==============================================================================
// Optimizations
//==============================================================================

void GraphImpl::eliminateDeadCode() {
    bool changed = true;
    while (changed) {
        changed = false;

        std::vector<NodeImpl*> to_remove;
        for (auto* node : all_nodes_) {
            if (node->isDead() && node->name() != "start") {
                to_remove.push_back(node);
            }
        }

        for (auto* node : to_remove) {
            removeNode(node);
            changed = true;
        }
    }
}

bool GraphImpl::foldConstants() {
    bool changed = false;

    std::vector<NodeImpl*> to_process;
    for (auto* node : all_nodes_) {
        if (auto* binop = dynamic_cast<BinaryOpNode*>(node)) {
            to_process.push_back(binop);
        }
    }

    for (auto* node : to_process) {
        if (auto* replacement = tryFoldBinaryOp(dynamic_cast<BinaryOpNode*>(node))) {
            // Replace all uses of node with replacement
            for (auto* output : node->outputs()) {
                static_cast<NodeImpl*>(output)->replaceInput(node, replacement);
            }
            changed = true;
        }
    }

    return changed;
}

NodeImpl* GraphImpl::tryFoldBinaryOp(BinaryOpNode* node) {
    auto* lhs = dynamic_cast<ConstantNode*>(node->getInput(0));
    auto* rhs = dynamic_cast<ConstantNode*>(node->getInput(1));

    if (!lhs || !rhs) {
        return nullptr;
    }

    int64_t lval = lhs->getConstant();
    int64_t rval = rhs->getConstant();
    int64_t result = 0;

    switch (node->getOp()) {
        case BinaryOpType::Add: result = lval + rval; break;
        case BinaryOpType::Sub: result = lval - rval; break;
        case BinaryOpType::Mul: result = lval * rval; break;
        case BinaryOpType::Div:
            if (rval == 0) return nullptr;
            result = lval / rval;
            break;
        case BinaryOpType::And: result = lval & rval; break;
        case BinaryOpType::Or:  result = lval | rval; break;
        case BinaryOpType::Xor: result = lval ^ rval; break;
        case BinaryOpType::Shl: result = lval << rval; break;
        case BinaryOpType::Shr: result = lval >> rval; break;
        case BinaryOpType::Eq:  result = lval == rval; break;
        case BinaryOpType::Ne:  result = lval != rval; break;
        case BinaryOpType::Lt:  result = lval < rval; break;
        case BinaryOpType::Le:  result = lval <= rval; break;
        case BinaryOpType::Gt:  result = lval > rval; break;
        case BinaryOpType::Ge:  result = lval >= rval; break;
        default: return nullptr;
    }

    return static_cast<NodeImpl*>(createConstant(result));
}

bool GraphImpl::eliminateCommonSubexpressions() {
    bool changed = false;

    std::vector<NodeImpl*> nodes_copy = all_nodes_;

    for (size_t i = 0; i < nodes_copy.size(); i++) {
        for (size_t j = i + 1; j < nodes_copy.size(); j++) {
            if (areEquivalent(nodes_copy[i], nodes_copy[j])) {
                // Replace j with i
                for (auto* output : nodes_copy[j]->outputs()) {
                    static_cast<NodeImpl*>(output)->replaceInput(nodes_copy[j], nodes_copy[i]);
                }
                changed = true;
            }
        }
    }

    return changed;
}

bool GraphImpl::areEquivalent(NodeImpl* a, NodeImpl* b) const {
    if (a == b) return true;
    if (a->type() != b->type()) return false;
    if (a->inputCount() != b->inputCount()) return false;

    // Check if both are constants with same value
    if (auto* ca = dynamic_cast<const ConstantNode*>(a)) {
        if (auto* cb = dynamic_cast<const ConstantNode*>(b)) {
            return ca->getConstant() == cb->getConstant();
        }
    }

    // Check if both are binary ops with same operation and operands
    if (auto* ba = dynamic_cast<const BinaryOpNode*>(a)) {
        if (auto* bb = dynamic_cast<const BinaryOpNode*>(b)) {
            if (ba->getOp() != bb->getOp()) return false;
            return ba->getInput(0) == bb->getInput(0) &&
                   ba->getInput(1) == bb->getInput(1);
        }
    }

    return false;
}

bool GraphImpl::validate() const {
    // Check that all edges are consistent
    for (auto* node : all_nodes_) {
        for (auto* input : node->inputs()) {
            auto* input_impl = static_cast<const NodeImpl*>(input);
            bool found = false;
            for (auto* output : input_impl->outputs()) {
                if (output == node) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                return false;
            }
        }
    }

    return true;
}

void GraphImpl::dump(std::ostream& out) const {
    out << "Sea of Nodes Graph:\n";
    out << "  Nodes: " << all_nodes_.size() << "\n";
    for (auto* node : all_nodes_) {
        out << "  Node " << node->id() << ": " << node->name()
            << " (type=" << static_cast<int>(node->type()) << ")\n";
        out << "    Inputs: ";
        for (size_t i = 0; i < node->inputCount(); i++) {
            if (auto* input = node->getInput(i)) {
                out << input->id() << " ";
            }
        }
        out << "\n    Outputs: ";
        for (auto* output : node->outputs()) {
            out << static_cast<NodeImpl*>(output)->id() << " ";
        }
        out << "\n";
    }
}

std::string GraphImpl::toDot() const {
    std::ostringstream out;
    out << "digraph SeaOfNodes {\n";
    out << "  rankdir=TB;\n";
    out << "  node [shape=box];\n";

    for (auto* node : all_nodes_) {
        out << "  n" << node->id() << " [label=\"" << node->name()
            << "\\nid=" << node->id() << "\"];\n";

        for (size_t i = 0; i < node->inputCount(); i++) {
            if (auto* input = node->getInput(i)) {
                out << "  n" << input->id() << " -> n" << node->id() << ";\n";
            }
        }
    }

    out << "}\n";
    return out.str();
}

//==============================================================================
// Pattern Matcher Implementation
//==============================================================================

PatternMatcherImpl::PatternMatcherImpl(const std::string& name,
                                       std::function<bool(const Node*)> matcher,
                                       std::function<void(Node*)> replacer)
    : name_(name), matcher_(matcher), replacer_(replacer) {}

bool PatternMatcherImpl::matches(const Node* node) const {
    return matcher_(node);
}

std::vector<Node*> PatternMatcherImpl::findMatches(const Graph* graph) const {
    std::vector<Node*> matches;
    for (auto* node : graph->nodes()) {
        if (matcher_(node)) {
            matches.push_back(node);
        }
    }
    return matches;
}

//==============================================================================
// Lowering Pass Implementation
//==============================================================================

LoweringPassImpl::LoweringPassImpl(const std::string& name,
                                   std::vector<std::unique_ptr<PatternMatcher>> patterns)
    : name_(name), patterns_(std::move(patterns)) {}

bool LoweringPassImpl::runOnGraph(Graph* graph) {
    bool changed = false;

    for (auto& pattern : patterns_) {
        auto matches = pattern->findMatches(graph);
        if (!matches.empty()) {
            auto replacer = pattern->replacement();
            for (auto* match : matches) {
                replacer(match);
            }
            changed = true;
        }
    }

    return changed;
}

std::vector<PatternMatcher*> LoweringPassImpl::patterns() const {
    std::vector<PatternMatcher*> result;
    for (auto& p : patterns_) {
        result.push_back(p.get());
    }
    return result;
}

//==============================================================================
// Target Lowering Implementation
//==============================================================================

TargetLoweringImpl::TargetLoweringImpl(Target t) : target_(t) {
    // Create standard passes
    std::vector<std::unique_ptr<PatternMatcher>> cf_patterns;
    cf_patterns.push_back(std::make_unique<PatternMatcherImpl>(
        "fold_constants",
        [](const Node* n) {
            auto* binop = dynamic_cast<const BinaryOpNode*>(n);
            if (!binop) return false;
            return dynamic_cast<const ConstantNode*>(binop->getInput(0)) != nullptr &&
                   dynamic_cast<const ConstantNode*>(binop->getInput(1)) != nullptr;
        },
        [](Node*) { /* Folding handled by GraphImpl */ }
    ));

    passes_.push_back(std::make_unique<LoweringPassImpl>("constant-folding",
                                                         std::move(cf_patterns)));
}

std::vector<LoweringPass*> TargetLoweringImpl::passes() const {
    std::vector<LoweringPass*> result;
    for (auto& p : passes_) {
        result.push_back(p.get());
    }
    return result;
}

std::string TargetLoweringImpl::emit(const Graph* graph) const {
    auto* graph_impl = dynamic_cast<const GraphImpl*>(graph);
    if (!graph_impl) {
        return "// Error: Invalid graph type\n";
    }

    switch (target_) {
        case Target::Cpp:
            return emitCpp(graph_impl);
        case Target::MLIR:
            return emitMLIR(graph_impl);
        default:
            return "// Target not implemented\n";
    }
}

std::string TargetLoweringImpl::emitCpp(const GraphImpl* graph) const {
    std::ostringstream out;
    out << "// Generated C++ from Sea of Nodes IR\n";
    out << "#include <cstdint>\n\n";

    out << "int64_t generated_function() {\n";

    // Emit scheduled nodes
    auto scheduled = graph->getScheduledNodes();
    std::map<int, std::string> node_vars;

    for (auto* node : scheduled) {
        std::string var_name = "v" + std::to_string(node->id());
        node_vars[node->id()] = var_name;

        if (auto* const_node = dynamic_cast<const ConstantNode*>(node)) {
            out << "    int64_t " << var_name << " = "
                << const_node->getConstant() << ";\n";
        } else if (auto* binop = dynamic_cast<const BinaryOpNode*>(node)) {
            if (node->inputCount() >= 2) {
                std::string lhs = node_vars[binop->getInput(0)->id()];
                std::string rhs = node_vars[binop->getInput(1)->id()];
                std::string op_str;

                switch (binop->getOp()) {
                    case BinaryOpType::Add: op_str = "+"; break;
                    case BinaryOpType::Sub: op_str = "-"; break;
                    case BinaryOpType::Mul: op_str = "*"; break;
                    case BinaryOpType::Div: op_str = "/"; break;
                    case BinaryOpType::And: op_str = "&"; break;
                    case BinaryOpType::Or:  op_str = "|"; break;
                    case BinaryOpType::Xor: op_str = "^"; break;
                    default: op_str = "??"; break;
                }

                out << "    int64_t " << var_name << " = "
                    << lhs << " " << op_str << " " << rhs << ";\n";
            }
        }
    }

    // Return last non-control value
    for (auto it = scheduled.rbegin(); it != scheduled.rend(); ++it) {
        if (!(*it)->isControl() && !(*it)->isPhi()) {
            out << "    return " << node_vars[(*it)->id()] << ";\n";
            break;
        }
    }

    out << "}\n";
    return out.str();
}

std::string TargetLoweringImpl::emitMLIR(const GraphImpl* graph) const {
    std::ostringstream out;
    out << "// Generated MLIR from Sea of Nodes IR\n";
    out << "module {\n";
    out << "  func.func @generated_function() -> i64 {\n";

    auto scheduled = graph->getScheduledNodes();
    std::map<int, std::string> node_vars;

    for (auto* node : scheduled) {
        std::string var_name = "%v" + std::to_string(node->id());
        node_vars[node->id()] = var_name;

        if (auto* const_node = dynamic_cast<const ConstantNode*>(node)) {
            out << "    " << var_name << " = arith.constant "
                << const_node->getConstant() << " : i64\n";
        } else if (auto* binop = dynamic_cast<const BinaryOpNode*>(node)) {
            if (node->inputCount() >= 2) {
                std::string lhs = node_vars[binop->getInput(0)->id()];
                std::string rhs = node_vars[binop->getInput(1)->id()];
                std::string op_str;

                switch (binop->getOp()) {
                    case BinaryOpType::Add: op_str = "arith.addi"; break;
                    case BinaryOpType::Sub: op_str = "arith.subi"; break;
                    case BinaryOpType::Mul: op_str = "arith.muli"; break;
                    case BinaryOpType::Div: op_str = "arith.divsi"; break;
                    case BinaryOpType::And: op_str = "arith.andi"; break;
                    case BinaryOpType::Or:  op_str = "arith.ori"; break;
                    case BinaryOpType::Xor: op_str = "arith.xori"; break;
                    default: op_str = "unknown"; break;
                }

                out << "    " << var_name << " = " << op_str << " "
                    << lhs << ", " << rhs << " : i64\n";
            }
        }
    }

    // Return last value
    for (auto it = scheduled.rbegin(); it != scheduled.rend(); ++it) {
        if (!(*it)->isControl() && !(*it)->isPhi()) {
            out << "    func.return " << node_vars[(*it)->id()] << " : i64\n";
            break;
        }
    }

    out << "  }\n";
    out << "}\n";
    return out.str();
}

//==============================================================================
// Factory Functions
//==============================================================================

std::unique_ptr<Graph> createMockGraph() {
    return std::make_unique<GraphImpl>();
}

std::unique_ptr<TargetLowering> createTargetLowering(Target target) {
    return std::make_unique<TargetLoweringImpl>(target);
}

std::unique_ptr<LoweringPass> createConstantFoldingPass() {
    std::vector<std::unique_ptr<PatternMatcher>> patterns;
    return std::make_unique<LoweringPassImpl>("constant-folding", std::move(patterns));
}

std::unique_ptr<LoweringPass> createDeadCodeEliminationPass() {
    std::vector<std::unique_ptr<PatternMatcher>> patterns;
    return std::make_unique<LoweringPassImpl>("dead-code-elimination", std::move(patterns));
}

std::unique_ptr<LoweringPass> createCommonSubexpressionEliminationPass() {
    std::vector<std::unique_ptr<PatternMatcher>> patterns;
    return std::make_unique<LoweringPassImpl>("cse", std::move(patterns));
}

std::unique_ptr<Graph> astToIR(const std::any& ast) {
    // Create real graph
    auto graph = std::make_unique<GraphImpl>();

    // For demonstration, create a simple computation
    auto* const1 = graph->createConstant(42);
    auto* const2 = graph->createConstant(24);
    auto* add = graph->createBinaryOp(BinaryOpType::Add, const1, const2);

    return graph;
}

bool validateGraph(const Graph* graph) {
    if (auto* graph_impl = dynamic_cast<const GraphImpl*>(graph)) {
        return graph_impl->validate();
    }
    return false;
}

void dumpGraph(const Graph* graph, std::ostream& out) {
    if (auto* graph_impl = dynamic_cast<const GraphImpl*>(graph)) {
        graph_impl->dump(out);
    } else {
        out << "Error: Invalid graph type\n";
    }
}

std::string graphToDot(const Graph* graph) {
    if (auto* graph_impl = dynamic_cast<const GraphImpl*>(graph)) {
        return graph_impl->toDot();
    }
    return "digraph { error [label=\"Invalid graph\"]; }\n";
}

} // namespace cppfort::ir
