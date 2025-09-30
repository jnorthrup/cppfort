#include "node.h"
#include <algorithm>
#include <sstream>
#include <cassert>

namespace cppfort::ir {

// Initialize static ID counter
int Node::UNIQUE_ID = 1;

Node::Node() : _nid(UNIQUE_ID++), _type(nullptr) {
}

void Node::setInput(int idx, Node* n) {
    // Ensure inputs vector is large enough
    if (idx >= static_cast<int>(_inputs.size())) {
        _inputs.resize(idx + 1, nullptr);
    }

    // Remove this node from old input's outputs
    Node* old = _inputs[idx];
    if (old != nullptr) {
        old->removeOutput(this);
    }

    // Set new input
    _inputs[idx] = n;

    // Add this node to new input's outputs
    if (n != nullptr) {
        n->addOutput(this);
    }
}

void Node::addOutput(Node* n) {
    if (n != nullptr && std::find(_outputs.begin(), _outputs.end(), n) == _outputs.end()) {
        _outputs.push_back(n);
    }
}

void Node::removeOutput(Node* n) {
    auto it = std::find(_outputs.begin(), _outputs.end(), n);
    if (it != _outputs.end()) {
        _outputs.erase(it);
    }
}

std::string Node::toString() const {
    std::ostringstream ss;
    ss << _nid << ": " << label();
    return ss.str();
}

// Default compute implementation
Type* Node::compute() {
    return Type::BOTTOM;
}

// Peephole optimization
Node* Node::peephole() {
    // Compute initial or improved Type
    Type* type = _type = compute();

    // Replace constant computations from non-constants with a constant node
    if (dynamic_cast<ConstantNode*>(this) == nullptr && type->isConstant()) {
        // Find START node from one of our inputs (for graph connectivity)
        Node* start = nullptr;
        for (int i = 0; i < nIns(); i++) {
            Node* input = in(i);
            if (input) {
                // Walk up to find a ConstantNode which has START
                ConstantNode* c = dynamic_cast<ConstantNode*>(input);
                if (c && c->in(0)) {
                    start = c->in(0);
                    break;
                }
                // If not a constant, check if it's START itself
                StartNode* s = dynamic_cast<StartNode*>(input);
                if (s) {
                    start = s;
                    break;
                }
            }
        }

        kill();  // Kill this node because replacing with a Constant
        // Create the new ConstantNode - it will have its type set in constructor
        // No need to call peephole again on a constant
        return new ConstantNode(static_cast<TypeInteger*>(type)->value(), start);
    }

    return this;
}

// Kill this node if it's unused
void Node::kill() {
    // Ensure node has no uses
    assert(isUnused());

    // Set all inputs to null, recursively killing unused nodes
    for (int i = 0; i < nIns(); i++) {
        setInput(i, nullptr);
    }

    _inputs.clear();
    _type = nullptr;  // Flag as dead
    assert(isDead());
}

// StartNode implementation
StartNode::StartNode() : Node() {
    // Start has no inputs
    _type = Type::TOP;  // Start node has TOP type
}

// ConstantNode implementation
ConstantNode::ConstantNode(int value, Node* start) : Node(), _value(value) {
    // Set type immediately so it's available for parent nodes
    _type = TypeInteger::constant(_value);

    // Constants have Start as input to enable graph walking
    // This edge carries no semantic meaning, solely for visitation
    if (start != nullptr) {
        setInput(0, start);
    }
}

// ReturnNode implementation
ReturnNode::ReturnNode(Node* ctrl, Node* value) : Node() {
    // First input is control node
    setInput(0, ctrl);
    // Second input is the data node with return value
    setInput(1, value);
    _type = Type::TOP;  // Return node type
}

// BoolNode compute: result is boolean domain (0/1) unknown unless constants folded
Type* BoolNode::compute() {
    // Return boolean range type to preserve 0/1 information
    return TypeInteger::boolean();
}

// EQNode peephole: constant fold
Node* EQNode::peephole() {
    Node* lhs = in(0);
    Node* rhs = in(1);
    if (lhs && rhs && lhs->_type && rhs->_type) {
        auto i0 = dynamic_cast<TypeInteger*>(lhs->_type);
        auto i1 = dynamic_cast<TypeInteger*>(rhs->_type);
        if (i0 && i1 && i0->isConstant() && i1->isConstant()) {
            Node* start = nullptr;
            // try to discover a Start for anchoring
            for (Node* n : {lhs, rhs}) {
                if (auto c = dynamic_cast<ConstantNode*>(n)) start = c->in(0);
            }
            int v = (i0->value() == i1->value()) ? 1 : 0;
            kill();
            return new ConstantNode(v, start);
        }
    }
    _type = compute();
    return this;
}

// LTNode peephole: constant fold
Node* LTNode::peephole() {
    Node* lhs = in(0);
    Node* rhs = in(1);
    if (lhs && rhs && lhs->_type && rhs->_type) {
        auto i0 = dynamic_cast<TypeInteger*>(lhs->_type);
        auto i1 = dynamic_cast<TypeInteger*>(rhs->_type);
        if (i0 && i1 && i0->isConstant() && i1->isConstant()) {
            Node* start = nullptr;
            for (Node* n : {lhs, rhs}) {
                if (auto c = dynamic_cast<ConstantNode*>(n)) start = c->in(0);
            }
            int v = (i0->value() < i1->value()) ? 1 : 0;
            kill();
            return new ConstantNode(v, start);
        }
    }
    _type = compute();
    return this;
}

// AddNode implementation
AddNode::AddNode(Node* lhs, Node* rhs) : Node() {
    setInput(0, lhs);
    setInput(1, rhs);
}

Type* AddNode::compute() {
    Node* lhs = in(0);
    Node* rhs = in(1);

    if (lhs && rhs && lhs->_type && rhs->_type) {
        TypeInteger* i0 = dynamic_cast<TypeInteger*>(lhs->_type);
        TypeInteger* i1 = dynamic_cast<TypeInteger*>(rhs->_type);

        if (i0 && i1) {
            if (i0->isConstant() && i1->isConstant()) {
                return TypeInteger::constant(i0->value() + i1->value());
            }
        }
    }

    return TypeInteger::bottom();
}

// SubNode implementation
SubNode::SubNode(Node* lhs, Node* rhs) : Node() {
    setInput(0, lhs);
    setInput(1, rhs);
}

Type* SubNode::compute() {
    Node* lhs = in(0);
    Node* rhs = in(1);

    if (lhs && rhs && lhs->_type && rhs->_type) {
        TypeInteger* i0 = dynamic_cast<TypeInteger*>(lhs->_type);
        TypeInteger* i1 = dynamic_cast<TypeInteger*>(rhs->_type);

        if (i0 && i1) {
            if (i0->isConstant() && i1->isConstant()) {
                return TypeInteger::constant(i0->value() - i1->value());
            }
        }
    }

    return TypeInteger::bottom();
}

// MulNode implementation
MulNode::MulNode(Node* lhs, Node* rhs) : Node() {
    setInput(0, lhs);
    setInput(1, rhs);
}

Type* MulNode::compute() {
    Node* lhs = in(0);
    Node* rhs = in(1);

    if (lhs && rhs && lhs->_type && rhs->_type) {
        TypeInteger* i0 = dynamic_cast<TypeInteger*>(lhs->_type);
        TypeInteger* i1 = dynamic_cast<TypeInteger*>(rhs->_type);

        if (i0 && i1) {
            if (i0->isConstant() && i1->isConstant()) {
                return TypeInteger::constant(i0->value() * i1->value());
            }
        }
    }

    return TypeInteger::bottom();
}

// DivNode implementation
DivNode::DivNode(Node* lhs, Node* rhs) : Node() {
    setInput(0, lhs);
    setInput(1, rhs);
}

Type* DivNode::compute() {
    Node* lhs = in(0);
    Node* rhs = in(1);

    if (lhs && rhs && lhs->_type && rhs->_type) {
        TypeInteger* i0 = dynamic_cast<TypeInteger*>(lhs->_type);
        TypeInteger* i1 = dynamic_cast<TypeInteger*>(rhs->_type);

        if (i0 && i1) {
            if (i0->isConstant() && i1->isConstant()) {
                // Handle division by zero
                if (i1->value() == 0) {
                    return TypeInteger::bottom();
                }
                return TypeInteger::constant(i0->value() / i1->value());
            }
        }
    }

    return TypeInteger::bottom();
}

// MinusNode implementation (unary minus)
MinusNode::MinusNode(Node* value) : Node() {
    setInput(0, value);
}

Type* MinusNode::compute() {
    Node* val = in(0);

    if (val && val->_type) {
        TypeInteger* i0 = dynamic_cast<TypeInteger*>(val->_type);

        if (i0) {
            if (i0->isConstant()) {
                return TypeInteger::constant(-i0->value());
            }
        }
    }

    return TypeInteger::bottom();
}

// ScopeNode implementation
ScopeNode::ScopeNode() : _nextInputIdx(0) {
    // Start with one global scope
    push();
}

ScopeNode* ScopeNode::duplicate() const {
    auto* dup = new ScopeNode();
    dup->_scopes = _scopes;          // copy scope structure and indices
    dup->_nextInputIdx = _nextInputIdx;
    // mirror inputs vector size and set same node pointers
    dup->_inputs.resize(_inputs.size(), nullptr);
    for (size_t i = 0; i < _inputs.size(); ++i) {
        if (_inputs[i]) {
            dup->setInput(static_cast<int>(i), _inputs[i]);
        }
    }
    return dup;
}

std::unordered_map<std::string, Node*> ScopeNode::currentBindings() const {
    std::unordered_map<std::string, Node*> out;
    // fold from inner to outer; inner overrides
    for (auto it = _scopes.rbegin(); it != _scopes.rend(); ++it) {
        const auto& scope = *it;
        for (const auto& [name, idx] : scope) {
            Node* n = (idx < _inputs.size()) ? _inputs[idx] : nullptr;
            if (n) out[name] = n;  // Allow inner definitions to overwrite
        }
    }
    return out;
}

void ScopeNode::push() {
    _scopes.push_back(std::unordered_map<std::string, int>());
}

void ScopeNode::pop() {
    if (_scopes.empty()) return;

    // Get the variables defined in this scope
    const auto& currentScope = _scopes.back();

    // Remove the inputs for variables in this scope
    // This allows them to be garbage collected if no longer referenced
    std::vector<int> indicesToRemove;
    for (const auto& [name, idx] : currentScope) {
        indicesToRemove.push_back(idx);
    }

    // Sort indices in descending order to safely remove from _inputs
    std::sort(indicesToRemove.rbegin(), indicesToRemove.rend());

    // Remove the inputs (note: this may leave gaps in the input array)
    // In a production system, we might want to compact the array
    for (int idx : indicesToRemove) {
        if (idx < _inputs.size()) {
            Node* oldNode = _inputs[idx];
            _inputs[idx] = nullptr;  // Mark as removed
            if (oldNode) {
                oldNode->removeOutput(this);
                // Kill the node if it's no longer used
                if (oldNode->isUnused()) {
                    oldNode->kill();
                }
            }
        }
    }

    _scopes.pop_back();
}

int ScopeNode::define(const std::string& name, Node* value) {
    if (_scopes.empty()) {
        push();  // Ensure we have at least one scope
    }

    // Check if already defined in current scope
    if (_scopes.back().count(name) > 0) {
        // Update existing definition
        int idx = _scopes.back()[name];
        setInput(idx, value);
        return idx;
    }

    // Add new definition
    int idx = _nextInputIdx++;
    _scopes.back()[name] = idx;

    // Ensure _inputs is large enough
    while (_inputs.size() <= idx) {
        _inputs.push_back(nullptr);
    }

    setInput(idx, value);
    return idx;
}

void ScopeNode::update(const std::string& name, Node* value) {
    // Search for the variable from innermost to outermost scope
    for (int i = _scopes.size() - 1; i >= 0; --i) {
        auto& scope = _scopes[i];
        auto it = scope.find(name);
        if (it != scope.end()) {
            int idx = it->second;
            setInput(idx, value);
            return;
        }
    }

    // If not found, define it in current scope
    define(name, value);
}

Node* ScopeNode::lookup(const std::string& name) const {
    // Search from innermost to outermost scope
    for (int i = _scopes.size() - 1; i >= 0; --i) {
        const auto& scope = _scopes[i];
        auto it = scope.find(name);
        if (it != scope.end()) {
            int idx = it->second;
            return idx < _inputs.size() ? _inputs[idx] : nullptr;
        }
    }
    return nullptr;
}

bool ScopeNode::contains(const std::string& name) const {
    for (int i = _scopes.size() - 1; i >= 0; --i) {
        if (_scopes[i].count(name) > 0) {
            return true;
        }
    }
    return false;
}

// Region/Phi minimal behavior
void RegionNode::addPhi(PhiNode* phi) {
    if (!phi) return;
    _phis.push_back(phi);
    phi->setRegion(this);
}

Type* PhiNode::compute() {
    Node* v1 = in(1);
    Node* v2 = in(2);
    if (!v1 || !v2 || !v1->_type || !v2->_type) {
        return TypeInteger::bottom();
    }

    // Use type meet with cycle detection (generation counter).
    Type::GENERATION++;
    Type* meet = v1->_type->meet(v2->_type);
    if (meet == nullptr) {
        return Type::BOTTOM;
    }
    return meet;
}

} // namespace cppfort::ir
