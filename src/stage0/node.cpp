#include "node.h"
#include <algorithm>
#include <sstream>
#include <cassert>

namespace cppfort::ir {

// Initialize static members
int Node::UNIQUE_ID = 1;

// Initialize GVN table - Chapter 9
std::unordered_set<Node*, NodeHash, NodeEqual> Node::GVN;

Node::Node() : _nid(UNIQUE_ID++), _type(nullptr) {
}

void Node::setInput(int idx, Node* n) {
    // Unlock from GVN before modifying edges - Chapter 9
    unlock();

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

    // Try Global Value Numbering - Chapter 9
    Node* gvn_result = gvn();
    if (gvn_result != this) {
        return gvn_result;
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
StartNode::StartNode() : CFGNode() {
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

// ============================================================================
// Chapter 16: Bitwise Operation Implementations
// ============================================================================

// AndNode implementation
AndNode::AndNode(Node* lhs, Node* rhs) : Node() {
    setInput(0, lhs);
    setInput(1, rhs);
}

Type* AndNode::compute() {
    Node* lhs = in(0);
    Node* rhs = in(1);

    if (lhs && rhs && lhs->_type && rhs->_type) {
        TypeInteger* i0 = dynamic_cast<TypeInteger*>(lhs->_type);
        TypeInteger* i1 = dynamic_cast<TypeInteger*>(rhs->_type);

        if (i0 && i1) {
            if (i0->isConstant() && i1->isConstant()) {
                return TypeInteger::constant(i0->value() & i1->value());
            }
        }
    }

    return TypeInteger::bottom();
}

// OrNode implementation
OrNode::OrNode(Node* lhs, Node* rhs) : Node() {
    setInput(0, lhs);
    setInput(1, rhs);
}

Type* OrNode::compute() {
    Node* lhs = in(0);
    Node* rhs = in(1);

    if (lhs && rhs && lhs->_type && rhs->_type) {
        TypeInteger* i0 = dynamic_cast<TypeInteger*>(lhs->_type);
        TypeInteger* i1 = dynamic_cast<TypeInteger*>(rhs->_type);

        if (i0 && i1) {
            if (i0->isConstant() && i1->isConstant()) {
                return TypeInteger::constant(i0->value() | i1->value());
            }
        }
    }

    return TypeInteger::bottom();
}

// XorNode implementation
XorNode::XorNode(Node* lhs, Node* rhs) : Node() {
    setInput(0, lhs);
    setInput(1, rhs);
}

Type* XorNode::compute() {
    Node* lhs = in(0);
    Node* rhs = in(1);

    if (lhs && rhs && lhs->_type && rhs->_type) {
        TypeInteger* i0 = dynamic_cast<TypeInteger*>(lhs->_type);
        TypeInteger* i1 = dynamic_cast<TypeInteger*>(rhs->_type);

        if (i0 && i1) {
            if (i0->isConstant() && i1->isConstant()) {
                return TypeInteger::constant(i0->value() ^ i1->value());
            }
        }
    }

    return TypeInteger::bottom();
}

// ShlNode implementation (shift left)
ShlNode::ShlNode(Node* value, Node* shift) : Node() {
    setInput(0, value);
    setInput(1, shift);
}

Type* ShlNode::compute() {
    Node* value = in(0);
    Node* shift = in(1);

    if (value && shift && value->_type && shift->_type) {
        TypeInteger* v = dynamic_cast<TypeInteger*>(value->_type);
        TypeInteger* s = dynamic_cast<TypeInteger*>(shift->_type);

        if (v && s) {
            if (v->isConstant() && s->isConstant()) {
                return TypeInteger::constant(v->value() << s->value());
            }
        }
    }

    return TypeInteger::bottom();
}

// AShrNode implementation (arithmetic shift right)
AShrNode::AShrNode(Node* value, Node* shift) : Node() {
    setInput(0, value);
    setInput(1, shift);
}

Type* AShrNode::compute() {
    Node* value = in(0);
    Node* shift = in(1);

    if (value && shift && value->_type && shift->_type) {
        TypeInteger* v = dynamic_cast<TypeInteger*>(value->_type);
        TypeInteger* s = dynamic_cast<TypeInteger*>(shift->_type);

        if (v && s) {
            if (v->isConstant() && s->isConstant()) {
                // Arithmetic shift right (sign extension)
                return TypeInteger::constant(v->value() >> s->value());
            }
        }
    }

    return TypeInteger::bottom();
}

// LShrNode implementation (logical shift right)
LShrNode::LShrNode(Node* value, Node* shift) : Node() {
    setInput(0, value);
    setInput(1, shift);
}

Type* LShrNode::compute() {
    Node* value = in(0);
    Node* shift = in(1);

    if (value && shift && value->_type && shift->_type) {
        TypeInteger* v = dynamic_cast<TypeInteger*>(value->_type);
        TypeInteger* s = dynamic_cast<TypeInteger*>(shift->_type);

        if (v && s) {
            if (v->isConstant() && s->isConstant()) {
                // Logical shift right (zero extension)
                unsigned int val = static_cast<unsigned int>(v->value());
                return TypeInteger::constant(val >> s->value());
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

// Chapter 8: Support for lazy phi creation in loops
ScopeNode* ScopeNode::duplicate(bool forLoop) const {
    auto* dup = new ScopeNode();
    dup->_scopes = _scopes;
    dup->_nextInputIdx = _nextInputIdx;
    dup->_inputs.resize(_inputs.size(), nullptr);

    if (!forLoop) {
        // Normal duplication - copy all inputs
        for (size_t i = 0; i < _inputs.size(); ++i) {
            if (_inputs[i]) {
                dup->setInput(static_cast<int>(i), _inputs[i]);
            }
        }
    } else {
        // Loop duplication - set up lazy phi sentinels
        // First input is always control
        if (!_inputs.empty() && _inputs[0]) {
            dup->setInput(0, _inputs[0]);
        }

        // For other inputs, use this scope as sentinel for lazy phi creation
        for (size_t i = 1; i < _inputs.size(); ++i) {
            if (_inputs[i]) {
                // Use this scope as sentinel
                dup->setInput(static_cast<int>(i), const_cast<ScopeNode*>(this));
                // Also update our own input to prepare for phi
                const_cast<ScopeNode*>(this)->setInput(static_cast<int>(i), const_cast<ScopeNode*>(this));
            }
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

Node* RegionNode::peephole() {
    _type = Type::TOP;  // Control nodes have TOP type
    return this;
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

// Chapter 9: GVN implementations
std::size_t Node::hashCode() const {
    if (_hash != 0) return _hash;  // Use cached hash

    // Compute hash based on node type (label) and inputs
    std::size_t h = std::hash<std::string>{}(label());
    for (int i = 0; i < nIns(); i++) {
        Node* input = in(i);
        // Mix in input pointer hash
        h = h * 31 + std::hash<Node*>{}(input);
    }

    // Avoid zero hash (reserved for unlocked)
    if (h == 0) h = 1;

    return h;
}

bool Node::equals(const Node* other) const {
    if (this == other) return true;
    if (!other) return false;

    // Same opcode (label) and same inputs
    if (label() != other->label()) return false;
    if (nIns() != other->nIns()) return false;

    for (int i = 0; i < nIns(); i++) {
        if (in(i) != other->in(i)) return false;
    }

    return true;
}

void Node::unlock() {
    if (_hash != 0) {
        GVN.erase(this);
        _hash = 0;
    }
}

Node* Node::gvn() {
    if (_hash != 0) return this;  // Already in GVN

    // Try to find existing node
    auto it = GVN.find(this);
    if (it != GVN.end()) {
        // Found existing node - use it instead
        Node* existing = *it;

        // Join types (monotonic increase)
        if (existing->_type && _type) {
            Type::GENERATION++;
            Type* joined = existing->_type->meet(_type);
            if (joined) existing->_type = joined;
        }

        // Kill this node and return existing
        kill();
        return existing;
    }

    // Insert this node as canonical
    _hash = hashCode();
    GVN.insert(this);
    return this;
}

// NodeHash and NodeEqual implementations for GVN table
std::size_t NodeHash::operator()(const Node* n) const {
    return n ? n->hashCode() : 0;
}

bool NodeEqual::operator()(const Node* a, const Node* b) const {
    if (a == b) return true;
    if (!a || !b) return false;
    return a->equals(b);
}

// Chapter 7: LoopNode implementation
Node* LoopNode::peephole() {
    // Disable peepholes until loop is complete
    if (!hasAllInputs()) {
        _type = Type::TOP;
        return this;
    }

    // Once complete, behave like normal region
    return RegionNode::peephole();
}

void LoopNode::forceExit(CFGNode* stop) {
    // Create a NeverNode controlled by the loop header (this)
    NeverNode* never = new NeverNode(this);

    // Create true/false control projections from the NeverNode
    CProjNode* trueProj = new CProjNode(never, 0, "Never.T");
    CProjNode* falseProj = new CProjNode(never, 1, "Never.F");

    // Create a Return node on the true projection so Stop can collect it
    ReturnNode* ret = new ReturnNode(trueProj, nullptr);

    // Attach the return to the Stop node if possible
    if (StopNode* s = dynamic_cast<StopNode*>(stop)) {
        s->addReturn(ret);
    }

    // Use the false projection as the loop backedge so the loop can continue
    setBackedge(falseProj);

    // Invalidate cached loop depth so it will be recomputed later
    setLoopDepth(0);
}


void ScopeNode::mergeScopes(ScopeNode* that) {
    if (!that) return;

    // Get the region node (assumed to be control input)
    Node* region = in(0);

    // Create phis for differing values
    for (int i = 1; i < nIns() && i < that->nIns(); i++) {
        if (in(i) != that->in(i)) {
            // Need a phi - but check for lazy phi sentinels first
            Node* thisVal = in(i);
            Node* thatVal = that->in(i);

            // Handle lazy phi sentinels
            if (thisVal == this) {
                // Create actual phi
                auto* phi = new PhiNode("var", region, in(i), nullptr);
                setInput(i, phi->peephole());
            } else if (thisVal && thatVal) {
                // Normal merge - create phi
                auto* phi = new PhiNode("var", region, thisVal, thatVal);
                setInput(i, phi->peephole());
            }
        }
    }
}

void ScopeNode::endLoop(ScopeNode* back, ScopeNode* exit) {
    if (!back || !exit) return;

    Node* loopRegion = in(0);

    // Connect backedge phis
    for (int i = 1; i < nIns(); i++) {
        if (back->in(i) != this) {
            // Actual phi exists
            if (auto* phi = dynamic_cast<PhiNode*>(in(i))) {
                if (phi->region() == loopRegion && phi->in(2) == nullptr) {
                    phi->setInput(2, back->in(i));

                    // Eagerly remove useless phis
                    Node* simplified = phi->peephole();
                    if (simplified != phi) {
                        // subsume would be: replace all uses of phi with simplified
                        setInput(i, simplified);
                        phi->kill();
                    }
                }
            }
        }

        // Replace lazy phi sentinels on exit path
        if (exit->in(i) == this) {
            exit->setInput(i, in(i));
        }
    }
}

// cfg0 implementation - find first CFG node reachable from this node
Node* Node::cfg0() const {
    // If this is already a CFG node, return it
    if (isCFG()) return const_cast<Node*>(this);

    // For non-CFG nodes, look at control input
    if (in(0) && in(0)->isCFG()) {
        return in(0);
    }

    // For floating nodes, return nullptr (caller handles)
    return nullptr;
}

// ============================================================================
// CFGNode implementations
// ============================================================================

CFGNode* CFGNode::idom() {
    // Default implementation: immediate dominator is control input
    return dynamic_cast<CFGNode*>(in(0));
}

CFGNode* CFGNode::cfg0() {
    // CFG nodes are their own cfg0
    return this;
}

int CFGNode::idepth() {
    if (_idepth != 0) return _idepth;

    CFGNode* dom = idom();
    if (!dom) return _idepth = 1;

    return _idepth = dom->idepth() + 1;
}

CFGNode* CFGNode::idom(CFGNode* rhs) {
    // Compute LCA of two nodes in dominator tree
    if (!rhs) return this;

    CFGNode* lhs = this;
    while (lhs != rhs) {
        int comp = lhs->idepth() - rhs->idepth();
        if (comp >= 0) lhs = lhs->idom();
        if (comp <= 0) rhs = rhs->idom();
        if (!lhs || !rhs) break;  // Safety check
    }
    return lhs;
}

int CFGNode::loopDepth() {
    if (_loopDepth != 0) return _loopDepth;

    CFGNode* cfg = dynamic_cast<CFGNode*>(in(0));
    if (!cfg) return _loopDepth = 1;

    return _loopDepth = cfg->loopDepth();
}

CFGNode* RegionNode::idom() {
    // LCA of all control inputs
    CFGNode* lca = nullptr;
    for (int i = 1; i < nIns(); i++) {
        if (Node* n = in(i)) {
            if (CFGNode* cfg = dynamic_cast<CFGNode*>(n)) {
                lca = cfg->idom(lca);
            }
        }
    }
    return lca;
}

int RegionNode::idepth() {
    if (_idepth != 0) return _idepth;

    // Maximum depth of all inputs plus one
    int d = 0;
    for (Node* n : _inputs) {
        if (n && n->isCFG()) {
            CFGNode* cfg = dynamic_cast<CFGNode*>(n);
            if (cfg) d = std::max(d, cfg->idepth() + 1);
        }
    }
    return _idepth = d;
}

int RegionNode::loopDepth() {
    if (_loopDepth != 0) return _loopDepth;

    // Use first non-null input
    for (int i = 1; i < nIns(); i++) {
        if (Node* n = in(i)) {
            if (CFGNode* cfg = dynamic_cast<CFGNode*>(n)) {
                return _loopDepth = cfg->loopDepth();
            }
        }
    }
    return _loopDepth = 1;
}

CFGNode* LoopNode::idom() {
    // Loop's idom is its entry, not LCA of entry and backedge
    return dynamic_cast<CFGNode*>(in(0));
}

int LoopNode::idepth() {
    if (_idepth != 0) return _idepth;

    CFGNode* dom = idom();
    if (!dom) return _idepth = 1;

    return _idepth = dom->idepth() + 1;
}

int LoopNode::loopDepth() {
    if (_loopDepth != 0) return _loopDepth;

    // Entry depth plus one for the loop
    CFGNode* entry = dynamic_cast<CFGNode*>(in(0));
    _loopDepth = entry ? entry->loopDepth() + 1 : 2;

    return _loopDepth;
}

CFGNode* IfNode::idom() {
    return dynamic_cast<CFGNode*>(in(0));
}

int StopNode::idepth() {
    if (_idepth != 0) return _idepth;

    // Maximum depth of all return inputs
    int d = 0;
    for (Node* n : _inputs) {
        if (n && n->isCFG()) {
            CFGNode* cfg = dynamic_cast<CFGNode*>(n);
            if (cfg) d = std::max(d, cfg->idepth() + 1);
        }
    }
    return _idepth = d;
}


// ============================================================================
// Band 4: Type System Extension Node Implementations
// ============================================================================

// Note: Type conversion implementations will use the existing CastNode
// or extend it in future. For now, focus on array operations.

// NewArrayNode implementation
Type* NewArrayNode::compute() {
    // Array allocation returns a pointer to array
    return TypeArray::dynamic(_element_type);
}

// ArrayLoadNode implementation
Type* ArrayLoadNode::compute() {
    Node* array_node = array();
    if (!array_node || !array_node->_type) return Type::BOTTOM;

    TypeArray* array_type = dynamic_cast<TypeArray*>(array_node->_type);
    if (!array_type) return Type::BOTTOM;

    return array_type->elementType();
}

Node* ArrayLoadNode::peephole() {
    // TODO: Constant folding for array loads with known index
    // TODO: Load-after-store elimination
    return this;
}

// ArrayStoreNode implementation
Type* ArrayStoreNode::compute() {
    // Store returns updated memory
    return mem()->_type;
}

Node* ArrayStoreNode::peephole() {
    // TODO: Dead store elimination
    // TODO: Store-after-store to same location
    return this;
}

// ArrayLengthNode implementation
Type* ArrayLengthNode::compute() {
    Node* array_node = array();
    if (!array_node || !array_node->_type) return TypeInteger::bottom();

    TypeArray* array_type = dynamic_cast<TypeArray*>(array_node->_type);
    if (!array_type) return TypeInteger::bottom();

    // If array has known fixed size, return constant
    if (!array_type->isDynamic()) {
        return TypeInteger::constant(array_type->length());
    }

    // Otherwise return non-constant integer
    return TypeInteger::bottom();
}

Node* ArrayLengthNode::peephole() {
    // If array length is constant, we already return it from compute()
    return this;
}

// ============================================================================
// Chapter 16: Constructor Validation
// ============================================================================

bool NewNode::validateInitialization() const {
    if (!_type) {
        // No type metadata - cannot validate
        return true;
    }

    // Check each field in the struct
    for (const Field& field : _type->fields()) {
        // Check if field has an initializer in the allocation site
        Node* init = getFieldInit(field.name);

        // If no initializer at allocation site, check if field has default value
        if (!init) {
            init = field.initialValue;
        }

        // Final fields MUST be initialized
        if (field.isFinal && !init) {
            // TODO: Report error with field name
            return false;
        }

        // Non-nullable pointer types MUST be initialized
        TypePointer* ptrType = dynamic_cast<TypePointer*>(field.type);
        if (ptrType && !ptrType->isNullable() && !init) {
            // TODO: Report error with field name
            return false;
        }
    }

    return true;
}

// ============================================================================
// Chapter 18: Function Nodes
// ============================================================================

Type* FunNode::compute() {
    // Function nodes represent their function type
    return _sig;
}

Node* FunNode::peephole() {
    return this;
}

bool FunNode::inProgress() const {
    // Function is in progress if it has unknown callers (null inputs beyond the first)
    for (size_t i = 1; i < _inputs.size(); ++i) {
        if (_inputs[i] == nullptr) {
            return true;
        }
    }
    return false;
}

Type* ParmNode::compute() {
    // ParmNode merges argument types from all call sites
    // Input 0 is the FunNode, inputs 1+ are arguments from call sites

    if (nIns() <= 1) {
        // No arguments yet, return bottom
        return Type::BOTTOM;
    }

    // Start with the first argument type
    Type* result = in(1)->_type;
    if (!result) return Type::BOTTOM;

    // Meet with all other argument types
    for (int i = 2; i < nIns(); ++i) {
        Node* arg = in(i);
        if (!arg || !arg->_type) return Type::BOTTOM;

        Type::GENERATION++;
        result = result->meet(arg->_type);
        if (result == Type::BOTTOM) return Type::BOTTOM;
    }

    return result;
}

Node* ParmNode::peephole() {
    return this;
}

CallEndNode* CallNode::cend() const {
    // CallEnd is always the first output if it exists
    if (!_outputs.empty()) {
        return dynamic_cast<CallEndNode*>(_outputs[0]);
    }
    return nullptr;
}

Type* CallNode::compute() {
    // Call node type is control flow
    return Type::BOTTOM;  // TODO: Return proper control type
}
Node* CallNode::peephole() {
    // Link exact constant functions
    Node* fptr_node = fptr();
    if (!fptr_node) return this;

    // Check if fptr is a constant function
    ConstantNode* const_fptr = dynamic_cast<ConstantNode*>(fptr_node);
    if (!const_fptr) return this;

    TypeFunPtr* fun_type = dynamic_cast<TypeFunPtr*>(const_fptr->_type);
    if (!fun_type || fun_type->fidx() < 0) return this;

    // Find the function with this fidx
    // TODO: Need a way to find FunNode by fidx
    // For now, assume we can find it somehow
    // FunNode* fun = findFunByFidx(fun_type->fidx());
    // if (!fun) return this;
    // For now, skip inlining logic - just return this
    return this;
}

Type* CallEndNode::compute() {
    // CallEnd returns the return type of the called function(s)
    // Input 0 is CallNode, inputs 1+ are linked functions

    if (nIns() <= 1) {
        // No linked functions yet
        return Type::BOTTOM;
    }

    // Get return type from first linked function
    FunNode* fun = dynamic_cast<FunNode*>(in(1));
    if (!fun || !fun->sig()) return Type::BOTTOM;

    Type* result = fun->sig()->ret();

    // Meet with return types of other linked functions
    for (int i = 2; i < nIns(); ++i) {
        FunNode* other_fun = dynamic_cast<FunNode*>(in(i));
        if (!other_fun || !other_fun->sig()) return Type::BOTTOM;

        Type::GENERATION++;
        result = result->meet(other_fun->sig()->ret());
        if (result == Type::BOTTOM) return Type::BOTTOM;
    }

    return result;
}

Node* CallEndNode::peephole() {
    // TODO: Implement call end optimizations
    return this;
}

} // namespace cppfort::ir
