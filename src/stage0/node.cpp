#include "node.h"
#include <algorithm>
#include <sstream>

namespace cppfort::ir {

// Initialize static ID counter
int Node::UNIQUE_ID = 1;

Node::Node() : _nid(UNIQUE_ID++) {
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

// StartNode implementation
StartNode::StartNode() : Node() {
    // Start has no inputs
}

// ConstantNode implementation
ConstantNode::ConstantNode(int value, Node* start) : Node(), _value(value) {
    // Constants have Start as input to enable graph walking
    // This edge carries no semantic meaning, solely for visitation
    setInput(0, start);
}

// ReturnNode implementation
ReturnNode::ReturnNode(Node* ctrl, Node* value) : Node() {
    // First input is control node
    setInput(0, ctrl);
    // Second input is the data node with return value
    setInput(1, value);
}

} // namespace cppfort::ir