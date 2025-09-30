#ifndef CPPFORT_NODE_H
#define CPPFORT_NODE_H

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include "type.h"

namespace cppfort::ir {

/**
 * Base class for all nodes in the Sea of Nodes intermediate representation.
 *
 * Each Node represents an "instruction" and is linked to other Nodes by def-use dependencies.
 * Following Simple compiler Chapter 1 design.
 */
class Node {
protected:
    static int UNIQUE_ID;  // Global counter for unique node IDs

public:
    /**
     * Inputs to the node. These are use-def references to Nodes.
     * Generally fixed length, ordered, nulls allowed.
     * Ordering is required because e.g. "a/b" is different from "b/a".
     * The first input (offset 0) is often a Control node.
     */
    std::vector<Node*> _inputs;

    /**
     * Outputs reference Nodes that are not null and have this Node as an
     * input. These nodes are users of this node, thus these are def-use
     * references to Nodes.
     *
     * Outputs directly match inputs, making a directed graph that can be
     * walked in either direction.
     */
    std::vector<Node*> _outputs;

    /**
     * Unique dense integer ID assigned to each node.
     * Useful for debugging and as an index into bit vectors.
     */
    const int _nid;

    /**
     * Type of this node - represents the set of values it can take.
     * Following Simple compiler Chapter 2.
     */
    Type* _type;

    // Constructor
    Node();
    virtual ~Node() = default;

    /**
     * Check if this is a Control Flow Graph node.
     */
    virtual bool isCFG() const { return false; }

    /**
     * Add an input at the specified index.
     * Maintains the invariant that edges are bidirectional.
     */
    void setInput(int idx, Node* n);

    /**
     * Add this node to another node's outputs.
     */
    void addOutput(Node* n);

    /**
     * Remove this node from another node's outputs.
     */
    void removeOutput(Node* n);

    /**
     * Get a string representation for debugging.
     */
    virtual std::string toString() const;

    /**
     * Get the label for graph visualization.
     */
    virtual std::string label() const = 0;

    /**
     * Compute the type for this node based on its inputs.
     * This is the core of type inference and constant folding.
     */
    virtual Type* compute();

    /**
     * Peephole optimization - called after node creation.
     * Returns potentially different node if optimization occurred.
     * Following Simple compiler Chapter 2.
     */
    virtual Node* peephole();

    /**
     * Kill this node if it's unused, recursively killing unused inputs.
     */
    void kill();

    /**
     * Check if this node has no outputs (is unused).
     */
    bool isUnused() const { return _outputs.empty(); }

    /**
     * Check if this node is dead (has been killed).
     */
    bool isDead() const { return _type == nullptr; }

    /**
     * Helper to get input at index, or nullptr if out of bounds.
     */
    Node* in(int idx) const {
        return idx < _inputs.size() ? _inputs[idx] : nullptr;
    }

    /**
     * Number of inputs.
     */
    int nIns() const { return _inputs.size(); }
};

/**
 * Start node represents the start of a function.
 * Following Simple compiler Chapter 1.
 */
class StartNode : public Node {
public:
    StartNode();

    bool isCFG() const override { return true; }
    std::string label() const override { return "Start"; }
};

/**
 * Constant node represents a constant value.
 * Following Simple compiler Chapter 1 - only integer literals for now.
 */
class ConstantNode : public Node {
public:
    const int _value;

    ConstantNode(int value, Node* start);

    std::string label() const override {
        return std::to_string(_value);
    }
};

/**
 * Return node represents function termination with a value.
 * Following Simple compiler Chapter 1.
 */
class ReturnNode : public Node {
public:
    ReturnNode(Node* ctrl, Node* value);

    bool isCFG() const override { return true; }
    std::string label() const override { return "Return"; }

    /**
     * Get the return value node (second input).
     */
    Node* value() const {
        return _inputs.size() > 1 ? _inputs[1] : nullptr;
    }
};

/**
 * Add node - adds two values.
 * Following Simple compiler Chapter 2.
 */
class AddNode : public Node {
public:
    AddNode(Node* lhs, Node* rhs);

    std::string label() const override { return "+"; }
    Type* compute() override;
};

/**
 * Subtract node - subtracts rhs from lhs.
 * Following Simple compiler Chapter 2.
 */
class SubNode : public Node {
public:
    SubNode(Node* lhs, Node* rhs);

    std::string label() const override { return "-"; }
    Type* compute() override;
};

/**
 * Multiply node - multiplies two values.
 * Following Simple compiler Chapter 2.
 */
class MulNode : public Node {
public:
    MulNode(Node* lhs, Node* rhs);

    std::string label() const override { return "*"; }
    Type* compute() override;
};

/**
 * Divide node - divides lhs by rhs.
 * Following Simple compiler Chapter 2.
 */
class DivNode : public Node {
public:
    DivNode(Node* lhs, Node* rhs);

    std::string label() const override { return "/"; }
    Type* compute() override;
};

/**
 * Unary minus node - negates a value.
 * Following Simple compiler Chapter 2.
 */
class MinusNode : public Node {
public:
    MinusNode(Node* value);

    std::string label() const override { return "-"; }
    Type* compute() override;
};

/**
 * ScopeNode - manages lexical scopes and symbol tables.
 * Following Simple compiler Chapter 3.
 *
 * This node is neither a Data nor Control node, but a utility for maintaining
 * symbol tables, name lookups, and determining where Phi nodes are required.
 * It leverages the Sea of Nodes def-use architecture to maintain liveness of
 * values in scope.
 */
class ScopeNode : public Node {
public:
    /**
     * Stack of symbol tables. Each symbol table is a map from variable name
     * to an index into this ScopeNode's inputs.
     */
    std::vector<std::unordered_map<std::string, int>> _scopes;

    /**
     * Track the next available input index for new variables.
     */
    int _nextInputIdx;

    ScopeNode();

    std::string label() const override { return "Scope"; }

    /**
     * Enter a new lexical scope by pushing a new symbol table.
     */
    void push();

    /**
     * Exit the current lexical scope by popping the top symbol table.
     * Also removes the corresponding input nodes.
     */
    void pop();

    /**
     * Define a variable in the current scope.
     * Returns the input index where the value is stored.
     */
    int define(const std::string& name, Node* value);

    /**
     * Update a variable's value.
     * Looks up the variable in the scope stack and updates its value.
     */
    void update(const std::string& name, Node* value);

    /**
     * Lookup a variable in the scope stack.
     * Returns nullptr if not found.
     */
    Node* lookup(const std::string& name) const;

    /**
     * Check if a variable exists in any scope.
     */
    bool contains(const std::string& name) const;

    /**
     * Get the current scope level (depth of scope stack).
     */
    int scopeLevel() const { return _scopes.size(); }
};

} // namespace cppfort::ir

#endif // CPPFORT_NODE_H