#ifndef CPPFORT_NODE_H
#define CPPFORT_NODE_H

#include <vector>
#include <memory>
#include <string>

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

} // namespace cppfort::ir

#endif // CPPFORT_NODE_H