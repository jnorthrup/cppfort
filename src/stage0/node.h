#ifndef CPPFORT_NODE_H
#define CPPFORT_NODE_H

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include "type.h"

namespace cppfort::ir {

// Forward declaration for GVN
class Node;

/**
 * Global Value Numbering support - Following Simple compiler Chapter 9
 */
struct NodeHash {
    std::size_t operator()(const Node* n) const;
};

struct NodeEqual {
    bool operator()(const Node* a, const Node* b) const;
};

/**
 * Base class for all nodes in the Sea of Nodes intermediate representation.
 *
 * Each Node represents an "instruction" and is linked to other Nodes by def-use dependencies.
 * Following Simple compiler Chapter 1 design.
 */
class Node {
protected:
    static int UNIQUE_ID;  // Global counter for unique node IDs

    // Global Value Numbering table - Chapter 9
    static std::unordered_set<Node*, NodeHash, NodeEqual> GVN;

    /**
     * Cached hash code and edge lock for GVN.
     * If 0, node is unlocked and NOT in GVN.
     * If non-zero, node is edge-locked and IS in GVN.
     */
    mutable std::size_t _hash = 0;

    /**
     * Dependencies for distant neighbor peepholes - Chapter 9
     */
    std::vector<Node*> _deps;

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

    /**
     * GVN support - Following Simple compiler Chapter 9
     */

    // Compute hash code for this node
    virtual std::size_t hashCode() const;

    // Check value equality for GVN
    virtual bool equals(const Node* other) const;

    // Unlock node from GVN table before modifying edges
    void unlock();

    // Try to find or insert in GVN table
    Node* gvn();

    // Add a dependency for distant neighbor peepholes
    void addDep(Node* dep) { _deps.push_back(dep); }

    // Get dependencies
    const std::vector<Node*>& deps() const { return _deps; }

    // MLIR integration hooks (Band 1 premature integration)
    virtual bool hasSideEffects() const { return false; }
    virtual bool isMemoryOp() const { return false; }
    virtual int schedulePriority() const { return 0; }

    // MLIR type for this node (preparation for emission)
    virtual std::string getMLIRType() const {
        if (_type) return "i32";  // Default to 32-bit integer
        return "unknown";
    }
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

// INSTRUCTION: Add after ReturnNode
class StopNode : public Node {
    // Collects all Returns - has variable number of inputs
    std::vector<ReturnNode*> _returns;
public:
    void addReturn(ReturnNode* ret) {
        _inputs.push_back(ret);
        if (ret) ret->_outputs.push_back(this);
        _returns.push_back(ret);
    }
    bool isCFG() const override { return true; }
    std::string label() const override { return "Stop"; }
};

class PhiNode;

class RegionNode : public Node {
protected:
    std::vector<PhiNode*> _phis;  // Phis controlled by this region
public:
    RegionNode(Node* ctrl1, Node* ctrl2) : Node() { setInput(0, ctrl1); setInput(1, ctrl2); }
    void addPhi(PhiNode* phi);  // CRITICAL: Must set phi->_inputs[0] = this
    bool isCFG() const override { return true; }
    std::string label() const override { return "Region"; }
    Node* peephole() override;

    // Check if this region has all inputs (for loops)
    virtual bool hasAllInputs() const { return in(1) != nullptr; }
};

/**
 * LoopNode - extends RegionNode for loop headers.
 * Following Simple compiler Chapter 7.
 *
 * A loop initially has only one input (entry) with the backedge
 * set to null until the loop body is parsed. This allows us to
 * disable peepholes until the loop is complete.
 */
class LoopNode : public RegionNode {
public:
    LoopNode(Node* entry) : RegionNode(entry, nullptr) {}

    std::string label() const override { return "Loop"; }

    // Loops are incomplete until backedge is set
    bool hasAllInputs() const override { return in(1) != nullptr; }

    // Override peephole to disable until loop is complete
    Node* peephole() override;

    // Set the backedge after parsing loop body
    void setBackedge(Node* backedge) { setInput(1, backedge); }
};

class PhiNode : public Node {
    std::string _label;  // Variable name for debugging
public:
    // CRITICAL: Region can be nullptr at construction
    PhiNode(const std::string& label, Node* region, Node* val1, Node* val2)
        : Node(), _label(label) { setInput(0, region); setInput(1, val1); setInput(2, val2); }

    Node* region() const { return in(0); }
    void setRegion(RegionNode* r) { setInput(0, r); }

    // CRITICAL: compute() must handle cycles
    Type* compute() override;
    std::string label() const override { return "Phi[" + _label + "]"; }
};

class IfNode : public Node {
public:
    IfNode(Node* ctrl, Node* pred) : Node() { setInput(0, ctrl); setInput(1, pred); }
    bool isCFG() const override { return true; }
    std::string label() const override { return "If"; }
};

class ProjNode : public Node {
    int _idx;  // 0 for true, 1 for false projection
public:
    ProjNode(Node* ctrl, int idx) : Node(), _idx(idx) { setInput(0, ctrl); }
    bool isCFG() const override { return in(0) && in(0)->isCFG(); }
    std::string label() const override {
        return std::string("Proj[") + (_idx ? "F" : "T") + "]";
    }
};

// Comparison nodes
class BoolNode : public Node {
protected:
    BoolNode(Node* lhs, Node* rhs) : Node() { setInput(0, lhs); setInput(1, rhs); }
public:
    Type* compute() override; // Return boolean type (0/1)
};

class EQNode : public BoolNode {
public:
    EQNode(Node* lhs, Node* rhs) : BoolNode(lhs, rhs) {}
    Node* peephole() override; // fold if both constant
    std::string label() const override { return "=="; }
};

class LTNode : public BoolNode {
public:
    LTNode(Node* lhs, Node* rhs) : BoolNode(lhs, rhs) {}
    Node* peephole() override; // fold if both constant
    std::string label() const override { return "<"; }
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
 * BreakNode - represents a break statement in a loop.
 * Following Simple compiler Chapter 8.
 *
 * Conceptually represents a control flow edge that exits the loop.
 */
class BreakNode : public Node {
public:
    BreakNode(Node* ctrl) : Node() { setInput(0, ctrl); }

    bool isCFG() const override { return true; }
    std::string label() const override { return "Break"; }
};

/**
 * ContinueNode - represents a continue statement in a loop.
 * Following Simple compiler Chapter 8.
 *
 * Conceptually represents a control flow edge back to loop header.
 */
class ContinueNode : public Node {
public:
    ContinueNode(Node* ctrl) : Node() { setInput(0, ctrl); }

    bool isCFG() const override { return true; }
    std::string label() const override { return "Continue"; }
};

/**
 * Base class for memory operations - Following Simple compiler Chapter 10
 *
 * Memory operations are serialized through memory slices determined
 * by alias classes (struct+field combinations).
 */
class MemOpNode : public Node {
protected:
    int _alias;  // Alias class for this memory operation

public:
    MemOpNode(int alias) : Node(), _alias(alias) {}

    bool isMemoryOp() const override { return true; }
    int alias() const { return _alias; }
};

/**
 * NewNode - Allocates memory for a new struct instance.
 * Following Simple compiler Chapter 10.
 *
 * Takes control as input to ensure proper scheduling.
 * Returns a pointer to the newly allocated struct.
 */
class NewNode : public Node {
    std::string _structType;  // Name of struct type being allocated

public:
    NewNode(Node* ctrl, const std::string& structType)
        : Node(), _structType(structType) { setInput(0, ctrl); }

    std::string label() const override { return "New[" + _structType + "]"; }
    bool hasSideEffects() const override { return true; }

    const std::string& structType() const { return _structType; }
};

/**
 * LoadNode - Loads a value from a struct field.
 * Following Simple compiler Chapter 10.
 *
 * Inputs:
 * 0: Memory slice (for this alias class)
 * 1: Pointer to struct
 * 2: Field offset (constant)
 */
class LoadNode : public MemOpNode {
    std::string _fieldName;  // For debugging

public:
    LoadNode(int alias, Node* mem, Node* ptr, Node* offset, const std::string& field)
        : MemOpNode(alias), _fieldName(field) {
        setInput(0, mem);
        setInput(1, ptr);
        setInput(2, offset);
    }

    std::string label() const override { return "Load[" + _fieldName + "]"; }

    Node* mem() const { return in(0); }
    Node* ptr() const { return in(1); }
    Node* offset() const { return in(2); }
};

/**
 * StoreNode - Stores a value to a struct field.
 * Following Simple compiler Chapter 10.
 *
 * Inputs:
 * 0: Memory slice (for this alias class)
 * 1: Pointer to struct
 * 2: Field offset (constant)
 * 3: Value to store
 *
 * Returns: Updated memory slice
 */
class StoreNode : public MemOpNode {
    std::string _fieldName;  // For debugging

public:
    StoreNode(int alias, Node* mem, Node* ptr, Node* offset, Node* value, const std::string& field)
        : MemOpNode(alias), _fieldName(field) {
        setInput(0, mem);
        setInput(1, ptr);
        setInput(2, offset);
        setInput(3, value);
    }

    std::string label() const override { return "Store[" + _fieldName + "]"; }
    bool hasSideEffects() const override { return true; }

    Node* mem() const { return in(0); }
    Node* ptr() const { return in(1); }
    Node* offset() const { return in(2); }
    Node* value() const { return in(3); }
};

/**
 * MemProjNode - Projects memory slices from Start or merges them at Return.
 * Following Simple compiler Chapter 10.
 *
 * Each alias class gets its own memory projection.
 */
class MemProjNode : public Node {
    int _alias;  // Alias class for this memory projection

public:
    MemProjNode(Node* ctrl, int alias) : Node(), _alias(alias) { setInput(0, ctrl); }

    std::string label() const override { return "MemProj[" + std::to_string(_alias) + "]"; }
    int alias() const { return _alias; }
};

/**
 * CastNode - Type cast operation for struct pointers.
 * Following Simple compiler Chapter 10.
 *
 * Performs upcasting in the type hierarchy.
 */
class CastNode : public Node {
    Type* _toType;  // Target type

public:
    CastNode(Node* ctrl, Node* value, Type* toType)
        : Node(), _toType(toType) {
        setInput(0, ctrl);
        setInput(1, value);
    }

    std::string label() const override { return "Cast"; }
    Type* toType() const { return _toType; }
    Type* compute() override { return _toType; }
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

    // Duplicate current scope bindings into a new ScopeNode
    // If forLoop is true, sets up lazy phi sentinels (Chapter 8)
    ScopeNode* duplicate(bool forLoop = false) const;

    // Expose a snapshot of current visible variables -> Nodes
    std::unordered_map<std::string, Node*> currentBindings() const;

    // Merge two scopes at a region (for if/while)
    void mergeScopes(ScopeNode* that);

    // End a loop by connecting backedge phis
    void endLoop(ScopeNode* back, ScopeNode* exit);

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
