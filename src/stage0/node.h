#ifndef CPPFORT_NODE_H
#define CPPFORT_NODE_H

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include "type.h"
#include "../utils/multi_index.h"

namespace cppfort::ir {

/**
 * Chapter 19: Target MLIR Dialect Enum
 * Represents the various MLIR dialects for instruction selection
 */
enum class TargetLanguage {
    MLIR_ARITH,    // Arithmetic dialect (add, sub, mul, div, etc.)
    MLIR_CF,       // Control Flow dialect (branches, jumps)
    MLIR_SCF,      // Structured Control Flow (if, for, while)
    MLIR_MEMREF,   // Memory Reference dialect (loads, stores)
    MLIR_FUNC,     // Function dialect (calls, returns)
    UNKNOWN        // Unspecified/unhandled dialect
};

// Forward declarations for function nodes
class FunNode;
class ParmNode;
class CallNode;
class CallEndNode;

// Forward declaration for GVN
class Node;

/**
 * Global Value Numbering support - Following Simple compiler Chapter 9
 */
struct NodeHash {
    ::std::size_t operator()(const Node* n) const;
};

struct NodeEqual {
    bool operator()(const Node* a, const Node* b) const;
};

/**
 * Base class for all nodes in the Sea of Nodes class MinusNode : public Node {
public:
    MinusNode(Node* val);

    ::std::string label() const override { return "-"; }
    Type* compute() override;
    NodeKind getKind() const override { return NodeKind::NEG; }
};diate representation.
 *
 * Each Node represents an "instruction" and is linked to other Nodes by def-use dependencies.
 * Following Simple compiler Chapter 1 design.
 */
class Node {
protected:
    static int UNIQUE_ID;  // Global counter for unique node IDs

    // Global Value Numbering table - Chapter 9
    static ::std::unordered_set<Node*, NodeHash, NodeEqual> GVN;

    /**
     * Cached hash code and edge lock for GVN.
     * If 0, node is unlocked and NOT in GVN.
     * If non-zero, node is edge-locked and IS in GVN.
     */
    mutable ::std::size_t _hash = 0;

    /**
     * Dependencies for distant neighbor peepholes - Chapter 9
     */
    ::std::vector<Node*> _deps;

public:
    /**
     * Inputs to the node. These are use-def references to Nodes.
     * Generally fixed length, ordered, nulls allowed.
     * Ordering is required because e.g. "a/b" is different from "b/a".
     * The first input (offset 0) is often a Control node.
     */
    ::std::vector<Node*> _inputs;

    /**
     * Outputs reference Nodes that are not null and have this Node as an
     * input. These nodes are users of this node, thus these are def-use
     * references to Nodes.
     *
     * Outputs directly match inputs, making a directed graph that can be
     * walked in either direction.
     */
    ::std::vector<Node*> _outputs;

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
     * Get the next unique ID for node creation.
     */
    static int nextUniqueId() { return UNIQUE_ID++; }

    /**
     * Get the unique node ID.
     */
    int id() const { return _nid; }

    /**
     * Check if this is a Control Flow Graph node.
     */
    virtual bool isCFG() const { return false; }

    /**
     * Get the NodeKind for this node - Band 5 pattern matching support.
     */
    virtual NodeKind getKind() const = 0;

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
    virtual ::std::string toString() const;

    /**
     * Get the label for graph visualization.
     */
    virtual ::std::string label() const = 0;

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
    virtual ::std::size_t hashCode() const;

    // Check value equality for GVN
    virtual bool equals(const Node* other) const;

    // Unlock node from GVN table before modifying edges
    void unlock();

    // Try to find or insert in GVN table
    Node* gvn();

    // Add a dependency for distant neighbor peepholes
    void addDep(Node* dep) { _deps.push_back(dep); }

    // Get dependencies
    const ::std::vector<Node*>& deps() const { return _deps; }

    // MLIR integration hooks (Band 1 premature integration)
    virtual bool hasSideEffects() const { return false; }
    virtual bool isMemoryOp() const { return false; }
    virtual int schedulePriority() const { return 0; }

    // MLIR type for this node (preparation for emission)
    virtual ::std::string getMLIRType() const {
        if (_type) return "i32";  // Default to 32-bit integer
        return "unknown";
    }

    /**
     * Get the first CFG node reachable from this node.
     * Used for scheduling floating nodes.
     */
    Node* cfg0() const;
};

/**
 * CFGNode - Base class for all Control Flow Graph nodes.
 *
 * CFG nodes are immovable anchors in the scheduling algorithm.
 * They maintain dominator information and loop nesting depth.
 */
class CFGNode : public Node {
protected:
    /**
     * Cached immediate dominator depth.
     * 0 means not computed yet.
     * Increases monotonically down the dominator tree.
     */
    int _idepth = 0;

    /**
     * Loop nesting depth.
     * 0 means not computed yet.
     * Increases with each nested loop level.
     */
    int _loopDepth = 0;

    /**
     * Anti-dependency tracking for memory operations.
     * Set to the node ID of the Load that requires ordering.
     */
    int _anti = 0;

public:
    CFGNode() : Node() {}

    bool isCFG() const override { return true; }

    /**
     * Check if this CFG node starts a basic block.
     */
    virtual bool blockHead() const { return false; }

    /**
     * Check if this CFG node ends a basic block.
     */
    virtual bool blockTail() const { return false; }

    /**
     * Get the immediate dominator of this node.
     * Returns nullptr for Start/Stop nodes.
     */
    virtual CFGNode* idom();

    /**
     * Get the immediate dominator depth.
     * Cached on first computation.
     */
    virtual int idepth();

    /**
     * Compute the LCA (Lowest Common Ancestor) of two dominators.
     */
    CFGNode* idom(CFGNode* rhs);

    /**
     * Get the loop nesting depth.
     * Cached on first computation.
     */
    virtual int loopDepth();

    /**
     * Get the first CFG node reachable from this node.
     * Used for scheduling floating nodes.
     */
    CFGNode* cfg0();

    /**
     * Force an exit from an infinite loop to make it reachable.
     */
    virtual void forceExit(CFGNode* stop) {}

    // Public accessors for GCM algorithm
    int getLoopDepth() const { return _loopDepth; }
    void setLoopDepth(int depth) { _loopDepth = depth; }
    int getAnti() const { return _anti; }
    void setAnti(int anti) { _anti = anti; }
};

/**
 * Start node represents the start of a function.
 * Following Simple compiler Chapter 1.
 */
class StartNode : public CFGNode {
public:
    StartNode();

    bool blockHead() const override { return true; }
    ::std::string label() const override { return "Start"; }
    CFGNode* idom() override { return nullptr; }
    int idepth() override { return 0; }
    int loopDepth() override { return 1; }
    NodeKind getKind() const override { return NodeKind::START; }
};

/**
 * Constant node represents a constant value.
 * Following Simple compiler Chapter 1 - only integer literals for now.
 */
class ConstantNode : public Node {
public:
    const int _value;

    ConstantNode(int value, Node* start);

    ::std::string label() const override {
        return ::std::to_string(_value);
    }
    NodeKind getKind() const override { return NodeKind::CONSTANT; }
};

/**
 * Return node represents function termination with a value.
 * Following Simple compiler Chapter 1.
 */
class ReturnNode : public Node {
public:
    ReturnNode(Node* ctrl, Node* value);

    bool isCFG() const override { return true; }
    ::std::string label() const override { return "Return"; }
    NodeKind getKind() const override { return NodeKind::RETURN; }

    /**
     * Get the return value node (second input).
     */
    Node* value() const {
        return _inputs.size() > 1 ? _inputs[1] : nullptr;
    }
};

// INSTRUCTION: Add after ReturnNode
class StopNode : public CFGNode {
    // Collects all Returns - has variable number of inputs
    ::std::vector<ReturnNode*> _returns;
public:
    void addReturn(ReturnNode* ret) {
        _inputs.push_back(ret);
        if (ret) ret->_outputs.push_back(this);
        _returns.push_back(ret);
    }
    bool blockHead() const override { return true; }
    bool blockTail() const override { return true; }
    ::std::string label() const override { return "Stop"; }
    CFGNode* idom() override { return nullptr; }
    int idepth() override;
    int loopDepth() override { return 1; }
};

class PhiNode;

class RegionNode : public CFGNode {
protected:
    ::std::vector<PhiNode*> _phis;  // Phis controlled by this region
public:
    RegionNode(Node* ctrl1, Node* ctrl2) : CFGNode() { setInput(0, ctrl1); setInput(1, ctrl2); }
    void addPhi(PhiNode* phi);  // CRITICAL: Must set phi->_inputs[0] = this
    bool blockHead() const override { return true; }
    ::std::string label() const override { return "Region"; }
    CFGNode* idom() override;
    int idepth() override;
    int loopDepth() override;
    Node* peephole() override;
    NodeKind getKind() const override { return NodeKind::REGION; }

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

    ::std::string label() const override { return "Loop"; }
    CFGNode* idom() override;
    int idepth() override;
    int loopDepth() override;
    NodeKind getKind() const override { return NodeKind::LOOP; }

    // Loops are incomplete until backedge is set
    bool hasAllInputs() const override { return in(1) != nullptr; }

    // Override peephole to disable until loop is complete
    Node* peephole() override;

    // Set the backedge after parsing loop body
    void setBackedge(Node* backedge) { setInput(1, backedge); }

    /**
     * Force an exit from this loop by creating a NeverNode and wiring
     * a true-projection to Stop and a false-projection as the backedge.
     * This follows the Chapter-11 pattern for making infinite loops reachable.
     */
    void forceExit(CFGNode* stop) override;
};

class PhiNode : public Node {
protected:
    ::std::string _label;  // Variable name for debugging
public:
    // CRITICAL: Region can be nullptr at construction
    PhiNode(const ::std::string& label, Node* region, Node* val1, Node* val2)
        : Node(), _label(label) { setInput(0, region); setInput(1, val1); setInput(2, val2); }

    Node* region() const { return in(0); }
    void setRegion(RegionNode* r) { setInput(0, r); }

    // CRITICAL: compute() must handle cycles
    Type* compute() override;
    ::std::string label() const override { return "Phi[" + _label + "]"; }
    NodeKind getKind() const override { return NodeKind::PHI; }
};

class IfNode : public CFGNode {
public:
    IfNode(Node* ctrl, Node* pred) : CFGNode() { setInput(0, ctrl); setInput(1, pred); }
    bool blockTail() const override { return true; }
    ::std::string label() const override { return "If"; }
    CFGNode* idom() override;
    NodeKind getKind() const override { return NodeKind::IF; }
};

class ProjNode : public Node {
    int _idx;  // 0 for true, 1 for false projection
public:
    ProjNode(Node* ctrl, int idx) : Node(), _idx(idx) { setInput(0, ctrl); }
    bool isCFG() const override { return in(0) && in(0)->isCFG(); }
    ::std::string label() const override {
        return ::std::string("Proj[") + (_idx ? "F" : "T") + "]";
    }
    int idx() const { return _idx; }
    NodeKind getKind() const override { return NodeKind::PROJ; }
};

/**
 * CProjNode - Control Projection node.
 * Used for If true/false branches and Start control projection.
 */
class CProjNode : public CFGNode {
    int _idx;  // 0 for true/ctrl, 1 for false
    ::std::string _label;

public:
    CProjNode(Node* ctrl, int idx, const ::std::string& label = "")
        : CFGNode(), _idx(idx), _label(label) {
        setInput(0, ctrl);
    }

    bool isCFG() const override { return true; }

    bool blockHead() const override {
        // Only starts a BB if projecting from If
        return dynamic_cast<IfNode*>(in(0)) != nullptr;
    }

    ::std::string label() const override {
        if (!_label.empty()) return _label;
        return ::std::string("CProj[") + (_idx ? "F" : "T") + "]";
    }

    CFGNode* idom() override { return dynamic_cast<CFGNode*>(in(0)); }

    int idepth() override {
        if (_idepth != 0) return _idepth;
        CFGNode* dom = idom();
        if (!dom) return _idepth = 1;
        return _idepth = dom->idepth() + 1;
    }

    int loopDepth() override {
        if (_loopDepth != 0) return _loopDepth;
        CFGNode* cfg = dynamic_cast<CFGNode*>(in(0));
        if (!cfg) return _loopDepth = 1;
        return _loopDepth = cfg->loopDepth();
    }

    int idx() const { return _idx; }
    NodeKind getKind() const override { return NodeKind::PROJ; }  // Control projection
};

/**
 * NeverNode - Special If node that never executes.
 * Used to handle infinite loops by creating dummy edges to Stop.
 */
class NeverNode : public IfNode {
    int _idepth = 0;
    int _loopDepth = 0;
public:
    NeverNode(Node* ctrl) : IfNode(ctrl, nullptr) {}

    ::std::string label() const override { return "Never"; }

    // Never executes, so predicate is always false
    Type* compute() override;
    Node* peephole() override;  // Chapter 19: peephole for NeverNode
    NodeKind getKind() const override { return NodeKind::IF; }  // Special If node
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
    ::std::string label() const override { return "=="; }
    NodeKind getKind() const override { return NodeKind::EQ; }
};

class LTNode : public BoolNode {
public:
    LTNode(Node* lhs, Node* rhs) : BoolNode(lhs, rhs) {}
    Node* peephole() override; // fold if both constant
    ::std::string label() const override { return "<"; }
    NodeKind getKind() const override { return NodeKind::LT; }
};

/**
 * Add node - adds two values.
 * Following Simple compiler Chapter 2.
 */
class AddNode : public Node {
public:
    AddNode(Node* lhs, Node* rhs);

    ::std::string label() const override { return "+"; }
    Type* compute() override;
    NodeKind getKind() const override { return NodeKind::ADD; }
};

/**
 * Subtract node - subtracts rhs from lhs.
 * Following Simple compiler Chapter 2.
 */
class SubNode : public Node {
public:
    SubNode(Node* lhs, Node* rhs);

    ::std::string label() const override { return "-"; }
    Type* compute() override;
    NodeKind getKind() const override { return NodeKind::SUB; }
};

/**
 * Multiply node - multiplies two values.
 * Following Simple compiler Chapter 2.
 */
class MulNode : public Node {
public:
    MulNode(Node* lhs, Node* rhs);

    ::std::string label() const override { return "*"; }
    Type* compute() override;
    NodeKind getKind() const override { return NodeKind::MUL; }
};

/**
 * Divide node - divides lhs by rhs.
 * Following Simple compiler Chapter 2.
 */
class DivNode : public Node {
public:
    DivNode(Node* lhs, Node* rhs);

    ::std::string label() const override { return "/"; }
    Type* compute() override;
    NodeKind getKind() const override { return NodeKind::DIV; }
};

/**
 * Chapter 16: Bitwise Operation Nodes
 * Following Simple compiler Chapter 16.
 */

/**
 * Bitwise AND node - performs bitwise AND operation.
 */
class AndNode : public Node {
public:
    AndNode(Node* lhs, Node* rhs);

    ::std::string label() const override { return "&"; }
    Type* compute() override;
    NodeKind getKind() const override { return NodeKind::AND; }
};

/**
 * Bitwise OR node - performs bitwise OR operation.
 */
class OrNode : public Node {
public:
    OrNode(Node* lhs, Node* rhs);

    ::std::string label() const override { return "|"; }
    Type* compute() override;
    NodeKind getKind() const override { return NodeKind::OR; }
};

/**
 * Bitwise XOR node - performs bitwise XOR operation.
 */
class XorNode : public Node {
public:
    XorNode(Node* lhs, Node* rhs);

    ::std::string label() const override { return "^"; }
    Type* compute() override;
    NodeKind getKind() const override { return NodeKind::XOR; }
};

/**
 * Shift left node - performs left shift operation.
 */
class ShlNode : public Node {
public:
    ShlNode(Node* value, Node* shift);

    ::std::string label() const override { return "<<"; }
    Type* compute() override;
    NodeKind getKind() const override { return NodeKind::SHL; }
};

/**
 * Arithmetic shift right node - performs arithmetic right shift.
 */
class AShrNode : public Node {
public:
    AShrNode(Node* value, Node* shift);

    ::std::string label() const override { return ">>>"; }
    Type* compute() override;
    NodeKind getKind() const override { return NodeKind::ASHR; }
};

/**
 * Logical shift right node - performs logical right shift.
 */
class LShrNode : public Node {
public:
    LShrNode(Node* value, Node* shift);

    ::std::string label() const override { return ">>"; }
    Type* compute() override;
    NodeKind getKind() const override { return NodeKind::LSHR; }
};

/**
 * Unary minus node - negates a value.
 * Following Simple compiler Chapter 2.
 */
class MinusNode : public Node {
public:
    MinusNode(Node* value);

    ::std::string label() const override { return "-"; }
    Type* compute() override;
    NodeKind getKind() const override { return NodeKind::NEG; }
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
    ::std::string label() const override { return "Break"; }
    NodeKind getKind() const override { return NodeKind::BREAK; }
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
    ::std::string label() const override { return "Continue"; }
    NodeKind getKind() const override { return NodeKind::CONTINUE; }
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
 * Extended in Chapter 16 for constructor support.
 *
 * Takes control as input to ensure proper scheduling.
 * Returns a pointer to the newly allocated struct.
 */
class NewNode : public Node {
    ::std::string _structType;  // Name of struct type being allocated
    TypeStruct* _type;        // Chapter 16: Struct type metadata
    ::std::unordered_map<::std::string, Node*> _fieldInits;  // Chapter 16: Field initializers

public:
    NewNode(Node* ctrl, const ::std::string& structType, TypeStruct* type = nullptr)
        : Node(), _structType(structType), _type(type) { setInput(0, ctrl); }

    ::std::string label() const override { return "New[" + _structType + "]"; }
    bool hasSideEffects() const override { return true; }
    NodeKind getKind() const override { return NodeKind::ALLOC; }

    const ::std::string& structType() const { return _structType; }

    /**
     * Chapter 16: Add a field initializer.
     * Used during constructor parsing: new Point { x=3; y=4; }
     */
    void setFieldInit(const ::std::string& fieldName, Node* value) {
        _fieldInits[fieldName] = value;
    }

    /**
     * Chapter 16: Get field initializer (if any).
     */
    Node* getFieldInit(const ::std::string& fieldName) const {
        auto it = _fieldInits.find(fieldName);
        return it != _fieldInits.end() ? it->second : nullptr;
    }

    /**
     * Chapter 16: Get all field initializers.
     */
    const ::std::unordered_map<::std::string, Node*>& fieldInits() const {
        return _fieldInits;
    }

    /**
     * Chapter 16: Validate that all required fields are initialized.
     * Returns true if valid, false otherwise.
     */
    bool validateInitialization() const;

    /**
     * Chapter 16: Get the TypeStruct metadata.
     */
    TypeStruct* getStructType() const { return _type; }
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
    ::std::string _fieldName;  // For debugging

public:
    LoadNode(int alias, Node* mem, Node* ptr, Node* offset, const ::std::string& field)
        : MemOpNode(alias), _fieldName(field) {
        setInput(0, mem);
        setInput(1, ptr);
        setInput(2, offset);
    }

    ::std::string label() const override { return "Load[" + _fieldName + "]"; }

    NodeKind getKind() const override { return NodeKind::LOAD; }

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
    ::std::string _fieldName;  // For debugging

public:
    StoreNode(int alias, Node* mem, Node* ptr, Node* offset, Node* value, const ::std::string& field)
        : MemOpNode(alias), _fieldName(field) {
        setInput(0, mem);
        setInput(1, ptr);
        setInput(2, offset);
        setInput(3, value);
    }

    ::std::string label() const override { return "Store[" + _fieldName + "]"; }
    bool hasSideEffects() const override { return true; }

    NodeKind getKind() const override { return NodeKind::STORE; }

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

    ::std::string label() const override { return "MemProj[" + ::std::to_string(_alias) + "]"; }
    int alias() const { return _alias; }

    NodeKind getKind() const override { return NodeKind::PROJ; }
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

    ::std::string label() const override { return "Cast"; }

    NodeKind getKind() const override { return NodeKind::CAST; }
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
    ::std::vector<::std::unordered_map<::std::string, int>> _scopes;

    /**
     * Track the next available input index for new variables.
     */
    int _nextInputIdx;

    ScopeNode();

    ::std::string label() const override { return "Scope"; }
    NodeKind getKind() const override { return NodeKind::CONSTANT; }  // Symbol table scope

    // Duplicate current scope bindings into a new ScopeNode
    // If forLoop is true, sets up lazy phi sentinels (Chapter 8)
    ScopeNode* duplicate(bool forLoop = false) const;

    // Expose a snapshot of current visible variables -> Nodes
    ::std::unordered_map<::std::string, Node*> currentBindings() const;

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
    int define(const ::std::string& name, Node* value);

    /**
     * Update a variable's value.
     * Looks up the variable in the scope stack and updates its value.
     */
    void update(const ::std::string& name, Node* value);

    /**
     * Lookup a variable in the scope stack.
     * Returns nullptr if not found.
     */
    Node* lookup(const ::std::string& name) const;

    /**
     * Check if a variable exists in any scope.
     */
    bool contains(const ::std::string& name) const;

    /**
     * Get the current scope level (depth of scope stack).
     */
    int scopeLevel() const { return _scopes.size(); }
};

// ============================================================================
// Band 4: Type System Extension Nodes (Chapters 12-15)
// ============================================================================

/**
 * Chapter 12-14: Type Conversion/Coercion Nodes
 *
 * Extended type conversions for Band 4 type system.
 * Uses existing CastNode (line ~662) but adds specialized nodes for:
 * - Floating point conversions
 * - Narrow/wide integer conversions
 * - Array bounds enforcement
 *
 * Note: The existing CastNode (ctrl, value, toType) is kept for compatibility.
 * New type-specific conversion nodes could be added here if needed.
 */

/**
 * Chapter 15: Array Allocation Node
 *
 * Allocates a new array with specified length.
 * Input 0: control
 * Input 1: length (integer expression)
 */
class NewArrayNode : public Node {
    Type* _element_type;

public:
    NewArrayNode(Node* ctrl, Node* length, Type* elem_type)
        : Node(), _element_type(elem_type) {
        setInput(0, ctrl);
        setInput(1, length);
    }

    Type* elementType() const { return _element_type; }

    ::std::string label() const override {
        return "NewArray[" + _element_type->toString() + "]";
    }

    bool hasSideEffects() const override { return true; }

    Type* compute() override;
};

/**
 * Chapter 15: Array Load Node
 *
 * Loads an element from an array.
 * Input 0: memory
 * Input 1: array pointer
 * Input 2: index
 */
class ArrayLoadNode : public MemOpNode {
public:
    ArrayLoadNode(int alias, Node* mem, Node* array, Node* index)
        : MemOpNode(alias) {
        setInput(0, mem);
        setInput(1, array);
        setInput(2, index);
    }

    ::std::string label() const override { return "ALoad"; }

    Node* mem() const { return in(0); }
    Node* array() const { return in(1); }
    Node* index() const { return in(2); }

    Type* compute() override;
    Node* peephole() override;
};

/**
 * Chapter 15: Array Store Node
 *
 * Stores a value to an array element.
 * Input 0: memory
 * Input 1: array pointer
 * Input 2: index
 * Input 3: value to store
 */
class ArrayStoreNode : public MemOpNode {
public:
    ArrayStoreNode(int alias, Node* mem, Node* array, Node* index, Node* value)
        : MemOpNode(alias) {
        setInput(0, mem);
        setInput(1, array);
        setInput(2, index);
        setInput(3, value);
    }

    ::std::string label() const override { return "AStore"; }

    bool hasSideEffects() const override { return true; }

    Node* mem() const { return in(0); }
    Node* array() const { return in(1); }
    Node* index() const { return in(2); }
    Node* value() const { return in(3); }

    Type* compute() override;
    Node* peephole() override;
};

/**
 * Chapter 15: Array Length Node
 *
 * Returns the length of an array (using '#' postfix operator).
 * Input 0: array pointer
 */
class ArrayLengthNode : public Node {
public:
    ArrayLengthNode(Node* array) : Node() {
        setInput(0, array);
    }

    ::std::string label() const override { return "ArrayLength"; }

    Node* array() const { return in(0); }

    Type* compute() override;
    Node* peephole() override;
};

/**
 * Chapter 18: Function Node
 *
 * Represents a function definition. Extends RegionNode to merge all call sites.
 * Input 0: START node (for linking)
 * Has ParmNodes for parameters and a ReturnNode for the function body.
 */
class FunNode : public RegionNode {
protected:
    TypeFunPtr* _sig;       // Function signature
    ReturnNode* _ret;       // Single return point for all returns in function
    bool _folding;          // True if function is being inlined/folded

public:
    FunNode(Node* start, TypeFunPtr* sig)
        : RegionNode(start, nullptr), _sig(sig), _ret(nullptr), _folding(false) {}

    ::std::string label() const override { return "Fun"; }
    NodeKind getKind() const override { return NodeKind::FUNCTION; }

    TypeFunPtr* sig() const { return _sig; }
    ReturnNode* ret() const { return _ret; }
    void setRet(ReturnNode* ret) { _ret = ret; }

    bool folding() const { return _folding; }
    void setFolding(bool f) { _folding = f; }

    // Check if function is in progress (has unknown callers)
    bool inProgress() const;

    Type* compute() override;
    Node* peephole() override;
};

/**
 * Chapter 18: Parameter Node
 *
 * Represents a function parameter. Extends PhiNode to merge arguments from all call sites.
 * Input 0: FunNode
 * Additional inputs: arguments from each call site
 */
class ParmNode : public PhiNode {
private:
    int _idx;  // Parameter index (0 = RPC, 1 = memory, 2+ = actual parameters)

public:
    ParmNode(const ::std::string& label, int idx, Type* declaredType, Node* fun)
        : PhiNode(label, nullptr, nullptr, nullptr), _idx(idx) {
        setInput(0, fun);
    }

    ::std::string label() const override { return "Parm_" + _label; }
    NodeKind getKind() const override { return NodeKind::PARAMETER; }

    int idx() const { return _idx; }
    FunNode* fun() const { return static_cast<FunNode*>(in(0)); }

    Type* compute() override;
    Node* peephole() override;
};

/**
 * Chapter 18: Call Node
 *
 * Represents a function call.
 * Input 0: control
 * Input 1: memory
 * Input 2+: arguments
 * Last input: function pointer
 */
class CallNode : public CFGNode {
public:
    CallNode(Node* ctrl, Node* mem, Node* fptr) : CFGNode() {
        setInput(0, ctrl);
        setInput(1, mem);
        setInput(2, fptr);  // Function pointer is last input
    }

    ::std::string label() const override { return "Call"; }
    NodeKind getKind() const override { return NodeKind::CALL; }

    Node* ctrl() const { return in(0); }
    Node* mem() const { return in(1); }
    Node* fptr() const { return in(nIns() - 1); }  // Function pointer is last

    // Arguments are inputs 2 to nIns()-2
    int nArgs() const { return nIns() - 3; }
    Node* arg(int i) const {
        if (i >= 0 && i < nArgs()) return in(i + 2);
        return nullptr;
    }

    // Find the CallEnd from this Call
    CallEndNode* cend() const;

    Type* compute() override;
    Node* peephole() override;
};

/**
 * Chapter 18: Call End Node
 *
 * Represents the end of a function call with projections for control, memory, and return value.
 * Input 0: CallNode
 * Additional inputs: all linked functions
 */
class CallEndNode : public CFGNode {
public:
    CallEndNode(CallNode* call) : CFGNode() {
        setInput(0, call);
    }

    ::std::string label() const override { return "CEnd"; }
    NodeKind getKind() const override { return NodeKind::CALL_END; }

    CallNode* call() const { return static_cast<CallNode*>(in(0)); }

    Type* compute() override;
    Node* peephole() override;
};

} // namespace cppfort::ir

#endif // CPPFORT_NODE_H
