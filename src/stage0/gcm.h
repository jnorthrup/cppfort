#ifndef CPPFORT_GCM_H
#define CPPFORT_GCM_H

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <bitset>
#include <iostream>
#include <queue>
#include <set>
#include <stack>

#include "node.h"

namespace cppfort::ir {

/**
 * Band 3: Chapter 11 - Global Code Motion
 *
 * This is THE critical convergence point where the unordered Sea of Nodes
 * graph becomes scheduled code. The GCM algorithm transforms floating data
 * nodes into a concrete execution schedule while preserving semantics.
 *
 * Following Simple compiler Chapter 11 implementation.
 */

// Forward declarations
class CFGNode;
class LoopNode;
class NeverNode;

// Use existing node definitions from node.h - no GCM-specific versions needed

/**
 * CProj - Control Projection node.
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

    bool blockHead() const {
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
};

/**
 * XCtrlNode - Dead control node.
 * Represents unreachable control flow.
 */
class XCtrlNode : public CFGNode {
public:
    ::std::string label() const override { return "XCtrl"; }

    CFGNode* idom() override { return nullptr; }
    int idepth() override { return 1; }
    int loopDepth() override { return 1; }
};

/**
 * Global Code Motion scheduler.
 *
 * Transforms the Sea of Nodes from an unordered graph into
 * a scheduled sequence of operations.
 */
class GlobalCodeMotion {
private:
    StartNode* _start;
    StopNode* _stop;

    // Scheduling results
    ::std::unordered_map<int, CFGNode*> _early;  // Early schedule
    ::std::unordered_map<int, CFGNode*> _late;   // Late schedule

    // Helper methods
    void fixInfiniteLoops();
    void scheduleEarly();
    void scheduleLate();
    void insertAntiDeps();

    // Early scheduling helpers
    void rpoWalk(CFGNode* n, ::std::unordered_set<int>& visited, ::std::vector<CFGNode*>& rpo);
    void schedEarlyNode(Node* n, ::std::unordered_set<int>& visited);

    // Late scheduling helpers
    void schedLateNode(Node* n, ::std::vector<Node*>& nodes);
    CFGNode* useBlock(Node* n, Node* use);
    bool better(CFGNode* lca, CFGNode* best);
    bool isForwardEdge(Node* use, Node* def);

    // Anti-dependency helpers
    CFGNode* findAntiDep(CFGNode* lca, LoadNode* load, CFGNode* early);
    CFGNode* antiDep(LoadNode* load, CFGNode* stblk, CFGNode* defblk, CFGNode* lca, Node* st);

public:
    GlobalCodeMotion(StartNode* start, StopNode* stop)
        : _start(start), _stop(stop) {}

    /**
     * Run the complete Global Code Motion algorithm.
     *
     * Phases:
     * 1. Fix infinite loops by adding dummy edges to Stop
     * 2. Compute early schedule (as early as possible)
     * 3. Compute late schedule (as late as possible)
     * 4. Insert anti-dependencies for memory operations
     */
    void schedule();

    /**
     * Get the scheduled block for a node.
     */
    CFGNode* getBlock(Node* n) const {
        auto it = _late.find(n->_nid);
        return it != _late.end() ? it->second : nullptr;
    }

    /**
     * Debug: print the late schedule.
     */
    void debugLateSchedule() const {
        ::std::cout << "Late schedule:\n";
        for (const auto& [nid, block] : _late) {
            ::std::cout << "  Node " << nid << " -> " << (block ? block->label() : "null") << "\n";
        }
    }

    /**
     * Check if a node is pinned (has fixed control).
     */
    static bool isPinned(Node* n) {
        // CFG nodes are always pinned
        if (n->isCFG()) return true;

        // Nodes with control input are pinned
        if (n->in(0) && n->in(0)->isCFG()) return true;

        // Phi nodes are pinned to their regions
        if (dynamic_cast<PhiNode*>(n)) return true;

        // Memory operations with side effects are pinned
        if (n->hasSideEffects()) return true;

        return false;
    }
};

} // namespace cppfort::ir

#endif // CPPFORT_GCM_H