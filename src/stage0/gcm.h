#pragma once

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <bitset>
#include <iostream>
#include <queue>
#include <set>
#include <stack>

#include "node.h"
#include "../utils/multi_index.h"

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

// Use existing node definitions from node.h - no GCM-specific versions needed

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
    NodeKind getKind() const override { return NodeKind::REGION; }  // Dead control region
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
