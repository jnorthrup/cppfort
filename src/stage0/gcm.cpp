#include "gcm.h"
#include <algorithm>
#include <cassert>
#include <queue>
#include <set>
#include <stack>
#include <unordered_set>
#include <vector>

namespace cppfort::ir {

// Note: Node implementations for Stop, Region, Loop idom/idepth/loopDepth
// are now in node.cpp to avoid duplication

// ============================================================================
// NeverNode implementation
// ============================================================================

Type* NeverNode::compute() {
    // Never executes, predicate is always false
    return TypeInteger::boolFalse();  // Always false
}

Node* NeverNode::peephole() {
    // Never executes, no optimization possible
    return nullptr;
}

// ============================================================================
// GlobalCodeMotion implementations
// ============================================================================

void GlobalCodeMotion::schedule() {
    // Phase 1: Fix infinite loops
    // NOTE: fixInfiniteLoops() is currently a stub and does not implement loop-exit detection.
    // To avoid relying on unimplemented behavior, the scheduler will skip this phase until
    // a proper implementation is provided.
    // TODO: Implement loop exit detection and re-enable this call.
    // fixInfiniteLoops();

    // Phase 2: Early schedule (as early as possible)
    scheduleEarly();

    // Phase 3: Late schedule (as late as possible)
    scheduleLate();

    // Phase 4: Insert anti-dependencies for memory operations
    insertAntiDeps();
}

void GlobalCodeMotion::fixInfiniteLoops() {
    // Backward reachability from Stop: find nodes that can reach Stop.
    ::std::unordered_set<int> reachable;
    ::std::stack<Node*> st;

    st.push(_stop);
    while (!st.empty()) {
        Node* n = st.top();
        st.pop();
        if (!n) continue;
        if (!reachable.insert(n->_nid).second) continue;
        for (Node* in : n->_inputs) {
            if (in) st.push(in);
        }
    }

    // Collect CFG nodes (RPO) to find loops
    ::std::vector<CFGNode*> rpo;
    ::std::unordered_set<int> visited;
    rpoWalk(static_cast<CFGNode*>(_start), visited, rpo);

    for (CFGNode* cfg : rpo) {
        if (LoopNode* loop = dynamic_cast<LoopNode*>(cfg)) {
            // If the loop header cannot reach Stop (not in reachable set),
            // consider it an infinite loop without an exit and force an exit.
            if (reachable.find(loop->_nid) == reachable.end()) {
                // Create loop exit using LoopNode::forceExit which wires Never/CProj/Return
                loop->forceExit(_stop);

                // Mark the newly-created nodes as reachable so we don't duplicate
                // (they were just appended via constructors)
                // Walk from the Stop inputs to add new nodes
                st.push(_stop);
                while (!st.empty()) {
                    Node* n = st.top(); st.pop();
                    if (!n) continue;
                    if (!reachable.insert(n->_nid).second) continue;
                    for (Node* in : n->_inputs) if (in) st.push(in);
                }
            }
        }
    }
}

void GlobalCodeMotion::scheduleEarly() {
    // Build reverse post-order of CFG nodes
    ::std::vector<CFGNode*> rpo;
    ::std::unordered_set<int> visited;

    rpoWalk(static_cast<CFGNode*>(_start), visited, rpo);

    // Process in reverse order (post-order)
    for (auto it = rpo.rbegin(); it != rpo.rend(); ++it) {
        CFGNode* cfg = *it;

        // Compute loop depth
        cfg->loopDepth();

        // Schedule inputs
        for (Node* n : cfg->_inputs) {
            schedEarlyNode(n, visited);
        }

        // For regions, also schedule Phi nodes
        if (RegionNode* region = dynamic_cast<RegionNode*>(cfg)) {
            int len = region->_outputs.size();
            for (int i = 0; i < len; i++) {
                if (PhiNode* phi = dynamic_cast<PhiNode*>(region->_outputs[i])) {
                    schedEarlyNode(phi, visited);
                }
            }
        }
    }
}

void GlobalCodeMotion::rpoWalk(CFGNode* n, ::std::unordered_set<int>& visited,
                                ::std::vector<CFGNode*>& rpo) {
    if (!n || visited.find(n->_nid) != visited.end()) {
        return;
    }

    visited.insert(n->_nid);

    // Visit outputs first (for post-order)
    for (Node* use : n->_outputs) {
        if (CFGNode* cfg = dynamic_cast<CFGNode*>(use)) {
            rpoWalk(cfg, visited, rpo);
        }
    }

    rpo.push_back(n);
}

void GlobalCodeMotion::schedEarlyNode(Node* n, ::std::unordered_set<int>& visited) {
    if (!n || visited.find(n->_nid) != visited.end()) {
        return;
    }

    visited.insert(n->_nid);

    // Schedule unpinned inputs first
    for (Node* def : n->_inputs) {
        if (def && !isPinned(def)) {
            schedEarlyNode(def, visited);
        }
    }

    // If not pinned, find earliest legal position
    if (!isPinned(n)) {
        CFGNode* early = static_cast<CFGNode*>(_start);

        // Find deepest input
        for (int i = 1; i < n->nIns(); i++) {
            if (Node* inp = n->in(i)) {
                CFGNode* cfg = nullptr;

                if (inp->isCFG()) {
                    cfg = dynamic_cast<CFGNode*>(inp);
                } else if (_early.find(inp->_nid) != _early.end()) {
                    cfg = _early[inp->_nid];
                }

                if (cfg && cfg->idepth() > early->idepth()) {
                    early = cfg;
                }
            }
        }

        // Set control input
        n->setInput(0, early);
        _early[n->_nid] = early;
    }
}

void GlobalCodeMotion::scheduleLate() {
    ::std::vector<Node*> nodes(Node::nextUniqueId());

    // Walk from Start in forward order
    schedLateNode(_start, nodes);

    // Apply the late schedule
    for (Node* n : nodes) {
        if (n && _late.find(n->_nid) != _late.end()) {
            n->setInput(0, _late[n->_nid]);
        }
    }
}

void GlobalCodeMotion::schedLateNode(Node* n, ::std::vector<Node*>& nodes) {
    if (!n || _late.find(n->_nid) != _late.end()) {
        return;
    }

    // Handle CFG nodes and Phi nodes specially
    if (CFGNode* cfg = dynamic_cast<CFGNode*>(n)) {
        _late[n->_nid] = cfg->blockHead() ? cfg : cfg->idom();
    } else if (PhiNode* phi = dynamic_cast<PhiNode*>(n)) {
        if (RegionNode* r = dynamic_cast<RegionNode*>(phi->region())) {
            _late[n->_nid] = dynamic_cast<CFGNode*>(r);
        }
    }

    // Walk stores before loads for anti-deps
    for (Node* use : n->_outputs) {
        if (isForwardEdge(use, n) && dynamic_cast<StoreNode*>(use)) {
            schedLateNode(use, nodes);
        }
    }

    // Walk all outputs
    for (Node* use : n->_outputs) {
        if (isForwardEdge(use, n)) {
            schedLateNode(use, nodes);
        }
    }

    // Walk data inputs (inputs after control input)
    for (int i = 1; i < n->nIns(); i++) {
        if (Node* inp = n->in(i)) {
            schedLateNode(inp, nodes);
        }
    }

    // Skip if already scheduled or pinned
    if (isPinned(n)) return;

    // Find LCA of uses
    CFGNode* early = _early[n->_nid];
    CFGNode* lca = nullptr;

    for (Node* use : n->_outputs) {
        CFGNode* useBlock = this->useBlock(n, use);
        if (useBlock) {
            lca = useBlock->idom(lca);
        }
    }

    // Handle loads - may need anti-deps
    if (LoadNode* load = dynamic_cast<LoadNode*>(n)) {
        lca = findAntiDep(lca, load, early);
    }

    // Find best placement between early and lca
    CFGNode* best = lca;
    if (lca) {
        lca = lca->idom();
        while (lca && lca != early->idom()) {
            if (better(lca, best)) {
                best = lca;
            }
            lca = lca->idom();
        }
    }

    // Avoid scheduling in If nodes
    if (dynamic_cast<IfNode*>(best)) {
        best = best->idom();
    }

    nodes[n->_nid] = n;
    _late[n->_nid] = best ? best : early;
}

CFGNode* GlobalCodeMotion::useBlock(Node* n, Node* use) {
    // For Phi nodes, find the matching region input
    if (PhiNode* phi = dynamic_cast<PhiNode*>(use)) {
        RegionNode* region = dynamic_cast<RegionNode*>(phi->region());
        if (!region) return nullptr;

        for (int i = 1; i < phi->nIns(); i++) {
            if (phi->in(i) == n) {
                return dynamic_cast<CFGNode*>(region->in(i));
            }
        }
    }

    // Otherwise use the late schedule
    auto it = _late.find(use->_nid);
    return it != _late.end() ? it->second : nullptr;
}

bool GlobalCodeMotion::better(CFGNode* lca, CFGNode* best) {
    // Prefer shallower loop depth
    if (lca->getLoopDepth() < best->getLoopDepth()) return true;
    if (lca->getLoopDepth() > best->getLoopDepth()) return false;

    // At same loop depth, prefer deeper control (more control-dependent)
    if (lca->idepth() > best->idepth()) return true;

    // Avoid If nodes
    if (dynamic_cast<IfNode*>(best)) return true;

    return false;
}

bool GlobalCodeMotion::isForwardEdge(Node* use, Node* def) {
    if (!use || !def) return false;

    // Check for loop backedges
    if (use->nIns() > 2 && use->in(2) == def) {
        if (dynamic_cast<LoopNode*>(use)) return false;
        if (PhiNode* phi = dynamic_cast<PhiNode*>(use)) {
            if (dynamic_cast<LoopNode*>(phi->region())) {
                return false;
            }
        }
    }

    return true;
}

void GlobalCodeMotion::insertAntiDeps() {
    // Anti-dependencies are primarily computed during late scheduling.
    // Placeholder retained for any future post-processing that may be required.
    // TODO: Add tests and a post-processing hook if additional anti-dep insertion
    //       is needed beyond what scheduleLate() computes.
}

CFGNode* GlobalCodeMotion::findAntiDep(CFGNode* lca, LoadNode* load, CFGNode* early) {
    // Mark potential blocks for this load
    for (CFGNode* cfg = lca; cfg && cfg != early->idom(); cfg = cfg->idom()) {
        cfg->setAnti(load->_nid);
    }

    // Check memory uses for potential conflicts
    Node* mem = load->mem();
    if (!mem) return lca;

    for (Node* use : mem->_outputs) {
        if (StoreNode* store = dynamic_cast<StoreNode*>(use)) {
            // Only consider stores to same alias
            if (store->alias() == load->alias()) {
                auto it = _late.find(store->_nid);
                if (it != _late.end()) {
                    CFGNode* storeBlock = it->second;
                    lca = antiDep(load, storeBlock, dynamic_cast<CFGNode*>(mem->cfg0()), lca, store);
                }
            }
        } else if (PhiNode* phi = dynamic_cast<PhiNode*>(use)) {
            // Handle memory Phi nodes
            RegionNode* region = dynamic_cast<RegionNode*>(phi->region());
            if (region) {
                for (int i = 1; i < phi->nIns(); i++) {
                    if (phi->in(i) == mem) {
                        CFGNode* cfg = dynamic_cast<CFGNode*>(region->in(i));
                        if (cfg) {
                            lca = antiDep(load, cfg, dynamic_cast<CFGNode*>(mem->cfg0()), lca, nullptr);
                        }
                    }
                }
            }
        }
    }

    return lca;
}

CFGNode* GlobalCodeMotion::antiDep(LoadNode* load, CFGNode* stblk, CFGNode* defblk,
                                    CFGNode* lca, Node* st) {
    // Walk from store block to def block
    for (CFGNode* blk = stblk; blk && blk != defblk->idom(); blk = blk->idom()) {
        // Check if load and store overlap
        if (blk->getAnti() == load->_nid) {
            // Raise load's LCA
            lca = blk->idom(lca);

            // Add anti-dependency edge if needed
            if (lca == stblk && st) {
                bool found = false;
                for (Node* inp : st->_inputs) {
                    if (inp == load) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    st->_inputs.push_back(load);
                    load->_outputs.push_back(st);
                }
            }

            return lca;
        }
    }

    return lca;
}

} // namespace cppfort::ir