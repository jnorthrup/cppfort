#include "iterpeeps.h"
#include <algorithm>
#include <cassert>
#include <queue>

namespace cppfort::ir {

void IterPeeps::addToWorklist(Node* n) {
    if (!n || n->isDead()) return;

    // Check if already in worklist
    if (_inWorklist.find(n) != _inWorklist.end()) return;

    _worklist.push_back(n);
    _inWorklist.insert(n);
}

void IterPeeps::addNeighbors(Node* n) {
    if (!n) return;

    // Add all uses (outputs)
    for (Node* use : n->_outputs) {
        addToWorklist(use);
    }

    // Add all defs (inputs)
    for (int i = 0; i < n->nIns(); i++) {
        addToWorklist(n->in(i));
    }

    // Add any registered dependencies (distant neighbors)
    for (Node* dep : n->deps()) {
        addToWorklist(dep);
    }
}

bool IterPeeps::processNode(Node* n) {
    if (!n || n->isDead()) return false;

    // Remove from worklist tracking
    _inWorklist.erase(n);

    // Apply peephole
    Node* improved = n->peephole();

    // If node changed, add neighbors to worklist
    if (improved != n) {
        // Add neighbors of both old and new nodes
        addNeighbors(n);
        addNeighbors(improved);

        // If the old node is dead, clean up its dependencies
        if (n->isDead()) {
            // Node was killed during peephole
            return true;
        }

        // If improved is different, subsume n with improved
        if (!n->isUnused()) {
            // Replace all uses of n with improved
            ::std::vector<Node*> users = n->_outputs;  // Copy because we'll modify
            for (Node* user : users) {
                for (int i = 0; i < user->nIns(); i++) {
                    if (user->in(i) == n) {
                        user->setInput(i, improved);
                    }
                }
            }

            // Now n should be unused and can be killed
            if (n->isUnused()) {
                n->kill();
            }
        }

        return true;
    }

    return false;
}

bool IterPeeps::progressOnList(Node* stop) const {
    // Debug check: verify invariant that nodes not on worklist
    // have no applicable peepholes

    // Collect all reachable nodes from stop
    ::std::unordered_set<Node*> visited;
    ::std::queue<Node*> toVisit;

    toVisit.push(stop);
    visited.insert(stop);

    while (!toVisit.empty()) {
        Node* current = toVisit.front();
        toVisit.pop();

        // Visit all inputs
        for (int i = 0; i < current->nIns(); i++) {
            Node* input = current->in(i);
            if (input && visited.find(input) == visited.end()) {
                visited.insert(input);
                toVisit.push(input);
            }
        }
    }

    // Check that all visited nodes either:
    // 1. Are on the worklist, OR
    // 2. Have no applicable peephole
    for (Node* n : visited) {
        if (_inWorklist.find(n) != _inWorklist.end()) {
            continue;  // On worklist, OK
        }

        // Try peephole - it should return the same node if no optimization
        Node* peep = n->peephole();
        if (peep != n) {
            // Found a node not on worklist that has applicable peephole!
            assert(false && "Node not on worklist has applicable peephole");
            return false;
        }
    }

    return true;
}

int IterPeeps::iterate(Node* stop) {
    if (!stop) return 0;

    // Initialize worklist with all nodes reachable from stop
    ::std::queue<Node*> toVisit;
    ::std::unordered_set<Node*> visited;

    toVisit.push(stop);
    visited.insert(stop);

    while (!toVisit.empty()) {
        Node* current = toVisit.front();
        toVisit.pop();

        addToWorklist(current);

        // Visit all inputs
        for (int i = 0; i < current->nIns(); i++) {
            Node* input = current->in(i);
            if (input && visited.find(input) == visited.end()) {
                visited.insert(input);
                toVisit.push(input);
            }
        }
    }

    // Process worklist until empty
    int iterations = 0;
    while (!_worklist.empty()) {
        iterations++;

        // Take node from worklist
        Node* n = _worklist.back();
        _worklist.pop_back();

        // Process it
        processNode(n);

        // Debug check (expensive, can be disabled in release)
#ifdef DEBUG
        assert(progressOnList(stop));
#endif
    }

    return iterations;
}

void IterPeeps::clear() {
    _worklist.clear();
    _inWorklist.clear();
}

} // namespace cppfort::ir