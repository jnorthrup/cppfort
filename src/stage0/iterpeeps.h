#ifndef CPPFORT_ITERPEEPS_H
#define CPPFORT_ITERPEEPS_H

#include <vector>
#include <unordered_set>
#include "node.h"

namespace cppfort::ir {

/**
 * Iterative Peephole Optimizer - Following Simple compiler Chapter 9
 *
 * Applies peephole optimizations iteratively until a fixed point is reached.
 * Uses a worklist algorithm to efficiently process nodes that may benefit
 * from optimization.
 */
class IterPeeps {
private:
    /**
     * Worklist of nodes to process.
     * Invariant: A node is either on the worklist OR no peephole applies.
     */
    ::std::vector<Node*> _worklist;

    /**
     * Set for fast membership testing.
     */
    ::std::unordered_set<Node*> _inWorklist;

    /**
     * Add a node to the worklist if not already present.
     */
    void addToWorklist(Node* n);

    /**
     * Add all uses and defs of a node to the worklist.
     * This ensures we check the "neighborhood" after any change.
     */
    void addNeighbors(Node* n);

    /**
     * Process one node from the worklist.
     * Returns true if the graph changed.
     */
    bool processNode(Node* n);

    /**
     * Debug assert: verify that all nodes not on worklist
     * have no applicable peepholes.
     */
    bool progressOnList(Node* stop) const;

public:
    /**
     * Run iterative peephole optimization to fixed point.
     *
     * @param stop The Stop/Return node of the graph
     * @return Number of iterations performed
     */
    int iterate(Node* stop);

    /**
     * Clear the worklist and visited set.
     */
    void clear();
};

} // namespace cppfort::ir

#endif // CPPFORT_ITERPEEPS_H