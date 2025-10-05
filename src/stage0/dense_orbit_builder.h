#pragma once

#include <vector>
#include <stack>
#include <string>
#include "orbit_ring.h"

namespace cppfort::stage0 {

/**
 * Densified Orbit Builder - constructs hierarchical orbit structures with optimal cache locality
 *
 * Uses packed 12-byte ring structures in contiguous arena allocation for:
 * - 4x memory reduction vs pointer-based trees
 * - Hardware prefetching of sequential ring access
 * - SIMD processing of multiple rings simultaneously
 */
class DenseOrbitBuilder {
private:
    OrbitArena arena;
    std::stack<uint16_t> scope_stack;  // Stack of ring indices for current scope hierarchy
    std::vector<uint16_t> root_rings;  // Top-level rings (no parent)

public:
    DenseOrbitBuilder() = default;

    /**
     * Start a new orbit ring at the given position
     * @param open_pos Opening delimiter position
     * @param delim_type Encoded delimiter type (0-8)
     * @param depth Current nesting depth
     * @return Index of the newly allocated ring
     */
    uint16_t startRing(uint32_t open_pos, uint8_t delim_type, uint8_t depth);

    /**
     * Complete a ring by setting its close position
     * @param ring_idx Index of ring to complete
     * @param close_pos Closing delimiter position
     */
    void completeRing(uint16_t ring_idx, uint32_t close_pos);

    /**
     * Push a ring onto the scope stack (enter new scope)
     * @param ring_idx Ring index to push
     */
    void pushScope(uint16_t ring_idx);

    /**
     * Pop the current scope from the stack (exit scope)
     * @return Index of the ring that was popped, or 0xFFFF if stack empty
     */
    uint16_t popScope();

    /**
     * Get current scope depth
     */
    size_t getScopeDepth() const { return scope_stack.size(); }

    /**
     * Get the ring at the top of the scope stack
     */
    uint16_t getCurrentScope() const {
        return scope_stack.empty() ? 0xFFFF : scope_stack.top();
    }

    /**
     * Get all root rings (top-level scopes)
     */
    const std::vector<uint16_t>& getRootRings() const { return root_rings; }

    /**
     * Get access to the underlying arena
     */
    const OrbitArena& getArena() const { return arena; }
    OrbitArena& getArena() { return arena; }

    /**
     * Reset builder state
     */
    void reset();

    /**
     * Build orbit hierarchy from source code using densified scanning
     * @param code Source code to analyze
     * @return Vector of root ring indices
     */
    std::vector<uint16_t> buildOrbits(const std::string& code);

    /**
     * SIMD-friendly orbit traversal - process rings in depth-first order
     * @param visitor Function to call for each ring (receives ring index and depth)
     */
    template<typename Visitor>
    void traverseDFS(Visitor&& visitor) const;

    /**
     * SIMD-friendly orbit traversal - process rings in breadth-first order
     * @param visitor Function to call for each ring (receives ring index and depth)
     */
    template<typename Visitor>
    void traverseBFS(Visitor&& visitor) const;
};

// Template implementations
template<typename Visitor>
void DenseOrbitBuilder::traverseDFS(Visitor&& visitor) const {
    if (root_rings.empty()) return;

    struct StackEntry {
        uint16_t ring_idx;
        uint8_t depth;
    };

    std::vector<StackEntry> stack;
    for (auto root_idx : root_rings) {
        stack.push_back({root_idx, 0});
    }

    while (!stack.empty()) {
        StackEntry entry = stack.back();
        stack.pop_back();

        visitor(entry.ring_idx, entry.depth);

        // Find child rings (rings that have this as parent)
        // Note: This is a simplified traversal - in practice you'd need
        // to maintain parent->child relationships or scan all rings
        const auto& rings = arena.getAllRings();
        for (uint16_t i = 0; i < rings.size(); ++i) {
            if (rings[i].parent_idx == entry.ring_idx) {
                stack.push_back({i, static_cast<uint8_t>(entry.depth + 1)});
            }
        }
    }
}

template<typename Visitor>
void DenseOrbitBuilder::traverseBFS(Visitor&& visitor) const {
    if (root_rings.empty()) return;

    std::vector<std::pair<uint16_t, uint8_t>> queue;
    for (auto root_idx : root_rings) {
        queue.emplace_back(root_idx, 0);
    }

    size_t front = 0;
    while (front < queue.size()) {
        auto [ring_idx, depth] = queue[front++];
        visitor(ring_idx, depth);

        // Find child rings
        const auto& rings = arena.getAllRings();
        for (uint16_t i = 0; i < rings.size(); ++i) {
            if (rings[i].parent_idx == ring_idx) {
                queue.emplace_back(i, static_cast<uint8_t>(depth + 1));
            }
        }
    }
}

} // namespace cppfort::stage0