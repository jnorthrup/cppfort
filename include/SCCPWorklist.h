//===- SCCPWorklist.h - SCCP Worklist for Dataflow Analysis -------------------===//
///
/// Worklist implementation for Sparse Conditional Constant Propagation.
/// Tracks which operations need to be reprocessed when their lattice values
/// change during dataflow analysis. Prevents duplicates and provides FIFO
/// ordering for deterministic analysis.
///
//===----------------------------------------------------------------------===//

#ifndef CPPFORT_SCCP_WORKLIST_H
#define CPPFORT_SCCP_WORKLIST_H

#include <deque>
#include <unordered_set>
#include <cstddef>

namespace cppfort::sccp {

/// Worklist for SCCP dataflow analysis.
///
/// The worklist tracks which operations (represented as opaque pointers)
/// need to be reprocessed when their lattice values change. It:
/// - Prevents duplicate entries for the same operation
/// - Provides FIFO ordering for deterministic analysis
/// - Efficiently tracks membership for duplicate prevention
class SCCPWorklist {
private:
    std::deque<void*> queue;        ///< FIFO queue of items to process
    std::unordered_set<void*> set;  ///< Set for O(1) duplicate checking

public:
    /// Create an empty worklist
    SCCPWorklist() = default;

    /// Check if the worklist is empty
    bool empty() const {
        return queue.empty();
    }

    /// Get the number of items in the worklist
    size_t size() const {
        return queue.size();
    }

    /// Add an item to the worklist
    /// If the item is already in the worklist, this is a no-op.
    void enqueue(void* item) {
        if (set.find(item) == set.end()) {
            queue.push_back(item);
            set.insert(item);
        }
    }

    /// Remove and return the next item from the worklist (FIFO order)
    /// Returns nullptr if the worklist is empty
    void* dequeue() {
        if (queue.empty()) {
            return nullptr;
        }

        void* item = queue.front();
        queue.pop_front();
        set.erase(item);

        return item;
    }

    /// Remove all items from the worklist
    void clear() {
        queue.clear();
        set.clear();
    }

    /// Check if an item is currently in the worklist
    bool contains(void* item) const {
        return set.find(item) != set.end();
    }
};

} // namespace cppfort::sccp

#endif // CPPFORT_SCCP_WORKLIST_H
