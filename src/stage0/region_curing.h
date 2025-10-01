#ifndef CPPFORT_REGION_CURING_H
#define CPPFORT_REGION_CURING_H

#include <vector>
#include <memory>
#include <map>
#include "node.h"

namespace cppfort::ir {

// Forward declarations for MLIR types
namespace mlir {
    class Block;
    class Region;
    class Operation;
    class Value;
    class Type;
}

/**
 * MLIR-style Block abstraction.
 *
 * A Block represents a sequence of operations that execute in order.
 * Blocks have arguments (phi nodes in Sea of Nodes terms) and contain
 * operations that may have nested regions.
 */
class Block {
private:
    int _id;
    ::std::vector<Node*> _arguments;        // Block arguments (phi nodes)
    ::std::vector<Node*> _operations;       // Operations in this block
    Node* _terminator = nullptr;          // Block terminator operation

    // MLIR integration
    mlir::Block* _mlirBlock = nullptr;

public:
    explicit Block(int id) : _id(id) {}

    int id() const { return _id; }

    // Block arguments (phis)
    void addArgument(Node* arg) { _arguments.push_back(arg); }
    const ::std::vector<Node*>& arguments() const { return _arguments; }
    Node* argument(int idx) const {
        return idx >= 0 && idx < _arguments.size() ? _arguments[idx] : nullptr;
    }

    // Operations in this block
    void addOperation(Node* op) { _operations.push_back(op); }
    const ::std::vector<Node*>& operations() const { return _operations; }

    // Terminator
    void setTerminator(Node* term) { _terminator = term; }
    Node* terminator() const { return _terminator; }

    // Check if block is properly terminated
    bool isTerminated() const { return _terminator != nullptr; }

    // MLIR integration
    void setMLIRBlock(mlir::Block* block) { _mlirBlock = block; }
    mlir::Block* mlirBlock() const { return _mlirBlock; }
};

/**
 * MLIR-style Region abstraction.
 *
 * A Region is a container for blocks. Regions can be nested within
 * operations (like control flow operations). The first block in a
 * region is the entry block.
 */
class Region {
private:
    ::std::vector<::std::unique_ptr<Block>> _blocks;
    Block* _entryBlock = nullptr;

    // MLIR integration
    mlir::Region* _mlirRegion = nullptr;

public:
    Region() = default;

    // Block management
    Block* addBlock() {
        auto block = ::std::make_unique<Block>(_blocks.size());
        Block* ptr = block.get();
        _blocks.push_back(::std::move(block));
        if (!_entryBlock) _entryBlock = ptr;
        return ptr;
    }

    Block* entryBlock() const { return _entryBlock; }
    const ::std::vector<::std::unique_ptr<Block>>& blocks() const { return _blocks; }

    Block* block(int idx) const {
        return idx >= 0 && idx < _blocks.size() ? _blocks[idx].get() : nullptr;
    }

    int numBlocks() const { return _blocks.size(); }

    // Check if region has valid structure
    bool isValid() const {
        return _entryBlock != nullptr && !_blocks.empty();
    }

    // MLIR integration
    void setMLIRRegion(mlir::Region* region) { _mlirRegion = region; }
    mlir::Region* mlirRegion() const { return _mlirRegion; }
};

/**
 * Converts Sea of Nodes graph to MLIR-style regions and blocks.
 *
 * This implements the MLIR structural conversion from the free-form
 * Sea of Nodes representation to the structured MLIR region/block model.
 */
class RegionBuilder {
public:
    /**
     * Build MLIR-style regions from Sea of Nodes graph.
     * Returns the top-level region containing all blocks.
     */
    ::std::unique_ptr<Region> buildRegions(Node* startNode);

private:
    /**
     * Identify block boundaries and create blocks.
     * Block leaders are CFG nodes that start new blocks.
     */
    void identifyBlocks(Node* startNode, ::std::vector<Node*>& blockLeaders);

    /**
     * Group nodes into blocks based on control flow.
     */
    void assignNodesToBlocks(Region* region, const ::std::vector<Node*>& blockLeaders,
                             ::std::map<Node*, Block*>& nodeToBlock);

    /**
     * Set block terminators and establish control flow edges.
     */
    void establishControlFlow(::std::map<Node*, Block*>& nodeToBlock);

    /**
     * Convert Sea of Nodes phis to block arguments.
     */
    void convertPhisToBlockArgs(Region* region);
};

} // namespace cppfort::ir

#endif // CPPFORT_REGION_CURING_H
