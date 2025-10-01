#include "region_curing.h"
#include "node.h"
#include <queue>
#include <map>
#include <set>

namespace cppfort::ir {

std::unique_ptr<Region> RegionBuilder::buildRegions(Node* startNode) {
    auto region = std::make_unique<Region>();

    // Identify block boundaries in the Sea of Nodes graph
    std::vector<Node*> blockLeaders;
    identifyBlocks(startNode, blockLeaders);

    // Map nodes to their containing blocks
    std::map<Node*, Block*> nodeToBlock;

    // Create blocks and assign nodes
    assignNodesToBlocks(region.get(), blockLeaders, nodeToBlock);

    // Establish control flow and terminators
    establishControlFlow(nodeToBlock);

    // Convert phi nodes to block arguments
    convertPhisToBlockArgs(region.get());

    return region;
}

void RegionBuilder::identifyBlocks(Node* startNode, std::vector<Node*>& blockLeaders) {
    std::set<Node*> visited;
    std::queue<Node*> worklist;

    worklist.push(startNode);
    visited.insert(startNode);

    // Start node is always a block leader
    blockLeaders.push_back(startNode);

    while (!worklist.empty()) {
        Node* node = worklist.front();
        worklist.pop();

        // Check all outputs for nodes that start new blocks
        for (Node* output : node->_outputs) {
            if (visited.find(output) == visited.end()) {
                visited.insert(output);

                // CFG nodes start new blocks
                if (output->isCFG()) {
                    blockLeaders.push_back(output);
                }

                worklist.push(output);
            }
        }
    }
}

void RegionBuilder::assignNodesToBlocks(Region* region, const std::vector<Node*>& blockLeaders,
                                       std::map<Node*, Block*>& nodeToBlock) {
    // For each block leader, create a block and assign reachable nodes
    for (size_t i = 0; i < blockLeaders.size(); ++i) {
        Node* leader = blockLeaders[i];
        Block* block = region->addBlock();

        // Start from leader and traverse until hitting another leader or terminator
        std::set<Node*> visited;
        std::queue<Node*> worklist;

        worklist.push(leader);
        visited.insert(leader);
        nodeToBlock[leader] = block;

        while (!worklist.empty()) {
            Node* node = worklist.front();
            worklist.pop();

            block->addOperation(node);

            // Stop at terminators (control flow nodes)
            if (node->isCFG() && node != leader) {
                block->setTerminator(node);
                break;
            }

            // Continue to non-CFG outputs
            for (Node* output : node->_outputs) {
                if (visited.find(output) == visited.end() &&
                    std::find(blockLeaders.begin(), blockLeaders.end(), output) == blockLeaders.end()) {
                    visited.insert(output);
                    nodeToBlock[output] = block;
                    worklist.push(output);
                }
            }
        }
    }
}

void RegionBuilder::establishControlFlow(std::map<Node*, Block*>& nodeToBlock) {
    // For each block, determine successors based on terminator
    for (auto& pair : nodeToBlock) {
        Block* block = pair.second;
        Node* terminator = block->terminator();

        if (!terminator) continue;

        // Different terminators create different control flow
        if (auto* ifNode = dynamic_cast<IfNode*>(terminator)) {
            // If nodes have true/false projections
            // Successors would be determined by projection targets
        } else if (auto* returnNode = dynamic_cast<ReturnNode*>(terminator)) {
            // Return nodes end the flow
        }
        // Add other terminator types as needed
    }
}

void RegionBuilder::convertPhisToBlockArgs(Region* region) {
    // Find all phi nodes and convert them to block arguments
    for (auto& blockPtr : region->blocks()) {
        Block* block = blockPtr.get();

        // Look for phi nodes in this block
        for (Node* op : block->operations()) {
            if (auto* phi = dynamic_cast<PhiNode*>(op)) {
                // Convert phi to block argument
                block->addArgument(phi);
            }
        }
    }
}

} // namespace cppfort::ir
