#include "cfg_recovery.h"
#include <algorithm>
#include <sstream>
#include <iostream>

namespace cppfort::stage2 {

ControlFlowGraph::ControlFlowGraph() = default;

ControlFlowGraph::~ControlFlowGraph() = default;

void ControlFlowGraph::buildFromInstructions(const std::vector<AsmInstruction>& instructions,
                                           const X86Analyzer& analyzer) {
    if (instructions.empty()) {
        return;
    }

    all_instructions_ = instructions;

    // Step 1: Identify basic block boundaries
    auto boundaries = identifyBlockBoundaries(instructions, analyzer);

    // Step 2: Create basic blocks
    createBasicBlocks(instructions, boundaries);

    // Step 3: Build control flow edges
    buildControlFlowEdges(analyzer);
}

const BasicBlock* ControlFlowGraph::getBasicBlock(uint64_t address) const {
    auto it = basic_blocks_.find(address);
    return it != basic_blocks_.end() ? &it->second : nullptr;
}

std::vector<const BasicBlock*> ControlFlowGraph::getAllBasicBlocks() const {
    std::vector<const BasicBlock*> blocks;
    blocks.reserve(basic_blocks_.size());
    for (const auto& pair : basic_blocks_) {
        blocks.push_back(&pair.second);
    }
    return blocks;
}

std::vector<uint64_t> ControlFlowGraph::getEntryPoints() const {
    std::vector<uint64_t> entries;
    for (const auto& pair : basic_blocks_) {
        const auto& block = pair.second;
        if (block.predecessors.empty() && !block.instructions.empty()) {
            // Check if first instruction is a function entry pattern
            const auto& first_instr = block.instructions[0];
            if (first_instr.opcode == "push" || first_instr.opcode == "mov" ||
                first_instr.opcode == "sub") {  // Common prologue patterns
                entries.push_back(block.start_address);
            }
        }
    }
    return entries;
}

std::vector<uint64_t> ControlFlowGraph::getExitPoints() const {
    std::vector<uint64_t> exits;
    for (const auto& pair : basic_blocks_) {
        const auto& block = pair.second;
        if (block.successors.empty() && !block.instructions.empty()) {
            // Check if last instruction is a return
            const auto& last_instr = block.instructions.back();
            if (last_instr.opcode == "ret") {
                exits.push_back(block.start_address);
            }
        }
    }
    return exits;
}

std::vector<std::string> ControlFlowGraph::analyzePatterns() const {
    std::vector<std::string> patterns;

    // Analyze for common control flow patterns
    for (const auto& pair : basic_blocks_) {
        const auto& block = pair.second;

        // Check for conditional blocks (blocks with 2 successors)
        if (block.successors.size() == 2) {
            patterns.push_back("Conditional block at 0x" +
                             std::to_string(block.start_address));
        }

        // Check for loop headers (blocks that jump back to themselves or earlier)
        for (uint64_t succ : block.successors) {
            if (succ <= block.start_address) {
                patterns.push_back("Potential loop at 0x" +
                                 std::to_string(block.start_address));
                break;
            }
        }

        // Check for function calls
        for (const auto& instr : block.instructions) {
            if (instr.opcode == "call") {
                patterns.push_back("Function call at 0x" +
                                 std::to_string(instr.address));
            }
        }
    }

    return patterns;
}

std::string ControlFlowGraph::exportDot() const {
    std::stringstream dot;
    dot << "digraph CFG {\n";
    dot << "  node [shape=rectangle];\n";

    // Add nodes
    for (const auto& pair : basic_blocks_) {
        const auto& block = pair.second;
        dot << "  \"0x" << std::hex << block.start_address << "\" [label=\"";

        // Add first few instructions to label
        for (size_t i = 0; i < std::min(size_t(3), block.instructions.size()); ++i) {
            if (i > 0) dot << "\\n";
            dot << block.instructions[i].mnemonic;
        }

        if (block.instructions.size() > 3) {
            dot << "\\n...";
        }

        dot << "\"];\n";
    }

    // Add edges
    for (const auto& edge : edges_) {
        dot << "  \"0x" << std::hex << edge.from_address << "\" -> \"0x"
            << std::hex << edge.to_address << "\"";

        if (edge.edge_type != "unconditional") {
            dot << " [label=\"" << edge.condition << "\"]";
        }

        dot << ";\n";
    }

    dot << "}\n";
    return dot.str();
}

std::vector<uint64_t> ControlFlowGraph::identifyBlockBoundaries(
    const std::vector<AsmInstruction>& instructions, const X86Analyzer& analyzer) {

    std::unordered_set<uint64_t> boundaries;
    boundaries.insert(instructions[0].address);  // Entry point

    for (const auto& instr : instructions) {
        X86InstructionInfo info = analyzer.analyzeInstruction(instr);

        // Instructions that end basic blocks
        if (info.is_jump || info.is_call || info.is_return) {
            boundaries.insert(instr.address);

            // Next instruction starts a new block
            auto it = std::find_if(instructions.begin(), instructions.end(),
                                 [addr = instr.address](const AsmInstruction& i) {
                                     return i.address > addr;
                                 });
            if (it != instructions.end()) {
                boundaries.insert(it->address);
            }
        }

        // Target of jumps/calls are also boundaries
        if (info.is_jump || info.is_call) {
            if (!info.operands.empty()) {
                uint64_t target = resolveJumpTarget(info.operands[0], instr.address);
                if (target != 0) {
                    boundaries.insert(target);
                }
            }
        }
    }

    // Sort boundaries
    std::vector<uint64_t> sorted_boundaries(boundaries.begin(), boundaries.end());
    std::sort(sorted_boundaries.begin(), sorted_boundaries.end());

    return sorted_boundaries;
}

void ControlFlowGraph::createBasicBlocks(const std::vector<AsmInstruction>& instructions,
                                       const std::vector<uint64_t>& boundaries) {
    for (size_t i = 0; i < boundaries.size(); ++i) {
        uint64_t start_addr = boundaries[i];
        uint64_t end_addr = (i + 1 < boundaries.size()) ? boundaries[i + 1] - 1 : UINT64_MAX;

        BasicBlock block(start_addr, end_addr);

        // Add instructions to this block
        for (const auto& instr : instructions) {
            if (instr.address >= start_addr && instr.address <= end_addr) {
                block.instructions.push_back(instr);
            }
        }

        if (!block.instructions.empty()) {
            basic_blocks_[start_addr] = std::move(block);
        }
    }
}

void ControlFlowGraph::buildControlFlowEdges(const X86Analyzer& analyzer) {
    for (auto& pair : basic_blocks_) {
        BasicBlock& block = pair.second;
        if (block.instructions.empty()) continue;

        const AsmInstruction& last_instr = block.instructions.back();
        X86InstructionInfo info = analyzer.analyzeInstruction(last_instr);

        // Determine successors based on instruction type
        if (info.is_return) {
            block.block_type = "return";
            // No successors for returns
        } else if (info.is_jump) {
            if (info.is_conditional) {
                block.block_type = "conditional";

                // Conditional jumps have two successors: target and fall-through
                if (!info.operands.empty()) {
                    uint64_t target = resolveJumpTarget(info.operands[0], last_instr.address);
                    if (target != 0) {
                        block.successors.push_back(target);
                        edges_.emplace_back(last_instr.address, target,
                                          "conditional_true", last_instr.opcode);
                    }
                }

                // Fall-through successor
                auto next_block_it = basic_blocks_.upper_bound(block.start_address);
                if (next_block_it != basic_blocks_.end()) {
                    uint64_t fallthrough = next_block_it->first;
                    block.successors.push_back(fallthrough);
                    edges_.emplace_back(last_instr.address, fallthrough,
                                      "conditional_false", "fallthrough");
                }
            } else {
                block.block_type = "unconditional";

                // Unconditional jump has one successor
                if (!info.operands.empty()) {
                    uint64_t target = resolveJumpTarget(info.operands[0], last_instr.address);
                    if (target != 0) {
                        block.successors.push_back(target);
                        edges_.emplace_back(last_instr.address, target, "unconditional");
                    }
                }
            }
        } else if (info.is_call) {
            block.block_type = "call";

            // Calls have fall-through successor
            auto next_block_it = basic_blocks_.upper_bound(block.start_address);
            if (next_block_it != basic_blocks_.end()) {
                uint64_t fallthrough = next_block_it->first;
                block.successors.push_back(fallthrough);
                edges_.emplace_back(last_instr.address, fallthrough, "call");
            }
        } else {
            block.block_type = "sequential";

            // Sequential execution - fall through to next block
            auto next_block_it = basic_blocks_.upper_bound(block.start_address);
            if (next_block_it != basic_blocks_.end()) {
                uint64_t fallthrough = next_block_it->first;
                block.successors.push_back(fallthrough);
                edges_.emplace_back(last_instr.address, fallthrough, "fallthrough");
            }
        }
    }

    // Build predecessor lists
    for (const auto& edge : edges_) {
        if (basic_blocks_.count(edge.to_address)) {
            basic_blocks_[edge.to_address].predecessors.push_back(edge.from_address);
        }
    }
}

uint64_t ControlFlowGraph::resolveJumpTarget(const std::string& operand,
                                           uint64_t current_address) const {
    // Handle different operand formats
    if (operand.empty()) return 0;

    // Direct address (hex)
    if (operand.find("0x") == 0) {
        try {
            return std::stoull(operand, nullptr, 16);
        } catch (...) {
            return 0;
        }
    }

    // Relative offset (decimal)
    try {
        int64_t offset = std::stoll(operand);
        return current_address + offset;
    } catch (...) {
        // Not a simple offset
    }

    // For now, return 0 for unresolved targets
    // In a full implementation, this would handle symbol resolution
    return 0;
}

} // namespace cppfort::stage2