#pragma once

#include "asm_parser.h"
#include "x86_64_analyzer.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>

namespace cppfort::stage2 {

// Basic block in the control flow graph
struct BasicBlock {
    uint64_t start_address;
    uint64_t end_address;
    std::vector<AsmInstruction> instructions;
    std::vector<uint64_t> successors;  // Addresses of successor blocks
    std::vector<uint64_t> predecessors; // Addresses of predecessor blocks
    std::string block_type;  // "entry", "exit", "conditional", "unconditional", "call", "return"

    BasicBlock(uint64_t start = 0, uint64_t end = 0)
        : start_address(start), end_address(end) {}
};

// Control flow edge
struct ControlFlowEdge {
    uint64_t from_address;
    uint64_t to_address;
    std::string edge_type;  // "unconditional", "conditional_true", "conditional_false", "call", "return"
    std::string condition;  // For conditional edges (e.g., "je", "jne")

    ControlFlowEdge(uint64_t from = 0, uint64_t to = 0,
                   const std::string& type = "unconditional",
                   const std::string& cond = "")
        : from_address(from), to_address(to), edge_type(type), condition(cond) {}
};

// Control Flow Graph
class ControlFlowGraph {
public:
    ControlFlowGraph();
    ~ControlFlowGraph();

    // Build CFG from instruction list
    void buildFromInstructions(const std::vector<AsmInstruction>& instructions,
                              const X86Analyzer& analyzer);

    // Get basic block by start address
    const BasicBlock* getBasicBlock(uint64_t address) const;

    // Get all basic blocks
    std::vector<const BasicBlock*> getAllBasicBlocks() const;

    // Get entry points (functions)
    std::vector<uint64_t> getEntryPoints() const;

    // Get exit points
    std::vector<uint64_t> getExitPoints() const;

    // Analyze control flow patterns
    std::vector<std::string> analyzePatterns() const;

    // Export CFG as GraphViz DOT format
    std::string exportDot() const;

private:
    // Identify basic block boundaries
    std::vector<uint64_t> identifyBlockBoundaries(const std::vector<AsmInstruction>& instructions,
                                                 const X86Analyzer& analyzer);

    // Create basic blocks from boundaries
    void createBasicBlocks(const std::vector<AsmInstruction>& instructions,
                          const std::vector<uint64_t>& boundaries);

    // Build control flow edges
    void buildControlFlowEdges(const X86Analyzer& analyzer);

    // Analyze instruction for control flow impact
    void analyzeInstructionForControlFlow(const AsmInstruction& instruction,
                                        const X86InstructionInfo& info,
                                        uint64_t current_block_start);

    // Resolve jump targets
    uint64_t resolveJumpTarget(const std::string& operand, uint64_t current_address) const;

    // Basic blocks indexed by start address
    std::unordered_map<uint64_t, BasicBlock> basic_blocks_;

    // Control flow edges
    std::vector<ControlFlowEdge> edges_;

    // All instructions (for reference)
    std::vector<AsmInstruction> all_instructions_;
};