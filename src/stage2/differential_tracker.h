#pragma once

#include "asm_parser.h"
#include <memory>
#include <unordered_map>
#include <string>
#include <vector>
#include <array>

namespace attestation {
class MerkleTree;
using Hash = std::array<uint8_t, 64>;
}

namespace cppfort::stage2 {

// Differential analysis result between two instruction sets
struct DifferentialResult {
    std::string baseline_optimization;  // e.g., "-O0", "-O1"
    std::string target_optimization;    // e.g., "-O1", "-O2"
    double cost_reduction;             // Percentage of instructions eliminated
    double pattern_survival_rate;      // Percentage of patterns that survived optimization
    attestation::Hash baseline_hash;
    attestation::Hash target_hash;
    std::vector<std::string> eliminated_patterns;
    std::vector<std::string> transformed_patterns;
};

// Tracks differential changes across optimization levels using Merkle trees
class DifferentialTracker {
public:
    DifferentialTracker();
    ~DifferentialTracker();

    // Add instruction set for a specific optimization level
    void addInstructionSet(const std::string& optimization_level,
                          const std::vector<AsmInstruction>& instructions);

    // Compute differential between two optimization levels
    DifferentialResult computeDifferential(const std::string& baseline_opt,
                                         const std::string& target_opt);

    // Get all tracked optimization levels
    std::vector<std::string> getOptimizationLevels() const;

    // Export differential data as JSON
    std::string exportJson() const;

private:
    // Hash an instruction set using Merkle tree
    attestation::Hash hashInstructionSet(const std::vector<AsmInstruction>& instructions);

    // Analyze pattern changes between instruction sets
    void analyzePatternChanges(const std::vector<AsmInstruction>& baseline,
                              const std::vector<AsmInstruction>& target,
                              DifferentialResult& result);

    // Count instruction patterns
    std::unordered_map<std::string, size_t> countPatterns(
        const std::vector<AsmInstruction>& instructions);

private:
    std::unordered_map<std::string, std::vector<AsmInstruction>> instruction_sets_;
    std::unordered_map<std::string, attestation::Hash> hashes_;
    std::unique_ptr<attestation::MerkleTree> merkle_tree_;
};

} // namespace cppfort::stage2