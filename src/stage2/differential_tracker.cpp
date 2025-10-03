#include "differential_tracker.h"

#include "../attestation/attestation.h"
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <regex>

namespace cppfort::stage2 {

DifferentialTracker::DifferentialTracker()
    : merkle_tree_(std::make_unique<attestation::MerkleTree>()) {
}

DifferentialTracker::~DifferentialTracker() = default;

void DifferentialTracker::addInstructionSet(const std::string& optimization_level,
                                          const std::vector<AsmInstruction>& instructions) {
    instruction_sets_[optimization_level] = instructions;
    hashes_[optimization_level] = hashInstructionSet(instructions);
}

DifferentialResult DifferentialTracker::computeDifferential(const std::string& baseline_opt,
                                                          const std::string& target_opt) {
    DifferentialResult result;
    result.baseline_optimization = baseline_opt;
    result.target_optimization = target_opt;

    auto baseline_it = instruction_sets_.find(baseline_opt);
    auto target_it = instruction_sets_.find(target_opt);

    if (baseline_it == instruction_sets_.end() || target_it == instruction_sets_.end()) {
        return result; // Return empty result if optimization levels not found
    }

    const auto& baseline_instructions = baseline_it->second;
    const auto& target_instructions = target_it->second;

    result.baseline_hash = hashes_[baseline_opt];
    result.target_hash = hashes_[target_opt];

    // Calculate cost reduction (instruction count reduction)
    if (!baseline_instructions.empty()) {
        double reduction = 1.0 - (static_cast<double>(target_instructions.size()) /
                                 baseline_instructions.size());
        result.cost_reduction = reduction * 100.0; // Convert to percentage
    }

    // Analyze pattern changes
    analyzePatternChanges(baseline_instructions, target_instructions, result);

    return result;
}

std::vector<std::string> DifferentialTracker::getOptimizationLevels() const {
    std::vector<std::string> levels;
    levels.reserve(instruction_sets_.size());
    for (const auto& pair : instruction_sets_) {
        levels.push_back(pair.first);
    }
    return levels;
}

std::string DifferentialTracker::exportJson() const {
    std::ostringstream json;
    json << "{\n";
    json << "  \"optimization_levels\": [\n";

    const auto& levels = getOptimizationLevels();
    for (size_t i = 0; i < levels.size(); ++i) {
        json << "    \"" << levels[i] << "\"";
        if (i < levels.size() - 1) json << ",";
        json << "\n";
    }
    json << "  ],\n";
    json << "  \"hashes\": {\n";

    for (size_t i = 0; i < levels.size(); ++i) {
        const auto& level = levels[i];
        auto hash_it = hashes_.find(level);
        if (hash_it != hashes_.end()) {
            json << "    \"" << level << "\": \"";
            for (uint8_t byte : hash_it->second) {
                json << std::hex << std::setw(2) << std::setfill('0')
                     << static_cast<int>(byte);
            }
            json << "\"";
            if (i < levels.size() - 1) json << ",";
            json << "\n";
        }
    }
    json << "  }\n";
    json << "}\n";

    return json.str();
}

attestation::Hash DifferentialTracker::hashInstructionSet(
    const std::vector<AsmInstruction>& instructions) {

    // Create a new Merkle tree for this instruction set
    attestation::MerkleTree tree;

    // Hash each instruction and add as leaf
    for (const auto& instruction : instructions) {
        // Create a hash from instruction components
        std::string instruction_str = std::to_string(instruction.address) + ":" +
                                    instruction.opcode + ":" + instruction.operands;

        attestation::Hash hash;
        // Simple hash using the string representation
        // In a real implementation, you'd use a proper cryptographic hash
        std::hash<std::string> hasher;
        size_t hash_val = hasher(instruction_str);

        // Convert size_t to 64-byte array (simplified)
        for (size_t i = 0; i < 8 && i < sizeof(size_t); ++i) {
            hash[i] = (hash_val >> (i * 8)) & 0xFF;
        }
        for (size_t i = 8; i < 64; ++i) {
            hash[i] = 0;
        }

        tree.addLeaf(hash);
    }

    tree.build();
    return tree.getRoot();
}

void DifferentialTracker::analyzePatternChanges(const std::vector<AsmInstruction>& baseline,
                                              const std::vector<AsmInstruction>& target,
                                              DifferentialResult& result) {
    auto baseline_patterns = countPatterns(baseline);
    auto target_patterns = countPatterns(target);

    size_t total_baseline_patterns = 0;
    size_t survived_patterns = 0;

    for (const auto& [pattern, count] : baseline_patterns) {
        total_baseline_patterns += count;
        auto target_it = target_patterns.find(pattern);
        if (target_it != target_patterns.end()) {
            survived_patterns += std::min(count, target_it->second);
        } else {
            result.eliminated_patterns.push_back(pattern);
        }
    }

    // Check for transformed patterns (patterns that changed)
    for (const auto& [pattern, count] : target_patterns) {
        if (baseline_patterns.find(pattern) == baseline_patterns.end()) {
            result.transformed_patterns.push_back(pattern);
        }
    }

    if (total_baseline_patterns > 0) {
        result.pattern_survival_rate = (static_cast<double>(survived_patterns) /
                                       total_baseline_patterns) * 100.0;
    }
}

std::unordered_map<std::string, size_t> DifferentialTracker::countPatterns(
    const std::vector<AsmInstruction>& instructions) {

    std::unordered_map<std::string, size_t> patterns;

    for (const auto& instruction : instructions) {
        // Use opcode as primary pattern
        patterns[instruction.opcode]++;

        // Also count opcode + operands patterns for more detail
        if (!instruction.operands.empty()) {
            std::string detailed_pattern = instruction.opcode + " " + instruction.operands;
            // Simplify operands (remove registers, immediates) for pattern matching
            std::string simplified = detailed_pattern;
            // Basic simplification - replace hex numbers and registers
            simplified = std::regex_replace(simplified, std::regex("0x[0-9a-f]+"), "IMM");
            simplified = std::regex_replace(simplified, std::regex("x[0-9]+"), "REG");
            simplified = std::regex_replace(simplified, std::regex("w[0-9]+"), "REG");

            patterns[simplified]++;
        }
    }

    return patterns;
}

} // namespace cppfort::stage2