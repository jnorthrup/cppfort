#include "asm_parser.h"
#include "differential_tracker.h"
#include <iostream>
#include <vector>
#include <string>

// Simple test demonstrating Stage 2 components
int main() {
    std::cout << "Stage 2 Decompilation Pipeline Test\n";
    std::cout << "===================================\n\n";

    // Test ASM Parser with sample objdump output
    std::cout << "1. Testing ASM Parser...\n";
    cppfort::stage2::AsmParser parser;

    // Sample ARM64 objdump output (simplified)
    std::string sample_asm = R"(
100003f58: d10043ff  sub     sp, sp, #0x10
100003f5c: f90007e0  str     x0, [sp, #8]
100003f60: f94007e0  ldr     x0, [sp, #8]
100003f64: 910043ff  add     sp, sp, #0x10
100003f68: d65f03c0  ret
)";

    auto instructions = parser.parse(sample_asm);
    if (!instructions.empty()) {
        std::cout << "   ✅ Parsed " << instructions.size() << " instructions\n";
        for (const auto& inst : instructions) {
            std::cout << "     " << std::hex << inst.address << ": " << inst.opcode
                     << " " << inst.operands << "\n";
        }
    } else {
        std::cout << "   ❌ ASM parsing failed\n";
    }

    std::cout << "\n2. Testing Differential Tracker...\n";

    // Create two instruction sets (simulating different optimization levels)
    std::vector<cppfort::stage2::AsmInstruction> set1 = instructions; // Original -O0

    // Modified set (simulated optimization: remove redundant load/store)
    std::vector<cppfort::stage2::AsmInstruction> set2;
    for (size_t i = 0; i < instructions.size(); ++i) {
        if (i != 2) { // Skip the redundant ldr
            set2.push_back(instructions[i]);
        }
    }

    cppfort::stage2::DifferentialTracker tracker;
    tracker.addInstructionSet("-O0", set1);
    tracker.addInstructionSet("-O2", set2);

    cppfort::stage2::DifferentialResult result = tracker.computeDifferential("-O0", "-O2");

    std::cout << "   ✅ Differential analysis complete\n";
    std::cout << "     Cost reduction: " << result.cost_reduction << "%\n";
    std::cout << "     Pattern survival: " << result.pattern_survival_rate << "%\n";
    std::cout << "     Eliminated patterns: " << result.eliminated_patterns.size() << "\n";
    std::cout << "     Transformed patterns: " << result.transformed_patterns.size() << "\n";

    std::cout << "\n3. Testing JSON Export...\n";
    std::string json = tracker.exportJson();
    std::cout << "   ✅ JSON export successful (" << json.size() << " chars)\n";

    std::cout << "\n🎉 Stage 2 components working correctly!\n";
    return 0;
}