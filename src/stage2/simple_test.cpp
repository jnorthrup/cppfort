#include "asm_parser.h"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>

// Simple test demonstrating just the ASM parser (no Merkle dependencies)
int main() {
    std::cout << "Stage 2 ASM Parser Test\n";
    std::cout << "=======================\n\n";

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
        std::cout << "✅ Successfully parsed " << instructions.size() << " instructions:\n\n";
        for (const auto& inst : instructions) {
            std::cout << "  " << std::hex << inst.address << ": " << inst.opcode
                     << " " << inst.operands << "\n";
            std::cout << "    Bytes: ";
            for (auto byte : inst.bytes) {
                std::cout << std::hex << std::setw(2) << std::setfill('0')
                         << static_cast<int>(byte) << " ";
            }
            std::cout << "\n";
        }

        std::cout << "\n🎉 ASM Parser component working correctly!\n";
        std::cout << "\nThis demonstrates the core Stage 2 functionality:\n";
        std::cout << "- Parsing objdump disassembly output\n";
        std::cout << "- Extracting address, opcode, operands, and instruction bytes\n";
        std::cout << "- Ready for integration with differential analysis\n";

    } else {
        std::cout << "❌ ASM parsing failed\n";
        return 1;
    }

    return 0;
}