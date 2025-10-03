#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace cppfort::stage2 {

// Represents a single assembly instruction
struct AsmInstruction {
    uint64_t address;           // Memory address
    std::string opcode;         // Instruction opcode (e.g., "mov", "add")
    std::string operands;       // Instruction operands as string
    std::vector<uint8_t> bytes; // Raw instruction bytes
    std::string mnemonic;       // Full instruction mnemonic

    AsmInstruction(uint64_t addr, const std::string& op, const std::string& ops,
                   const std::vector<uint8_t>& b, const std::string& mn)
        : address(addr), opcode(op), operands(ops), bytes(b), mnemonic(mn) {}
};

// Parses objdump -d output into structured instructions
class AsmParser {
public:
    AsmParser();

    // Parse objdump disassembly output
    // Returns vector of instructions, empty on failure
    std::vector<AsmInstruction> parse(const std::string& objdump_output);

    // Parse from file (runs objdump internally)
    std::vector<AsmInstruction> parseFile(const std::string& binary_path);

private:
    // Parse a single line of objdump output
    bool parseLine(const std::string& line, AsmInstruction& instruction);

    // Extract instruction bytes from hex string
    std::vector<uint8_t> parseBytes(const std::string& hex_string);
};

} // namespace cppfort::stage2