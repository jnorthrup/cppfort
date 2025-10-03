#include "asm_parser.h"

#include <iostream>
#include <sstream>
#include <iomanip>
#include <regex>
#include <cstdio>
#include <memory>

namespace cppfort::stage2 {

AsmParser::AsmParser() = default;

std::vector<AsmInstruction> AsmParser::parse(const std::string& objdump_output) {
    std::vector<AsmInstruction> instructions;
    std::istringstream stream(objdump_output);
    std::string line;

    while (std::getline(stream, line)) {
        AsmInstruction instruction{0, "", "", {}, ""};
        if (parseLine(line, instruction)) {
            instructions.push_back(std::move(instruction));
        }
    }

    return instructions;
}

std::vector<AsmInstruction> AsmParser::parseFile(const std::string& binary_path) {
    // Run objdump to get disassembly
    std::string command = "objdump -d " + binary_path + " 2>/dev/null";
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"), pclose);

    if (!pipe) {
        std::cerr << "Failed to run objdump on " << binary_path << std::endl;
        return {};
    }

    std::ostringstream disassembly;
    char buffer[4096];
    while (fgets(buffer, sizeof(buffer), pipe.get()) != nullptr) {
        disassembly << buffer;
    }

    return parse(disassembly.str());
}

bool AsmParser::parseLine(const std::string& line, AsmInstruction& instruction) {
    // objdump -d output format (ARM64 example):
    // 100003f8c: 90000008        adrp    x8, 0x100003000
    // 100003f90: f9400108        ldr     x8, [x8, #0x0]

    // Use regex to parse the format: address: bytes\tmnemonic\toperands
    std::regex pattern(R"(\s*([0-9a-f]+):\s*([0-9a-f\s]+)\s+([a-zA-Z][a-zA-Z0-9]*)\s*(.*))");
    std::smatch matches;

    if (!std::regex_match(line, matches, pattern) || matches.size() < 5) {
        return false; // Not an instruction line
    }

    // Parse address
    std::istringstream(matches[1].str()) >> std::hex >> instruction.address;

    // Parse instruction bytes
    instruction.bytes = parseBytes(matches[2].str());

    // Parse opcode and operands
    instruction.opcode = matches[3].str();
    instruction.operands = matches[4].str();

    // Reconstruct full mnemonic
    instruction.mnemonic = instruction.opcode;
    if (!instruction.operands.empty()) {
        instruction.mnemonic += "\t" + instruction.operands;
    }

    return true;
}

std::vector<uint8_t> AsmParser::parseBytes(const std::string& hex_string) {
    std::vector<uint8_t> bytes;
    std::istringstream stream(hex_string);
    std::string byte_str;

    while (stream >> byte_str) {
        if (byte_str.size() == 2) {
            uint32_t byte_val;
            std::istringstream(byte_str) >> std::hex >> byte_val;
            bytes.push_back(static_cast<uint8_t>(byte_val));
        }
    }

    return bytes;
}

} // namespace cppfort::stage2