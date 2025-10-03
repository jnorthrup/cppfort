#include "architecture_detector.h"
#include "asm_parser.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <unordered_set>

namespace cppfort::stage2 {

ArchitectureDetector::ArchitectureDetector() {
    // Constructor - no initialization needed
}

ArchitectureInfo ArchitectureDetector::detectFromBinary(const std::string& binary_path) {
    try {
        return parseElfHeader(binary_path);
    } catch (const std::exception& e) {
        std::cerr << "ELF parsing failed: " << e.what() << ", falling back to pattern detection\n";
        // If ELF parsing fails, we'll need instructions to fall back
        // For now, return unknown - caller should use detectFromInstructions
        return ArchitectureInfo();
    }
}

ArchitectureInfo ArchitectureDetector::detectFromInstructions(const std::vector<AsmInstruction>& instructions) {
    if (instructions.empty()) {
        return ArchitectureInfo();
    }

    return detectFromOpcodes(instructions);
}

std::vector<CpuArchitecture> ArchitectureDetector::getSupportedArchitectures() {
    return {
        CpuArchitecture::X86_64,
        CpuArchitecture::X86_32,
        CpuArchitecture::ARM64,
        CpuArchitecture::ARM32
    };
}

std::string ArchitectureDetector::architectureToString(CpuArchitecture arch) {
    switch (arch) {
        case CpuArchitecture::X86_64: return "x86-64";
        case CpuArchitecture::X86_32: return "x86-32";
        case CpuArchitecture::ARM64: return "ARM64";
        case CpuArchitecture::ARM32: return "ARM32";
        default: return "unknown";
    }
}

ArchitectureInfo ArchitectureDetector::parseElfHeader(const std::string& binary_path) {
    std::ifstream file(binary_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open binary file: " + binary_path);
    }

    // Read ELF header (first 64 bytes for identification)
    std::vector<char> header(64);
    file.read(header.data(), 64);
    if (!file) {
        throw std::runtime_error("Cannot read ELF header");
    }

    // Check ELF magic number (0x7F 'E' 'L' 'F')
    if (header[0] != 0x7F || header[1] != 'E' || header[2] != 'L' || header[3] != 'F') {
        throw std::runtime_error("Not an ELF file");
    }

    // ELF class (32-bit or 64-bit)
    uint8_t elf_class = header[4];
    bool is_64bit = (elf_class == 2); // ELFCLASS64

    // Data encoding (endianness)
    uint8_t data_encoding = header[5];
    bool is_little_endian = (data_encoding == 1); // ELFDATA2LSB

    // Machine type (architecture)
    uint16_t machine;
    if (is_little_endian) {
        machine = (static_cast<uint8_t>(header[19]) << 8) | static_cast<uint8_t>(header[18]);
    } else {
        machine = (static_cast<uint8_t>(header[18]) << 8) | static_cast<uint8_t>(header[19]);
    }

    CpuArchitecture arch = CpuArchitecture::UNKNOWN;
    std::string name;
    std::string description;

    switch (machine) {
        case 0x3E: // EM_X86_64
            arch = CpuArchitecture::X86_64;
            name = "x86-64";
            description = "AMD64 / Intel 64-bit architecture";
            break;
        case 0x03: // EM_386
            arch = CpuArchitecture::X86_32;
            name = "x86-32";
            description = "IA-32 / x86 32-bit architecture";
            break;
        case 0xB7: // EM_AARCH64
            arch = CpuArchitecture::ARM64;
            name = "ARM64";
            description = "AArch64 / ARM 64-bit architecture";
            break;
        case 0x28: // EM_ARM
            arch = CpuArchitecture::ARM32;
            name = "ARM32";
            description = "ARM 32-bit architecture";
            break;
        default:
            arch = CpuArchitecture::UNKNOWN;
            name = "unknown";
            description = "Unsupported architecture (machine type: 0x" +
                         std::to_string(machine) + ")";
            break;
    }

    return ArchitectureInfo(arch, name, is_64bit, is_little_endian, description);
}

ArchitectureInfo ArchitectureDetector::detectFromOpcodes(const std::vector<AsmInstruction>& instructions) {
    // Analyze instruction patterns to detect architecture
    // This is a fallback method when ELF parsing isn't available

    std::unordered_set<std::string> opcodes;
    for (const auto& instr : instructions) {
        opcodes.insert(instr.opcode);
    }

    // Count architecture-specific patterns
    int x86_score = 0;
    int arm_score = 0;

    for (const auto& opcode : opcodes) {
        if (isLikelyX86Instruction(opcode)) {
            x86_score++;
        }
        if (isLikelyArmInstruction(opcode)) {
            arm_score++;
        }
    }

    // Determine architecture based on scores
    if (x86_score > arm_score * 2) {
        // Strong x86 signal
        bool likely_64bit = opcodes.count("movabs") > 0 || opcodes.count("syscall") > 0;
        if (likely_64bit) {
            return ArchitectureInfo(CpuArchitecture::X86_64, "x86-64",
                                   true, true, "Detected via x86-64 instruction patterns");
        } else {
            return ArchitectureInfo(CpuArchitecture::X86_32, "x86-32",
                                   false, true, "Detected via x86-32 instruction patterns");
        }
    } else if (arm_score > x86_score * 2) {
        // Strong ARM signal
        bool likely_64bit = opcodes.count("ldr") > 0 && opcodes.count("str") > 0;
        if (likely_64bit) {
            return ArchitectureInfo(CpuArchitecture::ARM64, "ARM64",
                                   true, true, "Detected via ARM64 instruction patterns");
        } else {
            return ArchitectureInfo(CpuArchitecture::ARM32, "ARM32",
                                   false, true, "Detected via ARM32 instruction patterns");
        }
    }

    // Ambiguous or unknown
    return ArchitectureInfo(CpuArchitecture::UNKNOWN, "unknown", false, true,
                           "Cannot determine architecture from instruction patterns");
}

bool ArchitectureDetector::isLikelyX86Instruction(const std::string& opcode) {
    // Common x86 instruction patterns
    static const std::unordered_set<std::string> x86_opcodes = {
        "mov", "add", "sub", "cmp", "jmp", "je", "jne", "jg", "jl", "jge", "jle",
        "call", "ret", "push", "pop", "lea", "xor", "and", "or", "not", "test",
        "movzx", "movsx", "cdqe", "cqo", "syscall", "int", "nop"
    };

    return x86_opcodes.count(opcode) > 0;
}

bool ArchitectureDetector::isLikelyArmInstruction(const std::string& opcode) {
    // Common ARM instruction patterns
    static const std::unordered_set<std::string> arm_opcodes = {
        "ldr", "str", "mov", "add", "sub", "cmp", "b", "beq", "bne", "bgt", "blt",
        "bge", "ble", "bl", "bx", "blx", "push", "pop", "ldm", "stm", "adr",
        "mvn", "and", "orr", "eor", "bic", "tst", "svc", "nop"
    };

    return arm_opcodes.count(opcode) > 0;
}

} // namespace cppfort::stage2