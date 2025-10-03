#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace cppfort::stage2 {

// Supported CPU architectures for decompilation
enum class CpuArchitecture {
    UNKNOWN,
    X86_64,     // AMD64 / Intel 64-bit
    X86_32,     // IA-32 / x86 32-bit
    ARM64,      // AArch64
    ARM32       // ARM 32-bit
};

// Architecture detection result
struct ArchitectureInfo {
    CpuArchitecture architecture;
    std::string architecture_name;  // Human-readable name
    bool is_64bit;
    bool is_little_endian;
    std::string description;

    ArchitectureInfo(CpuArchitecture arch = CpuArchitecture::UNKNOWN,
                     const std::string& name = "unknown",
                     bool bit64 = false,
                     bool little_endian = true,
                     const std::string& desc = "")
        : architecture(arch), architecture_name(name), is_64bit(bit64),
          is_little_endian(little_endian), description(desc) {}
};

// Detects CPU architecture from binary file or instruction patterns
class ArchitectureDetector {
public:
    ArchitectureDetector();

    // Detect architecture from ELF binary file
    ArchitectureInfo detectFromBinary(const std::string& binary_path);

    // Detect architecture from instruction patterns (fallback method)
    ArchitectureInfo detectFromInstructions(const std::vector<AsmInstruction>& instructions);

    // Get supported architectures
    static std::vector<CpuArchitecture> getSupportedArchitectures();

    // Convert architecture enum to string
    static std::string architectureToString(CpuArchitecture arch);

private:
    // ELF header parsing for architecture detection
    ArchitectureInfo parseElfHeader(const std::string& binary_path);

    // Pattern-based detection using instruction opcodes
    ArchitectureInfo detectFromOpcodes(const std::vector<AsmInstruction>& instructions);

    // x86 vs ARM instruction pattern analysis
    bool isLikelyX86Instruction(const std::string& opcode);
    bool isLikelyArmInstruction(const std::string& opcode);
};

} // namespace cppfort::stage2