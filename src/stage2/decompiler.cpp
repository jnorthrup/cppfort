#include "decompiler.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>

namespace fs = std::filesystem;

namespace cppfort::stage2 {

Decompiler::Decompiler() {
    // Initialize components
    architecture_detector_ = std::make_unique<ArchitectureDetector>();
    x86_analyzer_ = std::make_unique<X86Analyzer>();
    variable_inference_ = std::make_unique<VariableInference>();
    cpp_generator_ = std::make_unique<CppGenerator>();
}

Decompiler::~Decompiler() = default;

DecompilationResult Decompiler::decompileBinary(const std::string& binary_path,
                                              const DecompilationOptions& options) {
    DecompilationResult result;
    result.architecture = "unknown";

    try {
        // Detect architecture from binary
        result.architecture = detectArchitectureFromBinary(binary_path);
        if (result.architecture == "unknown") {
            addError(result, "Failed to detect architecture from binary");
            return result;
        }

        // For now, we need assembly input. In a full implementation,
        // this would disassemble the binary first
        addError(result, "Binary decompilation requires disassembly step (not yet implemented)");
        return result;

    } catch (const std::exception& e) {
        addError(result, std::string("Binary decompilation failed: ") + e.what());
        return result;
    }
}

DecompilationResult Decompiler::decompileAssembly(const std::vector<std::string>& assembly_lines,
                                                const std::string& architecture_hint,
                                                const DecompilationOptions& options) {
    DecompilationResult result;

    try {
        // Parse assembly lines into instructions
        auto instructions = parseAssemblyLines(assembly_lines);
        if (instructions.empty()) {
            addError(result, "No valid instructions found in assembly");
            return result;
        }

        // Detect architecture
        result.architecture = architecture_hint.empty() ?
            detectArchitectureFromInstructions(instructions) : architecture_hint;

        if (result.architecture == "unknown") {
            addError(result, "Failed to detect architecture from instructions");
            return result;
        }

        // Route to architecture-specific decompilation
        if (result.architecture == "x86_64") {
            return decompileX86_64(instructions, options);
        } else if (result.architecture == "aarch64") {
            return decompileARM64(instructions, options);
        } else {
            addError(result, "Unsupported architecture: " + result.architecture);
            return result;
        }

    } catch (const std::exception& e) {
        addError(result, std::string("Assembly decompilation failed: ") + e.what());
        return result;
    }
}

DecompilationResult Decompiler::decompileX86_64(const std::vector<AsmInstruction>& instructions,
                                             const DecompilationOptions& options) {
    DecompilationResult result;
    result.architecture = "x86_64";

    try {
        // Build control flow graph
        ControlFlowGraph cfg = buildControlFlowGraph(instructions, "x86_64");

        // Infer variables
        VariableInference var_inference = inferVariables(instructions, cfg, "x86_64");

        // Generate C++ code
        CodeGenerationContext context;
        context.architecture = "x86_64";
        context.enable_optimization = options.enable_optimization;
        context.preserve_comments = options.preserve_comments;
        context.infer_types = options.infer_types;

        result.generated_code = cpp_generator_->generateCode(instructions, cfg, var_inference, context);

        // Validate and estimate accuracy
        validateDecompilation(result, instructions);
        result.accuracy_estimate = estimateAccuracy(instructions, result.generated_code);

        // Apply optimizations if enabled
        if (options.enable_optimization) {
            optimizeGeneratedCode(result, options);
        }

        return result;

    } catch (const std::exception& e) {
        addError(result, std::string("x86-64 decompilation failed: ") + e.what());
        return result;
    }
}

DecompilationResult Decompiler::decompileARM64(const std::vector<AsmInstruction>& instructions,
                                            const DecompilationOptions& options) {
    DecompilationResult result;
    result.architecture = "aarch64";

    // ARM64 support not yet implemented
    addError(result, "ARM64 decompilation not yet implemented");
    return result;
}

std::vector<AsmInstruction> Decompiler::parseAssemblyLines(const std::vector<std::string>& lines) {
    std::vector<AsmInstruction> instructions;

    for (const auto& line : lines) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#' || line[0] == ';') {
            continue;
        }

        // Basic instruction parsing (simplified)
        // In a full implementation, this would use the AsmParser
        AsmInstruction instr;
        instr.address = instructions.size() * 4;  // Dummy address
        instr.mnemonic = line.substr(0, line.find(' '));
        instr.operands = line.substr(line.find(' ') + 1);

        if (!instr.mnemonic.empty()) {
            instructions.push_back(instr);
        }
    }

    return instructions;
}

ControlFlowGraph Decompiler::buildControlFlowGraph(const std::vector<AsmInstruction>& instructions,
                                                 const std::string& architecture) {
    // Use the CFG recovery component
    ControlFlowGraph cfg;
    cfg.buildFromInstructions(instructions, architecture);
    return cfg;
}

VariableInference Decompiler::inferVariables(const std::vector<AsmInstruction>& instructions,
                                           const ControlFlowGraph& cfg,
                                           const std::string& architecture) {
    VariableInference inference;
    inference.analyzeInstructions(instructions, cfg, architecture);
    return inference;
}

void Decompiler::validateDecompilation(DecompilationResult& result,
                                     const std::vector<AsmInstruction>& original_instructions) {
    // Basic validation checks
    if (result.generated_code.empty()) {
        addError(result, "Generated code is empty");
        return;
    }

    // Check for basic syntax issues
    if (result.generated_code.find("undefined") != std::string::npos) {
        addWarning(result, "Generated code contains undefined references");
    }

    // Check instruction coverage
    size_t covered_instructions = 0;
    for (const auto& instr : original_instructions) {
        if (result.generated_code.find(instr.mnemonic) != std::string::npos) {
            covered_instructions++;
        }
    }

    double coverage = static_cast<double>(covered_instructions) / original_instructions.size();
    if (coverage < 0.5) {
        addWarning(result, "Low instruction coverage: " + std::to_string(coverage * 100) + "%");
    }
}

void Decompiler::optimizeGeneratedCode(DecompilationResult& result,
                                     const DecompilationOptions& options) {
    // Basic optimizations
    std::string& code = result.generated_code;

    // Remove redundant assignments
    // This is a simplified example - real optimization would be more sophisticated
    size_t pos = 0;
    while ((pos = code.find("int temp = ", pos)) != std::string::npos) {
        size_t end_pos = code.find(";", pos);
        if (end_pos != std::string::npos) {
            std::string assignment = code.substr(pos, end_pos - pos + 1);
            // Check if temp is used elsewhere (simplified check)
            if (code.find("temp", end_pos) == std::string::npos) {
                code.erase(pos, assignment.length());
                addWarning(result, "Removed unused temporary variable");
            } else {
                pos = end_pos + 1;
            }
        } else {
            break;
        }
    }
}

std::string Decompiler::detectArchitectureFromBinary(const std::string& binary_path) {
    if (!fs::exists(binary_path)) {
        return "unknown";
    }

    return architecture_detector_->detectFromBinary(binary_path);
}

std::string Decompiler::detectArchitectureFromInstructions(const std::vector<AsmInstruction>& instructions) {
    return architecture_detector_->detectFromInstructions(instructions);
}

double Decompiler::estimateAccuracy(const std::vector<AsmInstruction>& instructions,
                                  const std::string& generated_code) {
    // Simple accuracy estimation based on instruction coverage
    size_t covered = 0;
    for (const auto& instr : instructions) {
        if (generated_code.find(instr.mnemonic) != std::string::npos) {
            covered++;
        }
    }

    double coverage = static_cast<double>(covered) / instructions.size();

    // Adjust for code quality factors
    double quality_factor = 1.0;
    if (generated_code.find("undefined") != std::string::npos) {
        quality_factor *= 0.8;
    }
    if (generated_code.find("error") != std::string::npos) {
        quality_factor *= 0.9;
    }

    return coverage * quality_factor;
}

void Decompiler::addWarning(DecompilationResult& result, const std::string& warning) {
    result.warnings.push_back(warning);
}

void Decompiler::addError(DecompilationResult& result, const std::string& error) {
    result.errors.push_back(error);
}

} // namespace cppfort::stage2