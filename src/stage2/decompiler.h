#pragma once

#include "architecture_detector.h"
#include "asm_parser.h"
#include "cfg_recovery.h"
#include "variable_inference.h"
#include "cpp_generator.h"
#include "x86_64_analyzer.h"
#include <string>
#include <vector>
#include <memory>

namespace cppfort::stage2 {

// Decompilation result
struct DecompilationResult {
    std::string generated_code;
    std::string architecture;
    std::vector<std::string> warnings;
    std::vector<std::string> errors;
    double accuracy_estimate;  // 0.0 to 1.0

    bool success() const { return errors.empty() && !generated_code.empty(); }
};

// Decompilation options
struct DecompilationOptions {
    bool enable_optimization = true;
    bool preserve_comments = true;
    bool infer_types = true;
    double accuracy_target = 0.6;  // Target 60% accuracy
    std::string output_format = "cpp";  // "cpp" or "c"
};

// Main decompiler class
class Decompiler {
public:
    Decompiler();
    ~Decompiler();

    // Main decompilation methods
    DecompilationResult decompileBinary(const std::string& binary_path,
                                      const DecompilationOptions& options = DecompilationOptions());

    DecompilationResult decompileAssembly(const std::vector<std::string>& assembly_lines,
                                        const std::string& architecture_hint = "",
                                        const DecompilationOptions& options = DecompilationOptions());

    // Component access for testing/advanced usage
    std::unique_ptr<ArchitectureDetector> architecture_detector_;
    std::unique_ptr<X86Analyzer> x86_analyzer_;
    std::unique_ptr<VariableInference> variable_inference_;
    std::unique_ptr<CppGenerator> cpp_generator_;

private:
    // Architecture-specific decompilation
    DecompilationResult decompileX86_64(const std::vector<AsmInstruction>& instructions,
                                       const DecompilationOptions& options);

    DecompilationResult decompileARM64(const std::vector<AsmInstruction>& instructions,
                                      const DecompilationOptions& options);

    // Common processing steps
    std::vector<AsmInstruction> parseAssemblyLines(const std::vector<std::string>& lines);
    ControlFlowGraph buildControlFlowGraph(const std::vector<AsmInstruction>& instructions,
                                         const std::string& architecture);
    VariableInference inferVariables(const std::vector<AsmInstruction>& instructions,
                                   const ControlFlowGraph& cfg,
                                   const std::string& architecture);

    // Validation and optimization
    void validateDecompilation(DecompilationResult& result,
                             const std::vector<AsmInstruction>& original_instructions);
    void optimizeGeneratedCode(DecompilationResult& result,
                             const DecompilationOptions& options);

    // Utility methods
    std::string detectArchitectureFromBinary(const std::string& binary_path);
    std::string detectArchitectureFromInstructions(const std::vector<AsmInstruction>& instructions);
    double estimateAccuracy(const std::vector<AsmInstruction>& instructions,
                          const std::string& generated_code);

    // Error handling
    void addWarning(DecompilationResult& result, const std::string& warning);
    void addError(DecompilationResult& result, const std::string& error);
};

} // namespace cppfort::stage2