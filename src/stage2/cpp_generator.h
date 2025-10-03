#pragma once

#include "cfg_recovery.h"
#include "variable_inference.h"
#include "x86_64_analyzer.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace cppfort::stage2 {

// C++ code generation context
struct CodeGenerationContext {
    std::unordered_map<std::string, std::string> variable_declarations;  // var_name -> type
    std::vector<std::string> generated_code;
    int indent_level;
    std::string current_function_name;

    CodeGenerationContext() : indent_level(0), current_function_name("unknown") {}

    void addLine(const std::string& line);
    void increaseIndent();
    void decreaseIndent();
    std::string getIndent() const;
};

// C++ statement generation
struct CppStatement {
    std::string statement;
    std::string comment;  // Optional comment explaining the assembly origin

    CppStatement(const std::string& stmt = "", const std::string& cmt = "")
        : statement(stmt), comment(cmt) {}
};

// C++ code generator
class CppGenerator {
public:
    CppGenerator();
    ~CppGenerator();

    // Main generation methods
    std::string generateFunction(const ControlFlowGraph& cfg,
                                const VariableInference& var_inference,
                                const std::string& function_name = "decompiled_function");

    // Generate code for a single basic block
    std::vector<CppStatement> generateBlockCode(const BasicBlock& block,
                                               const VariableInference& var_inference,
                                               CodeGenerationContext& context);

    // Generate C++ statement from x86 instruction
    CppStatement generateInstruction(const X86InstructionInfo& info,
                                   const VariableInference& var_inference,
                                   CodeGenerationContext& context);

    // Control flow reconstruction
    std::string generateControlFlow(const ControlFlowGraph& cfg,
                                  const VariableInference& var_inference);

    // Variable declarations
    std::vector<std::string> generateVariableDeclarations(const VariableInference& var_inference);

private:
    // Instruction-to-C++ mapping
    CppStatement generateArithmeticInstruction(const X86InstructionInfo& info,
                                             const VariableInference& var_inference);
    CppStatement generateLogicInstruction(const X86InstructionInfo& info,
                                        const VariableInference& var_inference);
    CppStatement generateMemoryInstruction(const X86InstructionInfo& info,
                                         const VariableInference& var_inference);
    CppStatement generateControlInstruction(const X86InstructionInfo& info,
                                          const VariableInference& var_inference);

    // Type inference helpers
    std::string inferVariableType(const std::string& var_name,
                                const VariableInference& var_inference) const;
    std::string mapRegisterToType(const std::string& reg_name) const;

    // Operand processing
    std::string processOperand(const std::string& operand,
                             const VariableInference& var_inference,
                             CodeGenerationContext& context) const;

    // Utility methods
    std::string sanitizeIdentifier(const std::string& name) const;
    std::string generateUniqueVariableName(const std::string& base_name,
                                         CodeGenerationContext& context) const;
    bool isTemporaryVariable(const std::string& var_name) const;

    // Pattern-based code generation
    std::string detectAndGenerateLoop(const ControlFlowGraph& cfg,
                                    const VariableInference& var_inference);
    std::string detectAndGenerateConditional(const ControlFlowGraph& cfg,
                                           const VariableInference& var_inference);

    // Code optimization
    void optimizeGeneratedCode(std::vector<CppStatement>& statements);
    void removeRedundantAssignments(std::vector<CppStatement>& statements);
};

} // namespace cppfort::stage2