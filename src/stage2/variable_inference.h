#pragma once

#include "asm_parser.h"
#include "x86_64_analyzer.h"
#include "cfg_recovery.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>

namespace cppfort::stage2 {

// Inferred variable information
struct InferredVariable {
    std::string name;           // Generated variable name (var0, var1, etc.)
    std::string type_hint;      // Inferred type ("int", "ptr", "float", etc.)
    int size_hint;             // Size in bytes (4, 8, etc.)
    bool is_pointer;           // Whether this appears to be a pointer
    bool is_local;             // Whether this is a local variable
    bool is_parameter;         // Whether this is a function parameter
    std::unordered_set<uint64_t> used_addresses; // Where this variable is used

    InferredVariable(const std::string& n = "", const std::string& type = "unknown",
                    int size = 0, bool ptr = false, bool local = true, bool param = false)
        : name(n), type_hint(type), size_hint(size), is_pointer(ptr),
          is_local(local), is_parameter(param) {}
};

// Variable inference engine
class VariableInference {
public:
    VariableInference();
    ~VariableInference();

    // Analyze instructions to infer variables
    void analyzeInstructions(const std::vector<AsmInstruction>& instructions,
                           const X86Analyzer& analyzer,
                           const ControlFlowGraph& cfg);

    // Get all inferred variables
    std::vector<InferredVariable> getInferredVariables() const;

    // Get variable by name
    const InferredVariable* getVariable(const std::string& name) const;

    // Get variables used at a specific address
    std::vector<std::string> getVariablesAtAddress(uint64_t address) const;

    // Export variable analysis as JSON
    std::string exportJson() const;

private:
    // Track register usage patterns
    void analyzeRegisterUsage(const std::vector<AsmInstruction>& instructions,
                            const X86Analyzer& analyzer);

    // Analyze stack frame layout
    void analyzeStackFrame(const std::vector<AsmInstruction>& instructions,
                         const X86Analyzer& analyzer);

    // Analyze memory access patterns
    void analyzeMemoryAccess(const std::vector<AsmInstruction>& instructions,
                           const X86Analyzer& analyzer);

    // Infer types from usage patterns
    void inferTypesFromUsage();

    // Generate unique variable names
    std::string generateVariableName();

    // Check if an operand represents stack access
    bool isStackAccess(const std::string& operand) const;

    // Extract stack offset from operand
    int64_t extractStackOffset(const std::string& operand) const;

    // Inferred variables
    std::unordered_map<std::string, InferredVariable> variables_;

    // Address to variable mapping
    std::unordered_map<uint64_t, std::vector<std::string>> address_variables_;

    // Register to variable mapping (temporary registers)
    std::unordered_map<std::string, std::string> register_variables_;

    // Variable name counter
    int variable_counter_;
};