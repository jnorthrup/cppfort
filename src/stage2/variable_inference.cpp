#include "variable_inference.h"
#include "x86_64_analyzer.h"
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <iostream>

namespace cppfort::stage2 {

VariableInference::VariableInference() = default;

VariableInference::~VariableInference() = default;

void VariableInference::analyzeInstructions(const std::vector<AsmInstruction>& instructions,
                                          const ControlFlowGraph& cfg,
                                          const std::string& architecture) {
    inferred_variables_.clear();
    register_usage_.clear();

    if (architecture == "x86_64") {
        analyzeX86_64Instructions(instructions, cfg);
    } else if (architecture == "aarch64") {
        analyzeARM64Instructions(instructions, cfg);
    }

    // Post-process to refine types and names
    refineVariableTypes();
    assignVariableNames();
}

void VariableInference::analyzeX86_64Instructions(const std::vector<AsmInstruction>& instructions,
                                                const ControlFlowGraph& cfg) {
    // Track register usage patterns
    for (const auto& instr : instructions) {
        analyzeX86_64Instruction(instr);
    }

    // Analyze control flow for variable scoping
    analyzeControlFlow(cfg);
}

void VariableInference::analyzeX86_64Instruction(const AsmInstruction& instr) {
    // Analyze instruction operands for variable inference
    auto operands = parseOperands(instr.operands);

    for (const auto& operand : operands) {
        if (operand.type == OperandType::Register) {
            trackRegisterUsage(operand.value, instr);
        } else if (operand.type == OperandType::Memory) {
            analyzeMemoryAccess(operand.value, instr);
        }
    }
}

void VariableInference::analyzeARM64Instructions(const std::vector<AsmInstruction>& instructions,
                                               const ControlFlowGraph& cfg) {
    // ARM64 support not yet implemented
    // For now, treat as unknown
}

void VariableInference::trackRegisterUsage(const std::string& reg_name, const AsmInstruction& instr) {
    // Normalize register name (remove size suffixes)
    std::string base_reg = normalizeRegisterName(reg_name);

    // Track usage patterns
    auto& usage = register_usage_[base_reg];
    usage.access_count++;

    // Analyze instruction type for usage patterns
    if (isLoadInstruction(instr.mnemonic)) {
        usage.load_count++;
    } else if (isStoreInstruction(instr.mnemonic)) {
        usage.store_count++;
    } else if (isArithmeticInstruction(instr.mnemonic)) {
        usage.arithmetic_count++;
    }

    // Track data flow
    usage.last_instruction = instr.mnemonic;
    usage.last_address = instr.address;
}

void VariableInference::analyzeMemoryAccess(const std::string& mem_expr, const AsmInstruction& instr) {
    // Parse memory expressions like [rbp-8], [rsp+16], etc.
    // This is a simplified implementation

    if (mem_expr.find("rbp") != std::string::npos ||
        mem_expr.find("rsp") != std::string::npos) {
        // Stack-based memory access - likely local variables
        inferStackVariable(mem_expr, instr);
    } else if (mem_expr.find("rip") != std::string::npos) {
        // RIP-relative - likely global or static data
        inferGlobalVariable(mem_expr, instr);
    }
}

void VariableInference::inferStackVariable(const std::string& mem_expr, const AsmInstruction& instr) {
    // Extract offset from expressions like [rbp-8], [rsp+16]
    int offset = 0;
    bool is_rbp = mem_expr.find("rbp") != std::string::npos;

    // Simple offset parsing (would be more sophisticated in real implementation)
    size_t plus_pos = mem_expr.find('+');
    size_t minus_pos = mem_expr.find('-');

    if (plus_pos != std::string::npos) {
        try {
            offset = std::stoi(mem_expr.substr(plus_pos + 1));
        } catch (...) {
            offset = 0;
        }
    } else if (minus_pos != std::string::npos) {
        try {
            offset = -std::stoi(mem_expr.substr(minus_pos + 1));
        } catch (...) {
            offset = 0;
        }
    }

    std::string var_key = (is_rbp ? "local_" : "param_") + std::to_string(std::abs(offset));

    auto& var = inferred_variables_[var_key];
    var.name = var_key;
    var.location = is_rbp ? VariableLocation::Stack : VariableLocation::Parameter;
    var.offset = offset;
    var.access_count++;
    var.last_access_addr = instr.address;

    // Infer type based on instruction
    if (isLoadInstruction(instr.mnemonic)) {
        var.usage = VariableUsage::Read;
    } else if (isStoreInstruction(instr.mnemonic)) {
        var.usage = VariableUsage::Write;
    }
}

void VariableInference::inferGlobalVariable(const std::string& mem_expr, const AsmInstruction& instr) {
    // RIP-relative addressing - global/static variables
    std::string var_key = "global_" + std::to_string(inferred_variables_.size());

    auto& var = inferred_variables_[var_key];
    var.name = var_key;
    var.location = VariableLocation::Global;
    var.access_count++;
    var.last_access_addr = instr.address;

    if (isLoadInstruction(instr.mnemonic)) {
        var.usage = VariableUsage::Read;
    } else if (isStoreInstruction(instr.mnemonic)) {
        var.usage = VariableUsage::Write;
    }
}

void VariableInference::analyzeControlFlow(const ControlFlowGraph& cfg) {
    // Analyze control flow to understand variable scoping and lifetimes
    // This is a simplified implementation

    for (const auto& block : cfg.blocks) {
        for (const auto& var_name : block.live_variables) {
            if (inferred_variables_.count(var_name)) {
                inferred_variables_[var_name].live_ranges.push_back(
                    std::make_pair(block.start_addr, block.end_addr));
            }
        }
    }
}

void VariableInference::refineVariableTypes() {
    // Refine variable types based on usage patterns and context
    for (auto& pair : inferred_variables_) {
        auto& var = pair.second;

        // Simple type inference based on usage
        if (var.location == VariableLocation::Stack) {
            // Local variables - infer from access patterns
            if (var.usage == VariableUsage::Read && var.access_count > 5) {
                var.inferred_type = "int";  // Common case
            } else if (var.usage == VariableUsage::Write) {
                var.inferred_type = "int";  // Default assumption
            }
        } else if (var.location == VariableLocation::Parameter) {
            var.inferred_type = "int";  // Function parameters
        } else if (var.location == VariableLocation::Global) {
            var.inferred_type = "int";  // Global variables
        }

        // Could be enhanced with more sophisticated type inference
        // based on instruction patterns, sizes, etc.
    }
}

void VariableInference::assignVariableNames() {
    // Assign meaningful variable names based on context
    int local_counter = 0;
    int param_counter = 0;
    int global_counter = 0;

    for (auto& pair : inferred_variables_) {
        auto& var = pair.second;

        if (var.location == VariableLocation::Stack) {
            var.name = "local_" + std::to_string(local_counter++);
        } else if (var.location == VariableLocation::Parameter) {
            var.name = "param_" + std::to_string(param_counter++);
        } else if (var.location == VariableLocation::Global) {
            var.name = "global_" + std::to_string(global_counter++);
        }
    }
}

std::vector<InferredVariable> VariableInference::getVariables() const {
    std::vector<InferredVariable> result;
    result.reserve(inferred_variables_.size());

    for (const auto& pair : inferred_variables_) {
        result.push_back(pair.second);
    }

    return result;
}

InferredVariable* VariableInference::getVariable(const std::string& name) {
    auto it = inferred_variables_.find(name);
    return it != inferred_variables_.end() ? &it->second : nullptr;
}

const InferredVariable* VariableInference::getVariable(const std::string& name) const {
    auto it = inferred_variables_.find(name);
    return it != inferred_variables_.end() ? &it->second : nullptr;
}

// Utility functions

std::string VariableInference::normalizeRegisterName(const std::string& reg) {
    // Normalize register names (remove size suffixes)
    if (reg.size() >= 3) {
        if (reg.substr(reg.size() - 2) == "64") {
            return reg.substr(0, reg.size() - 2);
        } else if (reg.substr(reg.size() - 2) == "32") {
            return reg.substr(0, reg.size() - 2);
        } else if (reg.substr(reg.size() - 2) == "16") {
            return reg.substr(0, reg.size() - 2);
        } else if (reg.substr(reg.size() - 1) == "8") {
            return reg.substr(0, reg.size() - 1);
        }
    }
    return reg;
}

std::vector<OperandInfo> VariableInference::parseOperands(const std::string& operands_str) {
    std::vector<OperandInfo> operands;

    // Simple operand parsing (would be more sophisticated in real implementation)
    std::stringstream ss(operands_str);
    std::string token;

    while (std::getline(ss, token, ',')) {
        // Trim whitespace
        token.erase(token.begin(), std::find_if(token.begin(), token.end(),
                   [](unsigned char ch) { return !std::isspace(ch); }));
        token.erase(std::find_if(token.rbegin(), token.rend(),
                   [](unsigned char ch) { return !std::isspace(ch); }).base(), token.end());

        if (token.empty()) continue;

        OperandInfo operand;

        if (token.find('[') != std::string::npos && token.find(']') != std::string::npos) {
            operand.type = OperandType::Memory;
        } else if (token.find('%') != std::string::npos || isRegisterName(token)) {
            operand.type = OperandType::Register;
        } else if (std::isdigit(token[0]) || token[0] == '-' || token[0] == '+') {
            operand.type = OperandType::Immediate;
        } else {
            operand.type = OperandType::Unknown;
        }

        operand.value = token;
        operands.push_back(operand);
    }

    return operands;
}

bool VariableInference::isRegisterName(const std::string& token) {
    // Simple check for common register names
    static const std::unordered_set<std::string> registers = {
        "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp",
        "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
        "eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "esp",
        "ax", "bx", "cx", "dx", "si", "di", "bp", "sp",
        "al", "bl", "cl", "dl"
    };

    return registers.count(token) > 0;
}

bool VariableInference::isLoadInstruction(const std::string& mnemonic) {
    static const std::unordered_set<std::string> load_instructions = {
        "mov", "movzx", "movsx", "lea", "pop"
    };
    return load_instructions.count(mnemonic) > 0;
}

bool VariableInference::isStoreInstruction(const std::string& mnemonic) {
    static const std::unordered_set<std::string> store_instructions = {
        "mov", "push"
    };
    return store_instructions.count(mnemonic) > 0;
}

bool VariableInference::isArithmeticInstruction(const std::string& mnemonic) {
    static const std::unordered_set<std::string> arith_instructions = {
        "add", "sub", "mul", "div", "inc", "dec", "neg"
    };
    return arith_instructions.count(mnemonic) > 0;
}

} // namespace cppfort::stage2