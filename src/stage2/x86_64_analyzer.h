#pragma once

#include "asm_parser.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace cppfort::stage2 {

// x86-64 register information
struct X86Register {
    std::string name;
    int size;  // in bytes
    bool is_general_purpose;
    bool is_stack_pointer;
    bool is_base_pointer;

    X86Register(const std::string& n, int s, bool gp = true,
                bool sp = false, bool bp = false)
        : name(n), size(s), is_general_purpose(gp),
          is_stack_pointer(sp), is_base_pointer(bp) {}
};

// x86-64 instruction semantic information
struct X86InstructionInfo {
    std::string mnemonic;
    std::string category;  // "arithmetic", "logic", "control", "memory", "stack"
    bool modifies_flags;
    bool is_conditional;
    bool is_call;
    bool is_return;
    bool is_jump;
    std::vector<std::string> operands;  // Parsed operands
    std::string semantic_description;

    X86InstructionInfo(const std::string& mn = "", const std::string& cat = "")
        : mnemonic(mn), category(cat), modifies_flags(false),
          is_conditional(false), is_call(false), is_return(false), is_jump(false) {}
};

// x86-64 instruction analyzer
class X86Analyzer {
public:
    X86Analyzer();
    ~X86Analyzer();

    // Analyze a single instruction
    X86InstructionInfo analyzeInstruction(const AsmInstruction& instruction);

    // Get register information
    const X86Register* getRegisterInfo(const std::string& reg_name) const;

    // Check if operand is a register
    bool isRegister(const std::string& operand) const;

    // Check if operand is a memory reference
    bool isMemoryReference(const std::string& operand) const;

    // Check if operand is an immediate value
    bool isImmediate(const std::string& operand) const;

    // Extract register name from operand (handles size prefixes like %rax, %eax, %ax, %al)
    std::string extractRegisterName(const std::string& operand) const;

    // Get canonical register name (full 64-bit name)
    std::string getCanonicalRegisterName(const std::string& reg) const;

private:
    // Initialize register database
    void initializeRegisters();

    // Initialize instruction patterns
    void initializeInstructionPatterns();

    // Parse instruction operands
    std::vector<std::string> parseOperands(const std::string& operands_str);

    // Register database
    std::unordered_map<std::string, X86Register> registers_;

    // Instruction pattern database
    std::unordered_map<std::string, X86InstructionInfo> instruction_patterns_;
};