#include "x86_64_analyzer.h"
#include <algorithm>
#include <sstream>
#include <regex>

namespace cppfort::stage2 {

X86Analyzer::X86Analyzer() {
    initializeRegisters();
    initializeInstructionPatterns();
}

X86Analyzer::~X86Analyzer() = default;

X86InstructionInfo X86Analyzer::analyzeInstruction(const AsmInstruction& instruction) {
    X86InstructionInfo info = instruction_patterns_[instruction.opcode];

    // Set the mnemonic
    info.mnemonic = instruction.mnemonic;

    // Parse operands
    info.operands = parseOperands(instruction.operands);

    // Generate semantic description
    std::stringstream desc;
    desc << info.category << " operation: " << instruction.opcode;
    if (!info.operands.empty()) {
        desc << " with operands: ";
        for (size_t i = 0; i < info.operands.size(); ++i) {
            if (i > 0) desc << ", ";
            desc << info.operands[i];
        }
    }
    info.semantic_description = desc.str();

    return info;
}

const X86Register* X86Analyzer::getRegisterInfo(const std::string& reg_name) const {
    auto it = registers_.find(extractRegisterName(reg_name));
    return it != registers_.end() ? &it->second : nullptr;
}

bool X86Analyzer::isRegister(const std::string& operand) const {
    if (operand.empty()) return false;
    std::string clean_operand = operand;

    // Remove size prefixes and % prefix
    if (clean_operand[0] == '%') {
        clean_operand = clean_operand.substr(1);
    }

    return registers_.count(extractRegisterName(clean_operand)) > 0;
}

bool X86Analyzer::isMemoryReference(const std::string& operand) const {
    if (operand.empty()) return false;

    // Check for common memory reference patterns
    return operand.find('(') != std::string::npos ||
           operand.find("0x") != std::string::npos ||
           operand.find("rip+") != std::string::npos ||
           operand.find("rip-") != std::string::npos;
}

bool X86Analyzer::isImmediate(const std::string& operand) const {
    if (operand.empty()) return false;

    // Check for immediate values (hex, decimal, or $ prefix)
    return operand[0] == '$' ||
           operand.find("0x") == 0 ||
           (operand[0] >= '0' && operand[0] <= '9');
}

std::string X86Analyzer::extractRegisterName(const std::string& operand) const {
    if (operand.empty()) return "";

    std::string reg = operand;

    // Remove % prefix if present
    if (reg[0] == '%') {
        reg = reg.substr(1);
    }

    // Handle size suffixes for x86-64 registers
    // rax/eax/ax/al -> rax
    // rbx/ebx/bx/bl -> rbx
    // etc.
    if (reg == "rax" || reg == "eax" || reg == "ax" || reg == "al") return "rax";
    if (reg == "rbx" || reg == "ebx" || reg == "bx" || reg == "bl") return "rbx";
    if (reg == "rcx" || reg == "ecx" || reg == "cx" || reg == "cl") return "rcx";
    if (reg == "rdx" || reg == "edx" || reg == "dx" || reg == "dl") return "rdx";
    if (reg == "rsi" || reg == "esi" || reg == "si" || reg == "sil") return "rsi";
    if (reg == "rdi" || reg == "edi" || reg == "di" || reg == "dil") return "rdi";
    if (reg == "rbp" || reg == "ebp" || reg == "bp" || reg == "bpl") return "rbp";
    if (reg == "rsp" || reg == "esp" || reg == "sp" || reg == "spl") return "rsp";
    if (reg == "r8" || reg == "r8d" || reg == "r8w" || reg == "r8b") return "r8";
    if (reg == "r9" || reg == "r9d" || reg == "r9w" || reg == "r9b") return "r9";
    if (reg == "r10" || reg == "r10d" || reg == "r10w" || reg == "r10b") return "r10";
    if (reg == "r11" || reg == "r11d" || reg == "r11w" || reg == "r11b") return "r11";
    if (reg == "r12" || reg == "r12d" || reg == "r12w" || reg == "r12b") return "r12";
    if (reg == "r13" || reg == "r13d" || reg == "r13w" || reg == "r13b") return "r13";
    if (reg == "r14" || reg == "r14d" || reg == "r14w" || reg == "r14b") return "r14";
    if (reg == "r15" || reg == "r15d" || reg == "r15w" || reg == "r15b") return "r15";

    // Special registers
    if (reg == "rip") return "rip";
    if (reg == "eflags" || reg == "rflags") return "rflags";

    return reg;
}

std::string X86Analyzer::getCanonicalRegisterName(const std::string& reg) const {
    return extractRegisterName(reg);
}

void X86Analyzer::initializeRegisters() {
    // General purpose registers
    registers_["rax"] = X86Register("rax", 8, true, false, false);
    registers_["rbx"] = X86Register("rbx", 8, true, false, false);
    registers_["rcx"] = X86Register("rcx", 8, true, false, false);
    registers_["rdx"] = X86Register("rdx", 8, true, false, false);
    registers_["rsi"] = X86Register("rsi", 8, true, false, false);
    registers_["rdi"] = X86Register("rdi", 8, true, false, false);
    registers_["rbp"] = X86Register("rbp", 8, true, false, true);
    registers_["rsp"] = X86Register("rsp", 8, true, true, false);
    registers_["r8"] = X86Register("r8", 8, true, false, false);
    registers_["r9"] = X86Register("r9", 8, true, false, false);
    registers_["r10"] = X86Register("r10", 8, true, false, false);
    registers_["r11"] = X86Register("r11", 8, true, false, false);
    registers_["r12"] = X86Register("r12", 8, true, false, false);
    registers_["r13"] = X86Register("r13", 8, true, false, false);
    registers_["r14"] = X86Register("r14", 8, true, false, false);
    registers_["r15"] = X86Register("r15", 8, true, false, false);

    // Special registers
    registers_["rip"] = X86Register("rip", 8, false, false, false);
    registers_["rflags"] = X86Register("rflags", 8, false, false, false);
}

void X86Analyzer::initializeInstructionPatterns() {
    // Arithmetic instructions
    instruction_patterns_["add"] = X86InstructionInfo("add", "arithmetic");
    instruction_patterns_["add"].modifies_flags = true;

    instruction_patterns_["sub"] = X86InstructionInfo("sub", "arithmetic");
    instruction_patterns_["sub"].modifies_flags = true;

    instruction_patterns_["mul"] = X86InstructionInfo("mul", "arithmetic");
    instruction_patterns_["mul"].modifies_flags = true;

    instruction_patterns_["div"] = X86InstructionInfo("div", "arithmetic");
    instruction_patterns_["div"].modifies_flags = true;

    instruction_patterns_["inc"] = X86InstructionInfo("inc", "arithmetic");
    instruction_patterns_["inc"].modifies_flags = true;

    instruction_patterns_["dec"] = X86InstructionInfo("dec", "arithmetic");
    instruction_patterns_["dec"].modifies_flags = true;

    // Logic instructions
    instruction_patterns_["and"] = X86InstructionInfo("and", "logic");
    instruction_patterns_["and"].modifies_flags = true;

    instruction_patterns_["or"] = X86InstructionInfo("or", "logic");
    instruction_patterns_["or"].modifies_flags = true;

    instruction_patterns_["xor"] = X86InstructionInfo("xor", "logic");
    instruction_patterns_["xor"].modifies_flags = true;

    instruction_patterns_["not"] = X86InstructionInfo("not", "logic");

    instruction_patterns_["test"] = X86InstructionInfo("test", "logic");
    instruction_patterns_["test"].modifies_flags = true;

    // Control flow instructions
    instruction_patterns_["jmp"] = X86InstructionInfo("jmp", "control");
    instruction_patterns_["jmp"].is_jump = true;

    instruction_patterns_["je"] = X86InstructionInfo("je", "control");
    instruction_patterns_["je"].is_jump = true;
    instruction_patterns_["je"].is_conditional = true;

    instruction_patterns_["jne"] = X86InstructionInfo("jne", "control");
    instruction_patterns_["jne"].is_jump = true;
    instruction_patterns_["jne"].is_conditional = true;

    instruction_patterns_["jg"] = X86InstructionInfo("jg", "control");
    instruction_patterns_["jg"].is_jump = true;
    instruction_patterns_["jg"].is_conditional = true;

    instruction_patterns_["jl"] = X86InstructionInfo("jl", "control");
    instruction_patterns_["jl"].is_jump = true;
    instruction_patterns_["jl"].is_conditional = true;

    instruction_patterns_["jge"] = X86InstructionInfo("jge", "control");
    instruction_patterns_["jge"].is_jump = true;
    instruction_patterns_["jge"].is_conditional = true;

    instruction_patterns_["jle"] = X86InstructionInfo("jle", "control");
    instruction_patterns_["jle"].is_jump = true;
    instruction_patterns_["jle"].is_conditional = true;

    instruction_patterns_["call"] = X86InstructionInfo("call", "control");
    instruction_patterns_["call"].is_call = true;

    instruction_patterns_["ret"] = X86InstructionInfo("ret", "control");
    instruction_patterns_["ret"].is_return = true;

    // Memory instructions
    instruction_patterns_["mov"] = X86InstructionInfo("mov", "memory");
    instruction_patterns_["lea"] = X86InstructionInfo("lea", "memory");
    instruction_patterns_["push"] = X86InstructionInfo("push", "stack");
    instruction_patterns_["pop"] = X86InstructionInfo("pop", "stack");

    // Comparison
    instruction_patterns_["cmp"] = X86InstructionInfo("cmp", "logic");
    instruction_patterns_["cmp"].modifies_flags = true;

    // Special instructions
    instruction_patterns_["nop"] = X86InstructionInfo("nop", "special");
    instruction_patterns_["syscall"] = X86InstructionInfo("syscall", "system");
    instruction_patterns_["int"] = X86InstructionInfo("int", "system");
}

std::vector<std::string> X86Analyzer::parseOperands(const std::string& operands_str) {
    std::vector<std::string> operands;
    if (operands_str.empty()) {
        return operands;
    }

    std::stringstream ss(operands_str);
    std::string operand;
    while (std::getline(ss, operand, ',')) {
        // Trim whitespace
        operand.erase(operand.begin(),
                     std::find_if(operand.begin(), operand.end(),
                                 [](unsigned char ch) { return !std::isspace(ch); }));
        operand.erase(std::find_if(operand.rbegin(), operand.rend(),
                                  [](unsigned char ch) { return !std::isspace(ch); }).base(),
                     operand.end());

        if (!operand.empty()) {
            operands.push_back(operand);
        }
    }

    return operands;
}

} // namespace cppfort::stage2