#include "cpp_generator.h"
#include "x86_64_analyzer.h"
#include <algorithm>
#include <sstream>
#include <iostream>
#include <regex>

namespace cppfort::stage2 {

void CodeGenerationContext::addLine(const std::string& line) {
    generated_code.push_back(std::string(indent_level * 4, ' ') + line);
}

void CodeGenerationContext::increaseIndent() {
    indent_level++;
}

void CodeGenerationContext::decreaseIndent() {
    if (indent_level > 0) indent_level--;
}

std::string CodeGenerationContext::getIndent() const {
    return std::string(indent_level * 4, ' ');
}

CppGenerator::CppGenerator() = default;
CppGenerator::~CppGenerator() = default;

std::string CppGenerator::generateFunction(const ControlFlowGraph& cfg,
                                         const VariableInference& var_inference,
                                         const std::string& function_name) {
    CodeGenerationContext context;
    context.current_function_name = function_name;

    // Generate function signature
    std::stringstream output;
    output << "// Decompiled function: " << function_name << "\n";
    output << "void " << sanitizeIdentifier(function_name) << "() {\n";

    // Generate variable declarations
    auto declarations = generateVariableDeclarations(var_inference);
    for (const auto& decl : declarations) {
        output << "    " << decl << "\n";
    }

    if (!declarations.empty()) {
        output << "\n";
    }

    // Generate function body
    auto statements = generateBlockCode(*cfg.getEntryBlock(), var_inference, context);

    // Add generated statements
    for (const auto& stmt : context.generated_code) {
        output << stmt << "\n";
    }

    // Add statements from basic block processing
    for (const auto& stmt : statements) {
        if (!stmt.comment.empty()) {
            output << "    " << stmt.statement << "  // " << stmt.comment << "\n";
        } else {
            output << "    " << stmt.statement << "\n";
        }
    }

    output << "}\n";
    return output.str();
}

std::vector<CppStatement> CppGenerator::generateBlockCode(const BasicBlock& block,
                                                        const VariableInference& var_inference,
                                                        CodeGenerationContext& context) {
    std::vector<CppStatement> statements;

    for (const auto& instruction : block.instructions) {
        X86Analyzer analyzer;
        auto info = analyzer.analyzeInstruction(instruction);
        auto stmt = generateInstruction(info, var_inference, context);
        if (!stmt.statement.empty()) {
            statements.push_back(stmt);
        }
    }

    return statements;
}

CppStatement CppGenerator::generateInstruction(const X86InstructionInfo& info,
                                             const VariableInference& var_inference,
                                             CodeGenerationContext& context) {
    if (info.category == "arithmetic") {
        return generateArithmeticInstruction(info, var_inference);
    } else if (info.category == "logic") {
        return generateLogicInstruction(info, var_inference);
    } else if (info.category == "memory") {
        return generateMemoryInstruction(info, var_inference);
    } else if (info.category == "control") {
        return generateControlInstruction(info, var_inference);
    } else if (info.category == "stack") {
        return generateStackInstruction(info, var_inference);
    }

    // Unknown instruction type
    return CppStatement("// Unknown instruction: " + info.mnemonic, info.semantic_description);
}

CppStatement CppGenerator::generateArithmeticInstruction(const X86InstructionInfo& info,
                                                       const VariableInference& var_inference) {
    if (info.operands.size() < 2) {
        return CppStatement("// Incomplete arithmetic instruction", info.semantic_description);
    }

    std::string dest = processOperand(info.operands[0], var_inference);
    std::string src = processOperand(info.operands[1], var_inference);

    std::string op;
    if (info.mnemonic == "add") op = "+=";
    else if (info.mnemonic == "sub") op = "-=";
    else if (info.mnemonic == "mul") op = "*=";
    else if (info.mnemonic == "div") op = "/=";
    else if (info.mnemonic == "imul") op = "*=";  // signed multiply
    else if (info.mnemonic == "idiv") op = "/=";  // signed divide
    else if (info.mnemonic == "inc") return CppStatement(dest + "++;", info.semantic_description);
    else if (info.mnemonic == "dec") return CppStatement(dest + "--;", info.semantic_description);
    else return CppStatement("// Unsupported arithmetic: " + info.mnemonic, info.semantic_description);

    return CppStatement(dest + " " + op + " " + src + ";", info.semantic_description);
}

CppStatement CppGenerator::generateLogicInstruction(const X86InstructionInfo& info,
                                                   const VariableInference& var_inference) {
    if (info.operands.size() < 2) {
        return CppStatement("// Incomplete logic instruction", info.semantic_description);
    }

    std::string dest = processOperand(info.operands[0], var_inference);
    std::string src = processOperand(info.operands[1], var_inference);

    std::string op;
    if (info.mnemonic == "and") op = "&=";
    else if (info.mnemonic == "or") op = "|=";
    else if (info.mnemonic == "xor") op = "^=";
    else if (info.mnemonic == "not") return CppStatement(dest + " = ~" + dest + ";", info.semantic_description);
    else if (info.mnemonic == "shl") op = "<<=";
    else if (info.mnemonic == "shr") op = ">>=";
    else if (info.mnemonic == "sar") op = ">>=";  // arithmetic shift right
    else return CppStatement("// Unsupported logic: " + info.mnemonic, info.semantic_description);

    return CppStatement(dest + " " + op + " " + src + ";", info.semantic_description);
}

CppStatement CppGenerator::generateMemoryInstruction(const X86InstructionInfo& info,
                                                    const VariableInference& var_inference) {
    if (info.operands.size() < 2) {
        return CppStatement("// Incomplete memory instruction", info.semantic_description);
    }

    if (info.mnemonic == "mov") {
        std::string dest = processOperand(info.operands[0], var_inference);
        std::string src = processOperand(info.operands[1], var_inference);
        return CppStatement(dest + " = " + src + ";", info.semantic_description);
    } else if (info.mnemonic == "lea") {
        std::string dest = processOperand(info.operands[0], var_inference);
        std::string src = processOperand(info.operands[1], var_inference);
        // LEA is typically address calculation, but for decompilation we treat it as assignment
        return CppStatement(dest + " = " + src + "; // LEA - address calculation", info.semantic_description);
    } else if (info.mnemonic == "xchg") {
        std::string op1 = processOperand(info.operands[0], var_inference);
        std::string op2 = processOperand(info.operands[1], var_inference);
        return CppStatement("std::swap(" + op1 + ", " + op2 + ");", info.semantic_description);
    }

    return CppStatement("// Unsupported memory instruction: " + info.mnemonic, info.semantic_description);
}

CppStatement CppGenerator::generateControlInstruction(const X86InstructionInfo& info,
                                                    const VariableInference& var_inference) {
    // Control flow instructions are handled at the block level
    // Individual jump/call instructions generate comments
    return CppStatement("// " + info.semantic_description, "");
}

CppStatement CppGenerator::generateStackInstruction(const X86InstructionInfo& info,
                                                   const VariableInference& var_inference) {
    if (info.operands.size() < 1) {
        return CppStatement("// Incomplete stack instruction", info.semantic_description);
    }

    std::string operand = processOperand(info.operands[0], var_inference);

    if (info.mnemonic == "push") {
        return CppStatement("// push " + operand + " onto stack", info.semantic_description);
    } else if (info.mnemonic == "pop") {
        return CppStatement("// pop " + operand + " from stack", info.semantic_description);
    }

    return CppStatement("// Unsupported stack instruction: " + info.mnemonic, info.semantic_description);
}

std::string CppGenerator::generateControlFlow(const ControlFlowGraph& cfg,
                                            const VariableInference& var_inference) {
    // Basic control flow reconstruction - this is a simplified version
    // A full implementation would detect loops, conditionals, etc.

    std::stringstream output;
    output << "// Control Flow Reconstruction\n";

    // For now, just generate basic block comments
    for (const auto& block : cfg.getBlocks()) {
        output << "// Block " << block->id << ":\n";
        CodeGenerationContext context;
        auto statements = generateBlockCode(*block, var_inference, context);

        for (const auto& stmt : statements) {
            output << "    " << stmt.statement;
            if (!stmt.comment.empty()) {
                output << "  // " << stmt.comment;
            }
            output << "\n";
        }
        output << "\n";
    }

    return output.str();
}

std::vector<std::string> CppGenerator::generateVariableDeclarations(const VariableInference& var_inference) {
    std::vector<std::string> declarations;

    // Get inferred variables
    auto variables = var_inference.getInferredVariables();

    for (const auto& var : variables) {
        std::string type = inferVariableType(var.name, var_inference);
        if (!type.empty()) {
            declarations.push_back(type + " " + sanitizeIdentifier(var.name) + ";");
        }
    }

    return declarations;
}

std::string CppGenerator::inferVariableType(const std::string& var_name,
                                          const VariableInference& var_inference) const {
    auto variables = var_inference.getInferredVariables();

    for (const auto& var : variables) {
        if (var.name == var_name) {
            // Map inferred type to C++ type
            if (var.type == "int64") return "int64_t";
            else if (var.type == "int32") return "int32_t";
            else if (var.type == "int16") return "int16_t";
            else if (var.type == "int8") return "int8_t";
            else if (var.type == "uint64") return "uint64_t";
            else if (var.type == "uint32") return "uint32_t";
            else if (var.type == "uint16") return "uint16_t";
            else if (var.type == "uint8") return "uint8_t";
            else if (var.type == "float64") return "double";
            else if (var.type == "float32") return "float";
            else if (var.type == "pointer") return "void*";
            else return "auto";  // Fallback
        }
    }

    return "auto";  // Default fallback
}

std::string CppGenerator::mapRegisterToType(const std::string& reg_name) const {
    // Map register names to likely types based on x86-64 conventions
    if (reg_name == "rax" || reg_name == "rbx" || reg_name == "rcx" || reg_name == "rdx") {
        return "int64_t";  // General purpose, often used for integers
    } else if (reg_name == "rsi" || reg_name == "rdi") {
        return "void*";    // Often used for pointers
    } else if (reg_name == "rbp" || reg_name == "rsp") {
        return "void*";    // Stack pointers
    } else if (reg_name.find('r') == 0 && reg_name.size() > 1) {
        return "int64_t";  // Other 64-bit registers
    }

    return "auto";
}

std::string CppGenerator::processOperand(const std::string& operand,
                                       const VariableInference& var_inference,
                                       CodeGenerationContext& context) const {
    X86Analyzer analyzer;

    if (analyzer.isRegister(operand)) {
        // Convert register to variable name
        std::string reg_name = analyzer.extractRegisterName(operand);
        std::string var_name = "var_" + reg_name;

        // Ensure variable is declared
        if (context.variable_declarations.find(var_name) == context.variable_declarations.end()) {
            std::string type = mapRegisterToType(reg_name);
            context.variable_declarations[var_name] = type;
        }

        return sanitizeIdentifier(var_name);
    } else if (analyzer.isMemoryReference(operand)) {
        // Handle memory references like (%rax), 8(%rbp), etc.
        // This is simplified - a full implementation would track memory layout
        return "/* memory reference: " + operand + " */";
    } else if (analyzer.isImmediate(operand)) {
        // Immediate values
        std::string imm = operand;
        if (!imm.empty() && imm[0] == '$') {
            imm = imm.substr(1);  // Remove AT&T immediate prefix
        }
        return imm;
    } else {
        // Unknown operand type
        return "/* unknown operand: " + operand + " */";
    }
}

std::string CppGenerator::sanitizeIdentifier(const std::string& name) const {
    std::string sanitized = name;

    // Replace invalid characters with underscores
    for (char& c : sanitized) {
        if (!std::isalnum(c) && c != '_') {
            c = '_';
        }
    }

    // Ensure it starts with a letter or underscore
    if (!sanitized.empty() && std::isdigit(sanitized[0])) {
        sanitized = "_" + sanitized;
    }

    // Handle C++ keywords
    static const std::unordered_set<std::string> keywords = {
        "auto", "break", "case", "char", "const", "continue", "default", "do",
        "double", "else", "enum", "extern", "float", "for", "goto", "if",
        "int", "long", "register", "return", "short", "signed", "sizeof",
        "static", "struct", "switch", "typedef", "union", "unsigned", "void",
        "volatile", "while"
    };

    if (keywords.count(sanitized)) {
        sanitized += "_var";
    }

    return sanitized;
}

std::string CppGenerator::generateUniqueVariableName(const std::string& base_name,
                                                   CodeGenerationContext& context) const {
    std::string name = base_name;
    int counter = 1;

    while (context.variable_declarations.find(name) != context.variable_declarations.end()) {
        name = base_name + "_" + std::to_string(counter++);
    }

    return name;
}

bool CppGenerator::isTemporaryVariable(const std::string& var_name) const {
    // Consider variables with "temp" or "tmp" in the name as temporary
    return var_name.find("temp") != std::string::npos ||
           var_name.find("tmp") != std::string::npos;
}

void CppGenerator::optimizeGeneratedCode(std::vector<CppStatement>& statements) {
    removeRedundantAssignments(statements);
    // Additional optimizations could be added here
}

void CppGenerator::removeRedundantAssignments(std::vector<CppStatement>& statements) {
    // Simple optimization: remove assignments where the same variable is assigned to itself
    statements.erase(
        std::remove_if(statements.begin(), statements.end(),
            [](const CppStatement& stmt) {
                // Look for patterns like "x = x;"
                std::string s = stmt.statement;
                // Remove spaces and semicolons for comparison
                s.erase(std::remove_if(s.begin(), s.end(), ::isspace), s.end());
                s.erase(std::remove(s.begin(), s.end(), ';'), s.end());

                // Check if it's "var = var"
                size_t equals_pos = s.find('=');
                if (equals_pos != std::string::npos) {
                    std::string left = s.substr(0, equals_pos);
                    std::string right = s.substr(equals_pos + 1);
                    return left == right;
                }
                return false;
            }),
        statements.end());
}

} // namespace cppfort::stage2