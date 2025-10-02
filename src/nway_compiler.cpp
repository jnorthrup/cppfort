// N-way compiler main driver
// Supports C/C++/CPP2 bidirectional transpilation with attestation

#include "ir/ir.h"
#include "parsers/c_parser.h"
#include "attestation/attestation.h"
#include "../src/stage0/parser.h"  // Existing CPP2 parser

#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <filesystem>
#include <chrono>

namespace fs = std::filesystem;

enum class Language {
    C,
    CPP,
    CPP2,
    Unknown
};

enum class CompileMode {
    Transpile,      // Source -> Source
    Compile,        // Source -> Binary
    IRDump,         // Source -> IR
    Verify,         // Verify attestation
    RoundTrip       // Round-trip test
};

struct CompilerOptions {
    std::string input_file;
    std::string output_file;
    Language input_lang = Language::Unknown;
    Language output_lang = Language::Unknown;
    CompileMode mode = CompileMode::Transpile;

    // Attestation options
    bool enable_attestation = true;
    bool deterministic = true;
    bool sign_output = true;
    std::string key_file;

    // Optimization
    int opt_level = 2;
    bool debug_info = false;

    // Paths
    std::vector<std::string> include_paths;
    std::vector<std::string> lib_paths;

    // Misc
    bool verbose = false;
    bool time_report = false;
};

class NWayCompiler {
public:
    NWayCompiler(const CompilerOptions& opts) : options(opts) {
        if (options.enable_attestation) {
            attestation::AttestationSystem::getInstance().initialize();
        }
    }

    int compile() {
        auto start_time = std::chrono::high_resolution_clock::now();

        // Detect languages
        if (options.input_lang == Language::Unknown) {
            options.input_lang = detectLanguage(options.input_file);
        }
        if (options.output_lang == Language::Unknown && options.mode == CompileMode::Transpile) {
            options.output_lang = detectLanguage(options.output_file);
        }

        // Start attestation
        if (options.enable_attestation) {
            attestation::AttestationSystem::getInstance().beginCompilation(options.input_file);
        }

        // Parse to IR
        auto ir = parseToIR();
        if (!ir) {
            std::cerr << "Failed to parse input file\n";
            return 1;
        }

        // Attest IR
        if (options.enable_attestation) {
            attestation::AttestationSystem::getInstance().attestIR(ir.get(), sizeof(*ir));
        }

        // Process based on mode
        int result = 0;
        switch (options.mode) {
            case CompileMode::Transpile:
                result = transpile(ir);
                break;
            case CompileMode::Compile:
                result = compileToBinary(ir);
                break;
            case CompileMode::IRDump:
                result = dumpIR(ir);
                break;
            case CompileMode::Verify:
                result = verifyAttestation();
                break;
            case CompileMode::RoundTrip:
                result = roundTripTest(ir);
                break;
        }

        // Finalize attestation
        if (options.enable_attestation && result == 0) {
            attestation::AttestationSystem::getInstance().attestOutput(options.output_file);
            auto record = attestation::AttestationSystem::getInstance().finalizeCompilation();

            if (options.verbose) {
                printAttestationRecord(record);
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        if (options.time_report) {
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            std::cout << "Compilation time: " << duration.count() << "ms\n";
        }

        return result;
    }

private:
    CompilerOptions options;

    Language detectLanguage(const std::string& filename) {
        fs::path path(filename);
        std::string ext = path.extension().string();

        if (ext == ".c" || ext == ".h") return Language::C;
        if (ext == ".cpp" || ext == ".cxx" || ext == ".cc" || ext == ".hpp") return Language::CPP;
        if (ext == ".cpp2") return Language::CPP2;

        return Language::Unknown;
    }

    std::shared_ptr<ir::Module> parseToIR() {
        std::string source = readFile(options.input_file);

        switch (options.input_lang) {
            case Language::C: {
                parsers::CParser parser(source);
                return parser.parse();
            }
            case Language::CPP2: {
                // Use existing CPP2 parser and convert to IR
                cppfront::Parser parser(source);
                auto ast = parser.parse();
                return convertCPP2ToIR(ast);
            }
            case Language::CPP:
                // TODO: Implement C++ parser
                std::cerr << "C++ parser not yet implemented\n";
                return nullptr;
            default:
                std::cerr << "Unknown input language\n";
                return nullptr;
        }
    }

    int transpile(std::shared_ptr<ir::Module> ir_module) {
        std::string output;

        switch (options.output_lang) {
            case Language::C:
                output = emitC(ir_module);
                break;
            case Language::CPP:
                output = emitCPP(ir_module);
                break;
            case Language::CPP2:
                output = emitCPP2(ir_module);
                break;
            default:
                std::cerr << "Unknown output language\n";
                return 1;
        }

        return writeFile(options.output_file, output) ? 0 : 1;
    }

    int compileToBinary(std::shared_ptr<ir::Module> ir_module) {
        // First transpile to C++
        std::string cpp_file = options.output_file + ".cpp";
        std::string cpp_code = emitCPP(ir_module);

        if (!writeFile(cpp_file, cpp_code)) {
            return 1;
        }

        // Compile with g++
        std::stringstream cmd;
        cmd << "g++ -std=c++20";

        if (options.opt_level > 0) {
            cmd << " -O" << options.opt_level;
        }
        if (options.debug_info) {
            cmd << " -g";
        }

        for (const auto& path : options.include_paths) {
            cmd << " -I" << path;
        }

        cmd << " " << cpp_file;
        cmd << " -o " << options.output_file;

        int result = std::system(cmd.str().c_str());

        // Clean up temp file
        fs::remove(cpp_file);

        return result;
    }

    int dumpIR(std::shared_ptr<ir::Module> ir_module) {
        std::stringstream ss;
        ss << "IR Module: " << ir_module->filename << "\n";
        ss << "Source Language: " << static_cast<int>(ir_module->source) << "\n";
        ss << "Functions: " << ir_module->functions.size() << "\n";
        ss << "Declarations: " << ir_module->declarations.size() << "\n\n";

        for (const auto& func : ir_module->functions) {
            ss << "Function: " << func->name << "\n";
            ss << "  Parameters: " << func->params.size() << "\n";
            ss << "  Statements: " << func->body.size() << "\n";
        }

        return writeFile(options.output_file, ss.str()) ? 0 : 1;
    }

    int verifyAttestation() {
        // Verify binary attestation
        attestation::AttestationSystem& system = attestation::AttestationSystem::getInstance();

        if (system.verifyBinary(options.input_file)) {
            std::cout << "✓ Attestation verified for " << options.input_file << "\n";
            return 0;
        } else {
            std::cout << "✗ Attestation verification failed for " << options.input_file << "\n";
            return 1;
        }
    }

    int roundTripTest(std::shared_ptr<ir::Module> ir_module) {
        // Test bidirectional transpilation
        std::cout << "Round-trip test: " << options.input_file << "\n";

        // Convert to each language and back
        std::string c_code = emitC(ir_module);
        std::string cpp_code = emitCPP(ir_module);
        std::string cpp2_code = emitCPP2(ir_module);

        // Parse each back to IR
        parsers::CParser c_parser(c_code);
        auto ir_from_c = c_parser.parse();

        // Compare IR structures
        bool match = compareIR(ir_module, ir_from_c);

        std::cout << "Round-trip: " << (match ? "PASS" : "FAIL") << "\n";
        return match ? 0 : 1;
    }

    // IR emission functions
    std::string emitC(std::shared_ptr<ir::Module> module) {
        std::stringstream ss;

        // Headers
        ss << "/* Generated by n-way compiler */\n";
        ss << "#include <stdio.h>\n";
        ss << "#include <stdlib.h>\n";
        ss << "#include <stdint.h>\n\n";

        // Forward declarations
        for (const auto& func : module->functions) {
            ss << emitCFunctionDecl(func) << ";\n";
        }
        ss << "\n";

        // Function definitions
        for (const auto& func : module->functions) {
            ss << emitCFunction(func) << "\n\n";
        }

        return ss.str();
    }

    std::string emitCFunction(std::shared_ptr<ir::Function> func) {
        std::stringstream ss;

        // Function signature
        ss << typeToC(func->return_type) << " " << func->name << "(";
        for (size_t i = 0; i < func->params.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << typeToC(func->params[i].second) << " " << func->params[i].first;
        }
        ss << ") {\n";

        // Function body
        for (const auto& stmt : func->body) {
            ss << "    " << statementToC(stmt) << "\n";
        }

        ss << "}";
        return ss.str();
    }

    std::string emitCFunctionDecl(std::shared_ptr<ir::Function> func) {
        std::stringstream ss;
        ss << typeToC(func->return_type) << " " << func->name << "(";
        for (size_t i = 0; i < func->params.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << typeToC(func->params[i].second) << " " << func->params[i].first;
        }
        ss << ")";
        return ss.str();
    }

    std::string typeToC(ir::TypePtr type) {
        if (!type) return "void";

        switch (type->kind) {
            case ir::TypeKind::Void: return "void";
            case ir::TypeKind::Bool: return "_Bool";
            case ir::TypeKind::Int8: return "int8_t";
            case ir::TypeKind::Int16: return "int16_t";
            case ir::TypeKind::Int32: return "int32_t";
            case ir::TypeKind::Int64: return "int64_t";
            case ir::TypeKind::UInt8: return "uint8_t";
            case ir::TypeKind::UInt16: return "uint16_t";
            case ir::TypeKind::UInt32: return "uint32_t";
            case ir::TypeKind::UInt64: return "uint64_t";
            case ir::TypeKind::Float32: return "float";
            case ir::TypeKind::Float64: return "double";
            case ir::TypeKind::Char: return "char";
            case ir::TypeKind::Pointer:
                return typeToC(type->pointee) + "*";
            case ir::TypeKind::Array:
                return typeToC(type->pointee) + "[" + std::to_string(type->array_size) + "]";
            default:
                return "void";
        }
    }

    std::string statementToC(ir::StmtPtr stmt) {
        if (!stmt) return ";";

        switch (stmt->kind) {
            case ir::StmtKind::Expression:
                return expressionToC(stmt->condition) + ";";
            case ir::StmtKind::Return:
                if (stmt->return_value) {
                    return "return " + expressionToC(stmt->return_value) + ";";
                }
                return "return;";
            case ir::StmtKind::If: {
                std::stringstream ss;
                ss << "if (" << expressionToC(stmt->condition) << ") ";
                ss << statementToC(stmt->then_stmt);
                if (stmt->else_stmt) {
                    ss << " else " << statementToC(stmt->else_stmt);
                }
                return ss.str();
            }
            case ir::StmtKind::While:
                return "while (" + expressionToC(stmt->condition) + ") " +
                       statementToC(stmt->then_stmt);
            case ir::StmtKind::Block: {
                std::stringstream ss;
                ss << "{\n";
                for (const auto& s : stmt->body) {
                    ss << "        " << statementToC(s) << "\n";
                }
                ss << "    }";
                return ss.str();
            }
            default:
                return ";";
        }
    }

    std::string expressionToC(ir::ExprPtr expr) {
        if (!expr) return "0";

        switch (expr->kind) {
            case ir::ExprKind::IntLiteral:
                return std::to_string(std::get<int64_t>(expr->data));
            case ir::ExprKind::FloatLiteral:
                return std::to_string(std::get<double>(expr->data));
            case ir::ExprKind::StringLiteral:
                return "\"" + std::get<std::string>(expr->data) + "\"";
            case ir::ExprKind::Identifier:
                return std::get<std::string>(expr->data);
            case ir::ExprKind::BinaryOp: {
                auto& ops = std::get<std::pair<ir::ExprPtr, ir::ExprPtr>>(expr->data);
                return "(" + expressionToC(ops.first) + " " + expr->op +
                       " " + expressionToC(ops.second) + ")";
            }
            case ir::ExprKind::Call: {
                std::stringstream ss;
                ss << "func(";  // TODO: Get actual function name
                auto& args = std::get<std::vector<ir::ExprPtr>>(expr->data);
                for (size_t i = 0; i < args.size(); ++i) {
                    if (i > 0) ss << ", ";
                    ss << expressionToC(args[i]);
                }
                ss << ")";
                return ss.str();
            }
            default:
                return "0";
        }
    }

    std::string emitCPP(std::shared_ptr<ir::Module> module) {
        std::stringstream ss;

        // Headers
        ss << "// Generated by n-way compiler\n";
        ss << "#include <iostream>\n";
        ss << "#include <cstdint>\n";
        ss << "#include <memory>\n";
        ss << "#include <vector>\n\n";

        // Namespaces
        if (!module->namespaces.empty()) {
            for (const auto& [ns, decls] : module->namespaces) {
                ss << "namespace " << ns << " {\n\n";
            }
        }

        // Functions
        for (const auto& func : module->functions) {
            ss << emitCPPFunction(func) << "\n\n";
        }

        // Close namespaces
        if (!module->namespaces.empty()) {
            for (size_t i = 0; i < module->namespaces.size(); ++i) {
                ss << "} // namespace\n";
            }
        }

        return ss.str();
    }

    std::string emitCPPFunction(std::shared_ptr<ir::Function> func) {
        std::stringstream ss;

        // Template parameters
        if (!func->template_params.empty()) {
            ss << "template<";
            for (size_t i = 0; i < func->template_params.size(); ++i) {
                if (i > 0) ss << ", ";
                ss << "typename " << func->template_params[i];
            }
            ss << ">\n";
        }

        // Function qualifiers
        if (func->is_constexpr) ss << "constexpr ";
        if (func->is_inline) ss << "inline ";

        // Return type and name
        ss << typeToCPP(func->return_type) << " " << func->name << "(";

        // Parameters
        for (size_t i = 0; i < func->params.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << typeToCPP(func->params[i].second) << " " << func->params[i].first;
        }
        ss << ")";

        // Qualifiers
        if (func->is_noexcept) ss << " noexcept";

        // Contracts (C++23)
        if (func->requires_clause) {
            ss << "\n    [[pre: " << *func->requires_clause << "]]";
        }
        if (func->ensures_clause) {
            ss << "\n    [[post: " << *func->ensures_clause << "]]";
        }

        ss << " {\n";

        // Function body
        for (const auto& stmt : func->body) {
            ss << "    " << statementToCPP(stmt) << "\n";
        }

        ss << "}";
        return ss.str();
    }

    std::string typeToCPP(ir::TypePtr type) {
        if (!type) return "void";

        std::string base;
        switch (type->kind) {
            case ir::TypeKind::Void: base = "void"; break;
            case ir::TypeKind::Bool: base = "bool"; break;
            case ir::TypeKind::Int8: base = "std::int8_t"; break;
            case ir::TypeKind::Int16: base = "std::int16_t"; break;
            case ir::TypeKind::Int32: base = "std::int32_t"; break;
            case ir::TypeKind::Int64: base = "std::int64_t"; break;
            case ir::TypeKind::UInt8: base = "std::uint8_t"; break;
            case ir::TypeKind::UInt16: base = "std::uint16_t"; break;
            case ir::TypeKind::UInt32: base = "std::uint32_t"; break;
            case ir::TypeKind::UInt64: base = "std::uint64_t"; break;
            case ir::TypeKind::Float32: base = "float"; break;
            case ir::TypeKind::Float64: base = "double"; break;
            case ir::TypeKind::Char: base = "char"; break;
            case ir::TypeKind::Pointer:
                return typeToCPP(type->pointee) + "*";
            case ir::TypeKind::Reference:
                return typeToCPP(type->pointee) + "&";
            case ir::TypeKind::Array:
                return "std::array<" + typeToCPP(type->pointee) + ", " +
                       std::to_string(type->array_size) + ">";
            case ir::TypeKind::Auto:
                return "auto";
            default:
                base = "void";
        }

        // Apply qualifiers
        if (type->is_const) base = "const " + base;
        if (type->is_volatile) base = "volatile " + base;

        return base;
    }

    std::string statementToCPP(ir::StmtPtr stmt) {
        // Similar to statementToC but with C++ features
        return statementToC(stmt);  // Simplified for now
    }

    std::string emitCPP2(std::shared_ptr<ir::Module> module) {
        std::stringstream ss;

        // CPP2 header
        ss << "// Generated by n-way compiler\n\n";

        // Functions
        for (const auto& func : module->functions) {
            ss << emitCPP2Function(func) << "\n\n";
        }

        return ss.str();
    }

    std::string emitCPP2Function(std::shared_ptr<ir::Function> func) {
        std::stringstream ss;

        ss << func->name << ": (";

        // Parameters with CPP2 passing modes
        for (size_t i = 0; i < func->params.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << func->params[i].first << ": " << typeToCPP2(func->params[i].second);
        }

        ss << ")";

        // Return type
        if (func->return_type && func->return_type->kind != ir::TypeKind::Void) {
            ss << " -> " << typeToCPP2(func->return_type);
        }

        // Contracts
        if (func->requires_clause) {
            ss << "\n    pre<{ " << *func->requires_clause << " }>";
        }
        if (func->ensures_clause) {
            ss << "\n    post<{ " << *func->ensures_clause << " }>";
        }

        ss << " = {\n";

        // Function body
        for (const auto& stmt : func->body) {
            ss << "    " << statementToCPP2(stmt) << "\n";
        }

        ss << "}";
        return ss.str();
    }

    std::string typeToCPP2(ir::TypePtr type) {
        if (!type) return "_";

        switch (type->kind) {
            case ir::TypeKind::Void: return "void";
            case ir::TypeKind::Bool: return "bool";
            case ir::TypeKind::Int32: return "int";
            case ir::TypeKind::Int64: return "long";
            case ir::TypeKind::Float32: return "float";
            case ir::TypeKind::Float64: return "double";
            case ir::TypeKind::Char: return "char";
            case ir::TypeKind::Auto: return "_";
            case ir::TypeKind::Pointer:
                return "*" + typeToCPP2(type->pointee);
            default:
                return "_";
        }
    }

    std::string statementToCPP2(ir::StmtPtr stmt) {
        if (!stmt) return ";";

        switch (stmt->kind) {
            case ir::StmtKind::Expression:
                return expressionToCPP2(stmt->condition) + ";";
            case ir::StmtKind::Return:
                if (stmt->return_value) {
                    return "return " + expressionToCPP2(stmt->return_value) + ";";
                }
                return "return;";
            case ir::StmtKind::If:
                return "if " + expressionToCPP2(stmt->condition) + " { " +
                       statementToCPP2(stmt->then_stmt) + " }";
            default:
                return ";";
        }
    }

    std::string expressionToCPP2(ir::ExprPtr expr) {
        // Similar to expressionToC but with CPP2 syntax
        return expressionToC(expr);  // Simplified for now
    }

    // Helper to convert CPP2 AST to IR
    std::shared_ptr<ir::Module> convertCPP2ToIR(const cppfront::AST& ast) {
        auto module = std::make_shared<ir::Module>();
        module->source = ir::SourceLang::CPP2;
        module->filename = options.input_file;

        // Convert functions
        for (const auto& cpp2_func : ast.functions) {
            auto func = std::make_shared<ir::Function>();
            func->name = cpp2_func.name;
            func->source = ir::SourceLang::CPP2;

            // Convert return type
            func->return_type = convertCPP2Type(cpp2_func.return_type);

            // Convert parameters
            for (const auto& param : cpp2_func.params) {
                func->params.push_back({param.name, convertCPP2Type(param.type)});
            }

            // Convert body
            for (const auto& stmt : cpp2_func.body) {
                func->body.push_back(convertCPP2Statement(stmt));
            }

            module->functions.push_back(func);
        }

        return module;
    }

    ir::TypePtr convertCPP2Type(const cppfront::Type& type) {
        ir::IRBuilder builder;

        if (type.name == "int") return builder.makeIntType(32, true);
        if (type.name == "void") return builder.makeVoidType();
        if (type.name == "double") return builder.makeFloatType(64);

        // Default
        return builder.makeVoidType();
    }

    ir::StmtPtr convertCPP2Statement(const cppfront::Statement& stmt) {
        ir::IRBuilder builder;
        // Simplified conversion
        return builder.makeExprStmt(nullptr);
    }

    bool compareIR(std::shared_ptr<ir::Module> a, std::shared_ptr<ir::Module> b) {
        if (!a || !b) return false;

        if (a->functions.size() != b->functions.size()) return false;
        if (a->declarations.size() != b->declarations.size()) return false;

        // Compare functions
        for (size_t i = 0; i < a->functions.size(); ++i) {
            if (a->functions[i]->name != b->functions[i]->name) return false;
            if (a->functions[i]->params.size() != b->functions[i]->params.size()) return false;
            if (a->functions[i]->body.size() != b->functions[i]->body.size()) return false;
        }

        return true;
    }

    std::string readFile(const std::string& path) {
        std::ifstream file(path);
        if (!file) {
            throw std::runtime_error("Cannot open file: " + path);
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }

    bool writeFile(const std::string& path, const std::string& content) {
        std::ofstream file(path);
        if (!file) {
            std::cerr << "Cannot write to file: " << path << "\n";
            return false;
        }
        file << content;
        return true;
    }

    void printAttestationRecord(const attestation::CompilationRecord& record) {
        std::cout << "\nAttestation Record:\n";
        std::cout << "  Source: " << record.source_file << "\n";
        std::cout << "  Source Hash: " << hashToString(record.source_hash) << "\n";
        std::cout << "  Build ID: " << hashToString(record.build_id) << "\n";
        std::cout << "  Output: " << record.output_file << "\n";
        std::cout << "  Output Hash: " << hashToString(record.output_hash) << "\n";
        std::cout << "  Signature: " << hashToString(record.signature) << "\n";
    }

    template<size_t N>
    std::string hashToString(const std::array<uint8_t, N>& hash) {
        std::stringstream ss;
        for (uint8_t byte : hash) {
            ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byte);
        }
        return ss.str();
    }
};

// Main entry point
int main(int argc, char* argv[]) {
    CompilerOptions options;

    // Parse command line
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-o" && i + 1 < argc) {
            options.output_file = argv[++i];
        } else if (arg == "-O0") {
            options.opt_level = 0;
        } else if (arg == "-O1") {
            options.opt_level = 1;
        } else if (arg == "-O2") {
            options.opt_level = 2;
        } else if (arg == "-O3") {
            options.opt_level = 3;
        } else if (arg == "-g") {
            options.debug_info = true;
        } else if (arg == "--no-attestation") {
            options.enable_attestation = false;
        } else if (arg == "--no-deterministic") {
            options.deterministic = false;
        } else if (arg == "--verify") {
            options.mode = CompileMode::Verify;
        } else if (arg == "--round-trip") {
            options.mode = CompileMode::RoundTrip;
        } else if (arg == "--ir-dump") {
            options.mode = CompileMode::IRDump;
        } else if (arg == "-v" || arg == "--verbose") {
            options.verbose = true;
        } else if (arg == "--time") {
            options.time_report = true;
        } else if (arg == "-I" && i + 1 < argc) {
            options.include_paths.push_back(argv[++i]);
        } else if (arg == "-L" && i + 1 < argc) {
            options.lib_paths.push_back(argv[++i]);
        } else if (arg == "--key" && i + 1 < argc) {
            options.key_file = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "N-Way Compiler with Attestation\n";
            std::cout << "Usage: " << argv[0] << " [options] input_file\n\n";
            std::cout << "Options:\n";
            std::cout << "  -o <file>           Output file\n";
            std::cout << "  -O[0-3]             Optimization level\n";
            std::cout << "  -g                  Debug information\n";
            std::cout << "  --no-attestation    Disable attestation\n";
            std::cout << "  --no-deterministic  Disable deterministic compilation\n";
            std::cout << "  --verify            Verify attestation\n";
            std::cout << "  --round-trip        Test round-trip transpilation\n";
            std::cout << "  --ir-dump           Dump IR representation\n";
            std::cout << "  -I <path>           Include path\n";
            std::cout << "  -L <path>           Library path\n";
            std::cout << "  --key <file>        Private key file\n";
            std::cout << "  -v, --verbose       Verbose output\n";
            std::cout << "  --time              Show timing information\n";
            std::cout << "  -h, --help          Show this help\n";
            return 0;
        } else if (!arg.empty() && arg[0] != '-') {
            options.input_file = arg;
        }
    }

    // Validate options
    if (options.input_file.empty()) {
        std::cerr << "Error: No input file specified\n";
        std::cerr << "Use --help for usage information\n";
        return 1;
    }

    if (options.output_file.empty() && options.mode != CompileMode::Verify) {
        // Default output name
        fs::path input_path(options.input_file);
        options.output_file = input_path.stem().string();

        if (options.mode == CompileMode::Transpile) {
            options.output_file += ".out" + input_path.extension().string();
        } else if (options.mode == CompileMode::IRDump) {
            options.output_file += ".ir";
        } else {
            options.output_file += ".out";
        }
    }

    try {
        NWayCompiler compiler(options);
        return compiler.compile();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}