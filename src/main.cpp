#include <iostream>
#include <fstream>
#include <span>
#include <string_view>
#include <iostream>
#include <cstring>

#include "lexer.hpp"
#include "parser.hpp"
#include "semantic_analyzer.hpp"
#include "code_generator.hpp"
#include "safety_checker.hpp"
#include "metafunction_processor.hpp"
#include "contract_processor.hpp"
#include "utils.hpp"
// #include "mlir_cpp2_dialect.hpp"  // Disabled: requires MLIR

// Forward declaration for the Sea of Nodes codegen helper
// namespace cppfort::mlir_son { class SeaOfNodesBuilder; }
// std::string generate_cpp_from_sea_of_nodes(const cppfort::mlir_son::SeaOfNodesBuilder& builder);

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options] <input.cpp2> <output.cpp>\n";
    std::cerr << "\nOptions:\n";
    std::cerr << "  --inline    Inline runtime (default, self-contained output)\n";
    std::cerr << "  --header    Use #include <cpp2_runtime.h>\n";
    std::cerr << "  --pch       Use #include <cpp2_pch.h> (precompiled header)\n";
    std::cerr << "  --help      Show this help\n";
}

int main(int argc, char* argv[]) {
    cpp2_transpiler::OutputMode output_mode = cpp2_transpiler::OutputMode::Inline;
    const char* input_file_arg = nullptr;
    const char* output_file_arg = nullptr;
    
    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--inline") == 0) {
            output_mode = cpp2_transpiler::OutputMode::Inline;
        } else if (strcmp(argv[i], "--header") == 0) {
            output_mode = cpp2_transpiler::OutputMode::Header;
        } else if (strcmp(argv[i], "--pch") == 0) {
            output_mode = cpp2_transpiler::OutputMode::PCH;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (argv[i][0] != '-') {
            if (!input_file_arg) {
                input_file_arg = argv[i];
            } else if (!output_file_arg) {
                output_file_arg = argv[i];
            } else {
                std::cerr << "Error: Too many arguments\n";
                print_usage(argv[0]);
                return 1;
            }
        } else {
            std::cerr << "Error: Unknown option: " << argv[i] << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }
    
    if (!input_file_arg || !output_file_arg) {
        print_usage(argv[0]);
        return 1;
    }

    try {
        std::string input_filename = input_file_arg;
        std::string output_filename = output_file_arg;

        std::ifstream input_file(input_filename);
        if (!input_file) {
            throw std::runtime_error("Cannot open input file: " + input_filename);
        }

        std::string source_code{std::istreambuf_iterator<char>(input_file),
                                std::istreambuf_iterator<char>()};

        cpp2_transpiler::Lexer lexer{std::string_view(source_code)};
        auto tokens = lexer.tokenize();

        // Mixed-mode C++1 passthrough: If no Cpp2-specific syntax is found,
        // emit the original source with minimal changes
        if (!lexer.has_cpp2_syntax()) {
            std::ofstream output_file(output_filename);
            if (!output_file) {
                throw std::runtime_error("Cannot open output file: " + output_filename);
            }

            // For pure C++ files, emit as-is (source already has cppfront-style comments)
            output_file << source_code;

            std::cout << "Successfully transpiled " << input_filename << " to " << output_filename << " (C++1 passthrough mode)\n";
            return 0;
        }

        cpp2_transpiler::Parser parser(tokens);
        auto ast = parser.parse();

        // Abort if parsing had errors - continuing with incomplete AST causes segfaults
        if (parser.had_errors()) {
            std::cerr << "Error: Parsing failed with " << parser.error_count << " error(s)\n";
            return 1;
        }

        cpp2_transpiler::SemanticAnalyzer semantic_analyzer;
        semantic_analyzer.analyze(*ast);

        cpp2_transpiler::SafetyChecker safety_checker;
        safety_checker.check(*ast);

        cpp2_transpiler::MetafunctionProcessor meta_processor;
        meta_processor.process(*ast);

        cpp2_transpiler::ContractProcessor contract_processor;
        contract_processor.process(*ast);

        // Default (AST -> Cpp1) code generation
        cpp2_transpiler::CodeGenerator code_generator(output_mode);
        auto cpp1_code = code_generator.generate(*ast);

        // --son pipeline removed for now (requires additional integration)

        std::ofstream output_file(output_filename);
        if (!output_file) {
            throw std::runtime_error("Cannot open output file: " + output_filename);
        }

        output_file << cpp1_code;

        std::cout << "Successfully transpiled " << input_filename << " to " << output_filename << "\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}