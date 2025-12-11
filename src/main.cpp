#include <iostream>
#include <fstream>
#include <span>
#include <string_view>
#include <print>

#include "lexer.hpp"
#include "parser.hpp"
#include "semantic_analyzer.hpp"
#include "code_generator.hpp"
#include "safety_checker.hpp"
#include "metafunction_processor.hpp"
#include "contract_processor.hpp"
#include "utils.hpp"

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::println("Usage: {} <input.cpp2> <output.cpp>", argv[0]);
        return 1;
    }

    try {
        std::string input_filename = argv[1];
        std::string output_filename = argv[2];

        std::ifstream input_file(input_filename);
        if (!input_file) {
            throw std::runtime_error("Cannot open input file: " + input_filename);
        }

        std::string source_code(std::istreambuf_iterator<char>(input_file),
                                std::istreambuf_iterator<char>());

        cpp2_transpiler::Lexer lexer(source_code);
        auto tokens = lexer.tokenize();

        cpp2_transpiler::Parser parser(tokens);
        auto ast = parser.parse();

        cpp2_transpiler::SemanticAnalyzer semantic_analyzer;
        semantic_analyzer.analyze(ast);

        cpp2_transpiler::SafetyChecker safety_checker;
        safety_checker.check(ast);

        cpp2_transpiler::MetafunctionProcessor meta_processor;
        meta_processor.process(ast);

        cpp2_transpiler::ContractProcessor contract_processor;
        contract_processor.process(ast);

        cpp2_transpiler::CodeGenerator code_generator;
        auto cpp1_code = code_generator.generate(ast);

        std::ofstream output_file(output_filename);
        if (!output_file) {
            throw std::runtime_error("Cannot open output file: " + output_filename);
        }

        output_file << cpp1_code;

        std::println("Successfully transpiled {} to {}", input_filename, output_filename);
        return 0;
    }
    catch (const std::exception& e) {
        std::println("Error: {}", e.what());
        return 1;
    }
}