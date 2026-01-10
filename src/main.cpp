#include <iostream>
#include <fstream>
#include <sstream>
#include <span>
#include <string_view>
#include <cstring>

#include "lexer.hpp"
#include "combinator_parser.hpp"
#include "slim_ast.hpp"
#include "emitter.hpp"

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options] <input.cpp2> <output.cpp>\n";
    std::cerr << "\nOptions:\n";
    std::cerr << "  --inline    Inline runtime (default, self-contained output)\n";
    std::cerr << "  --header    Use #include <cpp2_runtime.h>\n";
    std::cerr << "  --pch       Use #include <cpp2_pch.h> (precompiled header)\n";
    std::cerr << "  --help      Show this help\n";
}

int main(int argc, char* argv[]) {
    const char* input_file_arg = nullptr;
    const char* output_file_arg = nullptr;
    
    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--inline") == 0 ||
            strcmp(argv[i], "--header") == 0 ||
            strcmp(argv[i], "--pch") == 0) {
            // Options currently ignored - future: pass to emitter
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

        // All files go through the parser (no C++1 passthrough bypass)
        // This ensures preprocessor directives are preserved in the AST

        // Parse with slim combinator parser
        auto tree = cpp2::parser::parse(tokens);

        // Check for valid parse
        if (tree.nodes.empty() || tree.nodes[tree.root].child_count == 0) {
            std::cerr << "Error: Parsing failed - no declarations found\n";
            return 1;
        }

        // Generate C++ directly from ParseTree
        std::string cpp1_code = generate_from_tree(tree, tokens);

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