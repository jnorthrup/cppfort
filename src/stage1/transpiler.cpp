#include <fstream>
#include <sstream>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cctype>
#include <cassert>

#include "../stage0/transpiler.h"
#include "../stage0/emitter.h"
#include "../stage0/ir_serializer.h"

int main(int argc, char* argv[]) {
    // Support multiple invocation styles:
    // 1) transpiler <input.cpp2> <output.cpp>
    // 2) transpiler transpile <input.cpp2> <output.cpp> (used by regression scripts)
    // 3) transpiler transpile <input.ir> <output.cpp> --input-format ir
    std::string input_path;
    std::string output_path;
    std::string input_format = "cpp2";  // default to cpp2 source

    if (argc == 3) {
        input_path = argv[1];
        output_path = argv[2];
    } else if (argc >= 4 && std::string(argv[1]) == "transpile") {
        input_path = argv[2];
        output_path = argv[3];

        // Parse additional options
        for (int i = 4; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--input-format" && i + 1 < argc) {
                input_format = argv[++i];
            }
        }
    } else {
        std::cerr << "Usage: transpiler <input.cpp2> <output.cpp>\n";
        std::cerr << "   or: transpiler transpile <input> <output.cpp> [--input-format <cpp2|ir>]\n";
        return 1;
    }

    // Read input file
    std::ifstream in(input_path);
    if (!in) {
        std::cerr << "Failed to open input file: " << input_path << "\n";
        return 1;
    }

    std::string src;
    std::string line;
    while (std::getline(in, line)) {
        src += line + "\n";
    }
    in.close();

    // Process based on input format
    cppfort::stage0::Emitter emitter;
    cppfort::stage0::EmitOptions emit_options;
    emit_options.include_headers = true;
    emit_options.inline_cpp2_header = false;  // Do not inline cpp2.h contents for standalone output
    emit_options.bundle_headers = false;       // Bundle cpp2.h contents at beginning of output
    std::string transformed;

    try {
        cppfort::stage0::TranslationUnit ast;

        if (input_format == "ir") {
            // Deserialize IR format directly to AST
            std::cout << "Reading IR format input\n";
            ast = cppfort::stage0::IRSerializer::deserialize(src);
            std::cout << "Deserialized IR with " << ast.functions.size() << " functions\n";
        } else {
            // Parse cpp2 source to AST
            cppfort::stage0::Transpiler transpiler;
            ast = transpiler.parse(src, input_path);
            std::cout << "Parsed AST with " << ast.functions.size() << " functions\n";
        }

        // Debug: print function info
        for (const auto& fn : ast.functions) {
            std::cout << "Function: " << fn.name << "\n";
            for (const auto& p : fn.parameters) {
                int k = static_cast<int>(p.kind);
                std::cout << "  param " << p.name << " kind=" << k << " type='" << p.type << "'\n";
            }
        }

        // Emit AST as C++ source
        transformed = emitter.emit(ast, emit_options);
        std::cout << "Emitted C++ code, length: " << transformed.size() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Transpilation failed: " << e.what() << "\n";
        return 1;
    }

    // Write output
    std::ofstream out(output_path);
    if (!out) {
        std::cerr << "Failed to open output file: " << output_path << "\n";
        return 1;
    }

    out << transformed;
    std::cout << "Wrote transformed C++ to " << output_path << "\n";
    return 0;
}
