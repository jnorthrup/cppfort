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

int main(int argc, char* argv[]) {
    // Support two invocation styles:
    // 1) transpiler <input.cpp2> <output.cpp>
    // 2) transpiler transpile <input.cpp2> <output.cpp> (used by regression scripts)
    std::string input_path;
    std::string output_path;
    if (argc == 3) {
        input_path = argv[1];
        output_path = argv[2];
    } else if (argc == 4 && std::string(argv[1]) == "transpile") {
        input_path = argv[2];
        output_path = argv[3];
    } else {
        std::cerr << "Usage: transpiler <input.cpp2> <output.cpp>\n";
        std::cerr << "   or: transpiler transpile <input.cpp2> <output.cpp>\n";
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

    // Use stage0 Transpiler pipeline to transpile cpp2 to C++
    cppfort::stage0::Transpiler transpiler;
    cppfort::stage0::Emitter emitter;
    cppfort::stage0::EmitOptions emit_options;
    emit_options.include_headers = true;
    std::string transformed;
    try {
        // Parse cpp2 source to AST
        auto ast = transpiler.parse(src, input_path);
        std::cout << "Parsed AST with " << ast.functions.size() << " functions\n";
        // Debug: print parameter kinds for each function
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
