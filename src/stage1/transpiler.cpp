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
    // Support multiple invocation styles:
    // 1) transpiler <input.cpp2> <output.cpp>
    // 2) transpiler transpile <input.cpp2> <output.cpp> (used by regression scripts)
    // 3) transpiler <input.cpp2> <output.cpp> --inline-cpp2 (inline cpp2.h)
    // 4) transpiler <input.cpp2> <output.cpp> --bundle-cpp2 (bundle cpp2.h)

    std::string input_path;
    std::string output_path;
    bool inline_cpp2 = false;
    bool bundle_cpp2 = false;

    // Parse command line arguments
    if (argc >= 3) {
        // Check for flags at the end
        int arg_count = argc;
        while (arg_count > 3 && argv[arg_count - 1][0] == '-') {
            std::string flag = argv[arg_count - 1];
            if (flag == "--inline-cpp2") {
                inline_cpp2 = true;
                arg_count--;
            } else if (flag == "--bundle-cpp2") {
                bundle_cpp2 = true;
                arg_count--;
            } else {
                break;
            }
        }

        // Parse the basic arguments
        if (arg_count == 3) {
            input_path = argv[1];
            output_path = argv[2];
        } else if (arg_count == 4 && std::string(argv[1]) == "transpile") {
            input_path = argv[2];
            output_path = argv[3];
        } else {
            std::cerr << "Usage: transpiler <input.cpp2> <output.cpp> [options]\n";
            std::cerr << "   or: transpiler transpile <input.cpp2> <output.cpp>\n";
            std::cerr << "Options:\n";
            std::cerr << "  --inline-cpp2   Inline cpp2.h contents in generated code\n";
            std::cerr << "  --bundle-cpp2   Bundle cpp2.h contents at beginning of output\n";
            return 1;
        }
    } else {
        std::cerr << "Usage: transpiler <input.cpp2> <output.cpp> [options]\n";
        // Set up Stage0 (cpp2 -> C++) transpiler and emitter, honoring
        // command-line flags for inlining/bundling cpp2.h.
        std::string src;
        std::string line;

        cppfort::stage0::Transpiler transpiler;
        cppfort::stage0::Emitter emitter;
        cppfort::stage0::EmitOptions emit_options;
        emit_options.include_headers = true;
        emit_options.inline_cpp2_header = inline_cpp2;  // Inline cpp2.h contents for standalone output
        emit_options.bundle_headers = bundle_cpp2;       // Bundle cpp2.h contents at beginning of output
        std::string transformed;
    while (std::getline(in, line)) {
        src += line + "\n";
    }
    in.close();

    // Use stage0 Transpiler pipeline to transpile cpp2 to C++
    cppfort::stage0::Transpiler transpiler;
    cppfort::stage0::Emitter emitter;
    cppfort::stage0::EmitOptions emit_options;
    emit_options.include_headers = true;
    emit_options.inline_cpp2_header = false;  // Do not inline cpp2.h contents for standalone output
    emit_options.bundle_headers = false;       // Bundle cpp2.h contents at beginning of output
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
