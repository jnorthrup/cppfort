#include <iostream>
#include <string>
#include <fstream>

#include "bidirectional.h"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <test_file>\n";
        return 1;
    }

    std::string filename = argv[1];
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Failed to open file: " << filename << "\n";
        return 1;
    }

    std::string source((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());

    cppfort::stage0::BidirectionalTranspiler transpiler;

    std::cout << "Original source:\n" << source << "\n\n";

    // Test cpp2 -> C++ -> cpp2 roundtrip
    if (filename.find(".cpp2") != std::string::npos) {
        std::cout << "Testing cpp2 -> C++ -> cpp2 roundtrip:\n";

        std::string cpp_output = transpiler.roundtrip_cpp2(source, filename);
        std::cout << "Roundtrip result:\n" << cpp_output << "\n";

        // Test parsing and emitting as C++
        auto ast = transpiler.parse_cpp2(source, filename);
        cppfort::stage0::TransformOptions cpp_options;
        cpp_options.target_cpp2 = false;
        std::string cpp_emit = transpiler.emit_cpp(ast, cpp_options);
        std::cout << "Emitted as C++:\n" << cpp_emit << "\n";
    } else {
        std::cout << "Testing C++ -> cpp2 -> C++ roundtrip:\n";

        std::string cpp2_output = transpiler.roundtrip_cpp(source, filename);
        std::cout << "Roundtrip result:\n" << cpp2_output << "\n";

        // Test parsing and emitting as cpp2
        auto ast = transpiler.parse_cpp(source, filename);
        cppfort::stage0::TransformOptions cpp2_options;
        cpp2_options.target_cpp2 = true;
        std::string cpp2_emit = transpiler.emit_cpp2(ast, cpp2_options);
        std::cout << "Emitted as cpp2:\n" << cpp2_emit << "\n";
    }

    return 0;
}