#include <fstream>
#include <sstream>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>

#include "../stage0/emitter.h"

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: transpiler <input.cpp2> <output.cpp>\n";
        return 1;
    }

    const std::string input_path = argv[1];
    const std::string output_path = argv[2];

    // NOTE: This implementation now performs a minimal translation
    // from a simple cpp2 sketch to a valid C++ program.
    // It currently understands only a top‑level `main: () -> int = { … }`
    // construct and will rewrite it as a standard `int main() { … }`.
    // Any other content is passed through unchanged.

    std::ifstream in(input_path);
    if (!in) {
        std::cerr << "Failed to open input file: " << input_path << "\n";
        return 1;
    }

    std::ostringstream src_buf;
    src_buf << in.rdbuf();
    std::string src = src_buf.str();

    // Simple regex to locate the cpp2 main declaration.
    // It captures the body between the opening brace '{' and the closing brace '}'.
    std::regex main_regex(R"(main\s*:\s*\(\)\s*->\s*int\s*=\s*\{)");
    std::string transformed = std::regex_replace(src, main_regex, "int main() {");

    // Write the transformed source to the output file.
    std::ofstream out(output_path);
    if (!out) {
        std::cerr << "Failed to open output file: " << output_path << "\n";
        return 1;
    }

    out << transformed;
    std::cout << "Wrote transformed C++ to " << output_path << "\n";
    return 0;
}