#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>

#include "cpp2_emitter.h"
#include "pattern_loader.h"

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file> [pattern_file]\n";
        return 1;
    }

    std::string input_file = argv[1];
    std::string output_file = argv[2];
    std::string pattern_file = argc > 3 ? argv[3] : "../../patterns/cpp2_core_patterns.yaml";

    // Check input file
    if (!fs::exists(input_file)) {
        std::cerr << "Error: Input file '" << input_file << "' does not exist\n";
        return 1;
    }

    // Read input
    std::ifstream file(input_file, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open input file '" << input_file << "'\n";
        return 1;
    }

    std::string source((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
    file.close();

    std::cout << "Loaded " << source.size() << " bytes from " << input_file << "\n";

    // Load patterns
    cppfort::stage0::PatternLoader pattern_loader;
    if (!pattern_loader.load_yaml(pattern_file)) {
        std::cerr << "Warning: Failed to load patterns from " << pattern_file << "\n";
        std::cerr << "Attempting to continue without patterns...\n";
    } else {
        std::cout << "Loaded " << pattern_loader.pattern_count() << " patterns\n";
    }

    // Open output
    std::ofstream outfile(output_file);
    if (!outfile) {
        std::cerr << "Error: Cannot open output file '" << output_file << "'\n";
        return 1;
    }

    // Transpile using depth-based emitter
    cppfort::stage0::CPP2Emitter emitter;
    emitter.emit_depth_based(source, outfile, pattern_loader.patterns());
    outfile.close();

    std::cout << "Transpiled to " << output_file << "\n";
    return 0;
}
