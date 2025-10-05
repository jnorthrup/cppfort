#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>

#include "orbit_scanner.h"
#include "wide_scanner.h"
#include "multi_grammar_loader.h"

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <command> <input_file> [options...]\n";
        std::cerr << "Commands:\n";
        std::cerr << "  scan <file>          - Scan file and emit orbit streams with telemetry\n";
        std::cerr << "  anchors <file>       - Generate and display anchor points\n";
        std::cerr << "  boundaries <file>    - Scan and display boundaries with orbit data\n";
        return 1;
    }

    std::string command = argv[1];
    std::string input_file = argv[2];

    // Check if input file exists
    if (!fs::exists(input_file)) {
        std::cerr << "Error: Input file '" << input_file << "' does not exist\n";
        return 1;
    }

    // Read input file
    std::ifstream file(input_file, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open input file '" << input_file << "'\n";
        return 1;
    }

    std::string source((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
    file.close();

    std::cout << "Loaded " << source.size() << " bytes from " << input_file << "\n";

    if (command == "scan") {
        // Use wide scanner for basic analysis
        auto anchors = cppfort::ir::WideScanner::generateAlternatingAnchors(source);
        cppfort::ir::WideScanner scanner;
        auto boundaries = scanner.scanAnchorsWithOrbits(source, anchors);

        std::cout << "\n=== Wide Scanner Results ===\n";
        std::cout << "Loaded " << source.size() << " bytes\n";
        std::cout << "Generated " << anchors.size() << " anchor points\n";
        std::cout << "Found " << boundaries.size() << " boundaries\n";

        // Show sample boundaries
        if (!boundaries.empty()) {
            std::cout << "\nSample boundaries:\n";
            size_t count = std::min(size_t(10), boundaries.size());
            for (size_t i = 0; i < count; ++i) {
                const auto& boundary = boundaries[i];
                std::cout << "  pos=" << boundary.position
                         << " delim=" << (boundary.is_delimiter ? std::string(1, boundary.delimiter) : " ")
                         << " conf=" << boundary.orbit_confidence
                         << " mask=0x" << std::hex << boundary.lattice_mask << std::dec << "\n";
            }
        }

    } else if (command == "anchors") {
        // Generate anchor points
        auto anchors = cppfort::ir::WideScanner::generateAlternatingAnchors(source);

        std::cout << "Generated " << anchors.size() << " anchor points\n";
        for (size_t i = 0; i < std::min(size_t(20), anchors.size()); ++i) {
            const auto& anchor = anchors[i];
            std::cout << "  pos=" << anchor.position
                     << " spacing=" << anchor.spacing
                     << " utf8=" << (anchor.is_utf8_boundary ? "true" : "false") << "\n";
        }

    } else if (command == "boundaries") {
        // Generate anchors and scan boundaries
        auto anchors = cppfort::ir::WideScanner::generateAlternatingAnchors(source);
        cppfort::ir::WideScanner scanner;
        auto boundaries = scanner.scanAnchorsWithOrbits(source, anchors);

        std::cout << "Found " << boundaries.size() << " boundaries\n";
        for (size_t i = 0; i < std::min(size_t(20), boundaries.size()); ++i) {
            const auto& boundary = boundaries[i];
            std::cout << "  pos=" << boundary.position
                     << " delim=" << (boundary.is_delimiter ? std::string(1, boundary.delimiter) : " ")
                     << " conf=" << boundary.orbit_confidence
                     << " mask=0x" << std::hex << boundary.lattice_mask << std::dec << "\n";
        }

    } else {
        std::cerr << "Unknown command: " << command << "\n";
        return 1;
    }

    return 0;
}