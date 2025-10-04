#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "wide_scanner.h"

namespace {
void print_usage() {
    ::std::cerr << "Usage:\n";
    ::std::cerr << "  stage0 scan <input>\n";
}

::std::string read_file(const ::std::filesystem::path& path) {
    ::std::ifstream input(path);
    if (!input) {
        throw ::std::runtime_error("Failed to open input file: " + path.string());
    }
    ::std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    ::std::string command = argv[1];

    try {
        if (command == "scan") {
            if (argc < 3) {
                print_usage();
                return 1;
            }
            ::std::filesystem::path input_path = argv[2];
            auto source = read_file(input_path);

            // Generate orbit anchors
            auto anchors = cppfort::ir::WideScanner::generateAlternatingAnchors(source);
            ::std::cout << "Generated " << anchors.size() << " anchor points\n";

            // Scan with orbits
            auto boundaries = cppfort::ir::WideScanner::scanAnchorsWithOrbits(source, anchors);
            ::std::cout << "Found " << boundaries.size() << " boundaries\n";

            // Show first 10 boundaries
            for (size_t i = 0; i < std::min(boundaries.size(), size_t(10)); ++i) {
                const auto& b = boundaries[i];
                ::std::cout << "  pos=" << b.position
                          << " delim=" << (b.is_delimiter ? b.delimiter : ' ')
                          << " conf=" << b.orbit_confidence
                          << " mask=0x" << std::hex << b.lattice_mask << std::dec
                          << "\n";
            }

            return 0;
        }

        print_usage();
        return 1;
    } catch (const ::std::exception& ex) {
        ::std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
}
