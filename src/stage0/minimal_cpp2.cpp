// MINIMAL CPP2 Self-Hosting - Data-driven orbits only
#include "wide_scanner.h"
#include "rbcursive.h"
#include <iostream>
#include <fstream>
#include <string>

int main(int argc, char* argv[]) {
    if (argc < 2) return 1;

    // Read source
    std::ifstream file(argv[1]);
    std::string source((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());

    // Scanner generates anchors
    auto anchors = cppfort::ir::WideScanner::generateAlternatingAnchors(source);
    cppfort::ir::WideScanner scanner;
    auto boundaries = scanner.scanAnchorsWithOrbits(source, anchors);

    // Direct emit based on boundaries
    size_t last = 0;
    for (const auto& b : boundaries) {
        // Emit segment
        std::cout << source.substr(last, b.position - last);
        if (b.delimiter == ':') std::cout << ":";
        last = b.position + 1;
    }
    std::cout << source.substr(last);

    return 0;
}