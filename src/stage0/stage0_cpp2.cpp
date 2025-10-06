// Stage0 CPP2 Self-Hosting Transpiler
// Pure data-driven, orbit-based, no IR layer

#include "orbit_scanner.h"
#include "wide_scanner.h"
#include "unified_orbit_patterns.h"
#include "rbcursive.h"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace cppfort::ir;

// Direct orbit → CPP2 emission (no IR)
class CPP2Emitter {
public:
    std::string emit(const DetectionResult& detection, const std::string& source) {
        std::stringstream out;

        // Direct pattern-to-syntax mapping
        for (const auto& match : detection.matches) {
            emitPattern(out, match, source);
        }

        return out.str();
    }

private:
    void emitPattern(std::stringstream& out, const OrbitMatch& match, const std::string& source) {
        // Extract matched text
        std::string matched = source.substr(match.position, match.length);

        // Direct emission based on orbit type
        switch (match.orbit_type) {
            case 1: // Function
                out << matched << "\n";
                break;
            case 2: // Declaration
                out << matched << "\n";
                break;
            default:
                out << matched;
        }
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input.cpp2>\n";
        return 1;
    }

    // Read input
    std::ifstream file(argv[1]);
    std::string source((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());

    // Scanner → Orbits → Patterns
    OrbitScannerConfig config("patterns");
    OrbitScanner scanner(config);
    scanner.initialize();

    auto detection = scanner.scan(source);

    // Direct emission
    CPP2Emitter emitter;
    std::string output = emitter.emit(detection, source);

    std::cout << output;

    return 0;
}