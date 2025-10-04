// Test WideScanner with XAI 4.2 orbit system
// No other parser exists - THIS IS THE SCANNER

#include "src/stage0/wide_scanner.h"
#include "src/stage0/xai_orbit_types.h"
#include "src/stage0/heuristic_grid.h"
#include "src/stage0/lattice_classes.h"
#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>

using namespace cppfort;

void test_lattice_classification() {
    std::cout << "=== Test 1: Lattice Byte Classification ===\n";

    // Test colon classification
    uint16_t colon_mask = stage0::classify_byte(':');
    std::cout << "':' lattice mask: 0x" << std::hex << colon_mask << std::dec << "\n";
    std::cout << "  PUNCTUATION: " << ((colon_mask & stage0::LatticeClasses::PUNCTUATION) ? "YES" : "NO") << "\n";

    // Test identifier
    uint16_t id_mask = stage0::classify_byte('x');
    std::cout << "'x' lattice mask: 0x" << std::hex << id_mask << std::dec << "\n";
    std::cout << "  ALPHA: " << ((id_mask & stage0::LatticeClasses::ALPHA) ? "YES" : "NO") << "\n";
    std::cout << "  IDENTIFIER: " << ((id_mask & stage0::LatticeClasses::IDENTIFIER) ? "YES" : "NO") << "\n";

    std::cout << "\n";
}

void test_heuristic_grid() {
    std::cout << "=== Test 2: HeuristicGrid 4KB Chunk Analysis ===\n";

    std::string cpp2_code = R"(
        x: int = 5;
        func: (y: int) -> int = y * 2;
        point: type = { x: double; y: double; }
    )";

    std::span<const char> code_span(cpp2_code.data(), cpp2_code.size());
    stage0::HeuristicGrid grid(0);
    grid.build_grid(code_span);

    std::cout << "Grid mask: 0x" << std::hex << grid.grid_mask << std::dec << "\n";
    std::cout << "Prediction accuracy: " << grid.prediction_accuracy << "\n";
    std::cout << "Tiles analyzed: " << grid.tiles.size() << "\n";

    auto orbits = grid.estimate_confix_orbits();
    std::cout << "Detected orbits:\n";
    for (const auto& [name, conf] : orbits) {
        std::cout << "  " << name << ": " << conf << "\n";
    }

    std::cout << "\n";
}

void test_anchor_tuple() {
    std::cout << "=== Test 3: XAI 4.2 AnchorTuple (5 concurrent anchors) ===\n";

    std::string cpp2_code = "x: int = 5;";
    std::span<const char> code_span(cpp2_code.data(), cpp2_code.size());

    stage0::AnchorTuple tuple(code_span);
    tuple.interleave_evidence();

    std::cout << "Composite confidence: " << tuple.composite_confidence << "\n";
    std::cout << "Individual anchors:\n";

    const char* anchor_names[] = {
        "COUNT_DELIMITERS",
        "INDENTATION",
        "NUMBER_DUCK_TYPE",
        "LEGAL_CLASSES",
        "CASCADING_RANGES"
    };

    for (size_t i = 0; i < 5; ++i) {
        std::cout << "  " << anchor_names[i] << ": "
                  << tuple.anchors[i].confidence << "\n";
    }

    std::cout << "\n";
}

void test_wide_scanner_basic() {
    std::cout << "=== Test 4: WideScanner Basic SIMD Scanning ===\n";

    std::string code = "int main() { return 0; }";

    auto anchors = ir::WideScanner::generateAlternatingAnchors(code);
    std::cout << "Generated " << anchors.size() << " anchor points\n";

    auto boundaries = ir::WideScanner::scanAnchorsSIMD(code, anchors);
    std::cout << "Detected " << boundaries.size() << " boundaries\n";

    for (size_t i = 0; i < std::min(boundaries.size(), size_t(5)); ++i) {
        auto& b = boundaries[i];
        std::cout << "  Boundary " << i << ": pos=" << b.position
                  << " delim='" << b.delimiter << "' is_delim=" << b.is_delimiter << "\n";
    }

    std::cout << "\n";
}

void test_regression_file(const std::string& filepath) {
    std::cout << "=== Testing: " << std::filesystem::path(filepath).filename() << " ===\n";

    std::ifstream file(filepath);
    if (!file) {
        std::cout << "ERROR: Cannot open file\n\n";
        return;
    }

    std::string code((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());

    std::cout << "File size: " << code.size() << " bytes\n";

    // Phase 1: Lattice classification on first 1KB
    size_t sample_size = std::min(code.size(), size_t(1024));
    std::span<const char> sample_span(code.data(), sample_size);

    stage0::HeuristicGrid grid(0);
    grid.build_grid(sample_span);

    std::cout << "Lattice grid mask: 0x" << std::hex << grid.grid_mask << std::dec << "\n";
    std::cout << "Orbit confidence: " << grid.prediction_accuracy << "\n";

    // Phase 2: WideScanner delimiter detection
    auto anchors = ir::WideScanner::generateAlternatingAnchors(code);
    auto boundaries = ir::WideScanner::scanAnchorsSIMD(code, anchors);

    std::cout << "Anchors: " << anchors.size() << ", Boundaries: " << boundaries.size() << "\n";

    // Phase 3: Count colon occurrences (key disambiguation target)
    size_t colon_count = std::count(code.begin(), code.end(), ':');
    std::cout << "Colons detected: " << colon_count << " (disambiguation targets)\n";

    std::cout << "\n";
}

int main() {
    std::cout << "=== WideScanner + XAI 4.2 Orbit System Test Suite ===\n\n";

    test_lattice_classification();
    test_heuristic_grid();
    test_anchor_tuple();
    test_wide_scanner_basic();

    // Test on actual regression files
    std::vector<std::string> test_files = {
        "regression-tests/mixed-bounds-check.cpp2",
        "regression-tests/pure2-hello.cpp2",
        "regression-tests/mixed-function-expression-and-std-ranges-for-each.cpp2"
    };

    for (const auto& file : test_files) {
        if (std::filesystem::exists(file)) {
            test_regression_file(file);
        }
    }

    std::cout << "=== All Tests Complete ===\n";
    return 0;
}
