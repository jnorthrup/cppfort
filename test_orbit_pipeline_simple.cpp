#include <iostream>
#include <fstream>
#include <string>
#include "src/stage0/wide_scanner.h"
#include "src/stage0/rbcursive_regions.h"
#include "src/stage0/region_node.h"
#include "src/stage0/pattern_applier.h"

using namespace cppfort;

std::string loadFile(const std::string& path) {
    std::ifstream file(path);
    if (!file) return "";
    std::string content((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
    return content;
}

void testStructuralCarving() {
    std::cout << "\n═══════════════════════════════════════════════════════════\n";
    std::cout << "TEST: Structural Carving Pipeline\n";
    std::cout << "═══════════════════════════════════════════════════════════\n";
    
    // Test with a simple Cpp2 function
    std::string source = R"(
main : (args) = {
    std::cout << "Hello, Cpp2!";
    return 0;
}
)";
    
    std::cout << "Source code:\n" << source << "\n\n";
    
    // Step 1: Generate boundary events using WideScanner
    std::cout << "Step 1: Generating boundary events...\n";
    auto anchors = ir::WideScanner::generateAlternatingAnchors(source);
    auto boundaries = ir::WideScanner::scanAnchorsSIMD(source, anchors);
    
    std::cout << "Found " << boundaries.size() << " boundary events:\n";
    for (size_t i = 0; i < std::min(size_t(10), boundaries.size()); ++i) {
        const auto& b = boundaries[i];
        std::cout << "  [" << i << "] pos=" << b.position 
              << " char='" << b.delimiter 
                  << "' delim=" << b.is_delimiter 
                  << " conf=" << b.orbit_confidence << "\n";
    }
    if (boundaries.size() > 10) {
        std::cout << "  ... (" << boundaries.size() - 10 << " more)\n";
    }
    std::cout << "\n";
    
    // Step 2: Carve regions using RBCursiveRegions
    std::cout << "Step 2: Carving regions...\n";
    ir::RBCursiveRegions::CarveConfig carveConfig;
    carveConfig.enableWobbling = true;
    carveConfig.minConfidence = 0.5;
    
    ir::RBCursiveRegions carver(carveConfig);
    auto result = carver.carveRegions(boundaries, source);
    
    if (!result.success) {
        std::cout << "❌ Failed to carve regions: " << result.errorMessage << "\n";
        return;
    }
    
    std::cout << "✅ Carved " << result.regionCount << " regions:\n\n";
    
    // Step 3: Print region tree
    std::cout << "Step 3: Region tree structure:\n";
    ir::RBCursiveRegions::printCarvedRegions(*result.rootRegion, 0);
    std::cout << "\n";
    
    // Step 4: Test standalone carving function
    std::cout << "Step 4: Testing standalone carving function...\n";
    auto standaloneResult = ir::carveRegionsStandalone(source, carveConfig);
    if (standaloneResult.success) {
        std::cout << "✅ Standalone carving successful, " << standaloneResult.regionCount << " regions\n";
    } else {
        std::cout << "❌ Standalone carving failed\n";
    }
    std::cout << "\n";
}

void testJsonYamlScanners() {
    std::cout << "\n═══════════════════════════════════════════════════════════\n";
    std::cout << "TEST: JSON & YAML Orbit Scanners\n";
    std::cout << "═══════════════════════════════════════════════════════════\n";
    
    // Test JSON pattern file
    ir::JsonOrbitScanner jsonScanner("patterns/cppfort_core_patterns.json");
    if (jsonScanner.initialize()) {
        std::cout << "✅ JSON Orbit Scanner initialized\n";
        
        std::string jsonContent = loadFile("patterns/cppfort_core_patterns.json");
        if (!jsonContent.empty()) {
            auto result = jsonScanner.scanJson(jsonContent);
            std::cout << "Scanned JSON patterns, confidence: " << result.confidence << "\n";
            std::cout << "Reasoning: " << result.reasoning << "\n";
        }
    } else {
        std::cout << "❌ Failed to initialize JSON Orbit Scanner\n";
    }
    std::cout << "\n";
    
    // Test YAML pattern file
    ir::YamlOrbitScanner yamlScanner("patterns/cppfort_core_patterns.yaml");
    if (yamlScanner.initialize()) {
        std::cout << "✅ YAML Orbit Scanner initialized\n";
        
        std::string yamlContent = loadFile("patterns/cppfort_core_patterns.yaml");
        if (!yamlContent.empty()) {
            auto result = yamlScanner.scanYaml(yamlContent);
            std::cout << "Scanned YAML patterns, confidence: " << result.confidence << "\n";
            std::cout << "Reasoning: " << result.reasoning << "\n";
        }
    } else {
        std::cout << "❌ Failed to initialize YAML Orbit Scanner\n";
    }
    std::cout << "\n";
}

void testTwoWayConverter() {
    std::cout << "\n═══════════════════════════════════════════════════════════\n";
    std::cout << "TEST: JSON ↔ YAML Two-Way Converter\n";
    std::cout << "═══════════════════════════════════════════════════════════\n";
    
    std::string command = "cd /Users/jim/work/cppfort && tools/json_yaml --test";
    int result = system(command.c_str());
    
    if (result == 0) {
        std::cout << "✅ Two-way converter test passed\n";
    } else {
        std::cout << "❌ Two-way converter test failed (exit code: " << result << ")\n";
    }
    std::cout << "\n";
}

int main() {
    std::cout << "\n╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Orbit-Based Transpiler Pipeline Integration Test           ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
    
    testTwoWayConverter();
    testJsonYamlScanners();
    testStructuralCarving();
    
    std::cout << "\n╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  All Integration Tests Complete                              ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
    
    return 0;
}
