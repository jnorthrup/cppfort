#include <iostream>
#include "src/stage0/lattice_classes.h"
#include "src/stage0/evidence.h"

using namespace cppfort::stage0;

void testLatticeIntegration() {
    std::cout << "Testing lattice class to TypeEvidence integration\n\n";
    
    // Test character classification
    const char* test_string = "int x = 42;";
    
    for (int i = 0; test_string[i]; i++) {
        char c = test_string[i];
        uint16_t mask = classify_byte(c);
        
        std::cout << "Char '" << c << ":\n";
        
        // Show which classes it belongs to
        if (has_class(mask, CharClass::Alpha)) std::cout << "  Alpha ";
        if (has_class(mask, CharClass::Digit)) std::cout << "Digit ";
        if (has_class(mask, CharClass::Whitespace)) std::cout << "Whitespace ";
        if (has_class(mask, CharClass::Operator)) std::cout << "Operator ";
        if (has_class(mask, CharClass::Structural)) std::cout << "Structural ";
        if (has_class(mask, CharClass::IdentifierStart)) std::cout << "IdentifierStart ";
        if (has_class(mask, CharClass::AnchorCandidate)) std::cout << "AnchorCandidate ";
        std::cout << "\n";
        
        // Show how this feeds into TypeEvidence
        if (is_identifier_start(c)) {
            std::cout << "  → Starts identifier\n";
        }
        if (is_numeric_literal_char(c)) {
            std::cout << "  → Part of numeric literal\n";
        }
        if (is_anchor_candidate(c)) {
            std::cout << "  → Pattern anchor point\n";
        }
        if (is_structural(c)) {
            std::cout << "  → Structural boundary (affects confix depth)\n";
        }
    }
    
    std::cout << "\n";
}

void testHeuristicTile() {
    std::cout << "Testing HeuristicTile integration\n\n";
    
    std::string code = "int main() { return 42; }";
    
    // Create a tile
    HeuristicTile tile(0, std::span<const char>(code.data(), code.size()));
    tile.analyze_tile();
    
    std::cout << "Analyzed tile with " << code.size() << " bytes\n";
    std::cout << "Lattice mask: " << tile.lattice_mask << "\n";
    std::cout << "Orbit confidence: " << tile.orbit_confidence << "\n";
    
    std::cout << "Class counts:\n";
    if (tile.class_counters[0]) std::cout << "  Whitespace: " << tile.class_counters[0] << "\n";
    if (tile.class_counters[1]) std::cout << "  Digit: " << tile.class_counters[1] << "\n";
    if (tile.class_counters[2]) std::cout << "  Alpha: " << tile.class_counters[2] << "\n";
    if (tile.class_counters[3]) std::cout << "  Punct: " << tile.class_counters[3] << "\n";
    if (tile.class_counters[4]) std::cout << "  Operator: " << tile.class_counters[4] << "\n";
    if (tile.class_counters[5]) std::cout << "  Structural: " << tile.class_counters[5] << "\n";
    
    // These counts feed directly into TypeEvidence counters
    std::cout << "\nThese counts can populate TypeEvidence::digits, TypeEvidence::alpha, etc.\n";
}

int main() {
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Lattice Classes → TypeEvidence Integration Test            ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n\n";
    
    testLatticeIntegration();
    testHeuristicTile();
    
    return 0;
}
