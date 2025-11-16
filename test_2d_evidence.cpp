#include <iostream>
#include <string>
#include "src/stage0/evidence.h"
#include "src/stage0/type_evidence.h"
#include "src/stage0/lattice_classes.h"

using namespace cppfort::stage0;

void test2DEvidenceSystem() {
    std::cout << "Testing 2D Evidence System within EvidenceSpan locality bubbles\n\n";
    
    // Test with a simple function signature
    std::string code = "int main(int argc, char** argv) { return 0; }";
    
    std::cout << "Code: " << code << "\n\n";
    
    // Create TypeEvidence for the entire span
    TypeEvidence evidence;
    evidence.ingest(code);
    
    std::cout << "2D Evidence Analysis:\n";
    std::cout << "Layer 1 - Character Classification:\n";
    std::cout << "  digits: " << evidence.digits << "\n";
    std::cout << "  alpha: " << evidence.alpha << "\n";
    std::cout << "  whitespaces: " << evidence.whitespaces << "\n\n";
    
    std::cout << "Layer 2 - Confix Type Tracking (2D):\n";
    std::cout << "  [confix_type][open/close] counts:\n";
    
    // Show the 2D structure: confix_type × open/close
    for (uint8_t i = 1; i < static_cast<uint8_t>(ConfixType::MAX_TYPE); ++i) {
        ConfixType type = static_cast<ConfixType>(i);
        std::string type_name;
        switch (type) {
            case ConfixType::PAREN: type_name = "PAREN"; break;
            case ConfixType::BRACE: type_name = "BRACE"; break;
            case ConfixType::BRACKET: type_name = "BRACKET"; break;
            case ConfixType::ANGLE: type_name = "ANGLE"; break;
            default: type_name = "UNKNOWN"; break;
        }
        
        std::cout << "    " << type_name << ": open=" << evidence.confix_open[i] 
                  << " close=" << evidence.confix_close[i]
                  << " max_depth=" << evidence.max_confix_depth[i]
                  << " min_depth=" << evidence.min_confix_depth[i] << "\n";
    }
    
    std::cout << "\nLayer 3+ - Other classifications:\n";
    std::cout << "  c_keywords: " << evidence.c_keywords << "\n";
    std::cout << "  cpp_keywords: " << evidence.cpp_keywords << "\n";
    std::cout << "  identifiers: " << evidence.c_identifiers << "\n";
    
    // Test balanced confix check
    std::cout << "\nEvidenceSpan locality validation:\n";
    std::cout << "  Balanced confixes: " << (has_balanced_confixes(evidence) ? "YES" : "NO") << "\n";
    
    auto depths = get_confix_depths(evidence);
    std::cout << "  Confix depths: ";
    for (size_t i = 1; i < 5; ++i) {
        std::cout << "[" << i << "]=" << depths[i] << " ";
    }
    std::cout << "\n\n";
}

void testLatticeToEvidenceMapping() {
    std::cout << "Testing lattice classification → 2D evidence mapping\n\n";
    
    std::string test_chars = "{}()[]<>;:+-*/\"\"";
    
    std::cout << "Character classification and evidence mapping:\n";
    for (char c : test_chars) {
        uint16_t mask = classify_byte(c);
        
        std::cout << "Char '" << c << "':\n";
        
        // Show lattice classification
        if (has_class(mask, CharClass::Structural)) std::cout << "  → Structural ";
        if (has_class(mask, CharClass::Operator)) std::cout << "→ Operator ";
        if (has_class(mask, CharClass::Punct)) std::cout << "→ Punct ";
        if (has_class(mask, CharClass::AnchorCandidate)) std::cout << "→ Anchor ";
        std::cout << "\n";
        
        // Show 2D evidence mapping
        ConfixType confix_type = get_confix_type(c);
        if (confix_type != ConfixType::INVALID) {
            std::cout << "  → Maps to ConfixType[" << static_cast<int>(confix_type) << "]\n";
        }
        
        // Show evidence update
        TypeEvidence evidence;
        update_type_evidence_from_lattice(c, evidence);
        
        if (is_structural(c)) {
            uint8_t type_idx = static_cast<uint8_t>(get_confix_type(c));
            std::cout << "  → Evidence: confix_open[" << (int)type_idx << "]=" 
                      << evidence.confix_open[type_idx] << "\n";
        }
        std::cout << "\n";
    }
}

void testEvidenceSpanLocality() {
    std::cout << "Testing EvidenceSpan locality with 2D evidence\n\n";
    
    // Simulate analyzing a region within an EvidenceSpan
    std::string region_content = "vector<map<string, int>> data;";
    
    EvidenceSpan span;
    span.start_pos = 0;
    span.end_pos = region_content.length();
    span.content = region_content;
    
    std::cout << "Region in EvidenceSpan: \"" << region_content << "\"\n";
    std::cout << "Span locality: [" << span.start_pos << ", " << span.end_pos << "]\n\n";
    
    // Build evidence for this locality bubble
    TypeEvidence evidence;
    evidence.ingest(region_content);
    
    std::cout << "2D evidence within this locality:\n";
    
    // Show template angle bracket complexity (the challenging case)
    uint8_t angle_idx = static_cast<uint8_t>(ConfixType::ANGLE);
    std::cout << "  ANGLE confix: open=" << evidence.confix_open[angle_idx] 
              << " close=" << evidence.confix_close[angle_idx]
              << " max_depth=" << evidence.max_confix_depth[angle_idx] << "\n";
    
    // Show how this helps with the >> ambiguity
    std::cout << "  >> disambiguation: max_depth=" << evidence.max_confix_depth[angle_idx] 
              << " suggests " << (evidence.max_confix_depth[angle_idx] >= 2 ? "template closes" : "right shift") << "\n";
    
    std::cout << "\nEvidenceSpan validation:\n";
    std::cout << "  Pattern would be " << (has_balanced_confixes(evidence) ? "VALID" : "INVALID") 
              << " within this locality\n";
}

int main() {
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  2D Evidence System: Confix Type × Open/Close/Depth        ║\n";
    std::cout << "║  Operating within EvidenceSpan Locality Bubbles            ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n\n";
    
    test2DEvidenceSystem();
    testLatticeToEvidenceMapping();
    testEvidenceSpanLocality();
    
    return 0;
}
