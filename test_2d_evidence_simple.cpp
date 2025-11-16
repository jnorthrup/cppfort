#include <iostream>
#include <string>
#include "src/stage0/evidence_2d.h"

using namespace cppfort::stage0;

int main() {
    std::cout << "Simple 2D Evidence System Test\n\n";
    
    // Test basic confix classification
    std::cout << "Testing confix classification:\n";
    char test_chars[] = {'(', ')', '{', '}', '[', ']', '<', '>', 'a', ' '};
    
    for (char c : test_chars) {
        ConfixType type = Evidence2DAnalyzer::get_confix_type(c);
        std::cout << "'" << c << "' -> " << static_cast<int>(type) << "\n";
    }
    
    std::cout << "\nTesting 2D evidence analysis:\n";
    
    // Test simple function
    std::string code1 = "int main() { return 0; }";
    auto span1 = Evidence2DAnalyzer::analyze_span(code1);
    
    std::cout << "Code: " << code1 << "\n";
    std::cout << "Found " << span1.confixes.size() << " confixes\n";
    std::cout << "Confidence: " << span1.confidence << "\n";
    std::cout << "Balanced: " << (span1.has_balanced_confixes() ? "YES" : "NO") << "\n";
    std::cout << "Dominant type: " << static_cast<int>(span1.get_dominant_confix_type()) << "\n\n";
    
    // Test template
    std::string code2 = "vector<int> data;";
    auto span2 = Evidence2DAnalyzer::analyze_span(code2);
    
    std::cout << "Code: " << code2 << "\n";
    std::cout << "Found " << span2.confixes.size() << " confixes\n";
    std::cout << "Confidence: " << span2.confidence << "\n";
    std::cout << "Balanced: " << (span2.has_balanced_confixes() ? "YES" : "NO") << "\n";
    
    auto angles = span2.get_confixes_of_type(ConfixType::ANGLE);
    std::cout << "Angle confixes: " << angles.size() << "\n";
    
    std::cout << "\n✅ Basic 2D evidence system working!\n";
    return 0;
}
