#include <iostream>
#include <string>
#include "src/stage0/evidence_2d.h"

using namespace cppfort::stage0;

void test_template_cases() {
    std::cout << "Testing template disambiguation cases:\n\n";
    
    struct TestCase {
        std::string code;
        std::string name;
        size_t expected_angles;  // Number of angle pairs
        std::string description;
    };
    
    TestCase cases[] = {
        {"vector<int> x;", "simple_template", 1, "Basic template"},
        {"a >> b", "right_shift", 0, "Right shift operator"},
        {"map<string, int> table;", "template_with_comma", 1, "Template with comma"},
        {"vector<map<string,int>> data;", "nested_template", 2, "Nested templates with >>"},
        {"function<vector<map<int, string>>>();", "deeply_nested", 3, "Deeply nested templates"},
        {"if (x >> 2) {", "bitshift_in_condition", 0, "Bitshift in condition"},
        {"cout << data >> 8;", "mixed_stream_shift", 0, "Stream operators"},
        {"auto result = a >> b;", "simple_shift", 0, "Simple right shift"},
        {"template<typename T> class C { vector<T> v; };", "template_definition", 2, "Template definition with member"},
        {"vector<int> a, b; auto x = a >> b;", "template_then_shift", 1, "Template variables then shift"}
    };
    
    for (const auto& test : cases) {
        std::cout << "Test: " << test.name << " - " << test.description << "\n";
        std::cout << "Code: " << test.code << "\n";
        
        auto span = Evidence2DAnalyzer::analyze_span(test.code);
        auto angle_confixes = span.get_confixes_of_type(ConfixType::ANGLE);
        
        std::cout << "Found " << angle_confixes.size() << " angle confixes\n";
        std::cout << "Expected " << test.expected_angles << " angle pairs\n";
        
        // Count angle pairs (each pair is 2 confixes)
        size_t angle_pairs = angle_confixes.size() / 2;
        
        std::cout << "Result: " << angle_pairs << " angle pairs\n";
        
        if (angle_pairs == test.expected_angles) {
            std::cout << "✅ PASS\n";
        } else {
            std::cout << "❌ FAIL - Expected " << test.expected_angles 
                     << ", got " << angle_pairs << "\n";
        }
        
        std::cout << "Confidence: " << span.confidence << "\n";
        std::cout << "Balanced: " << (span.has_balanced_confixes() ? "YES" : "NO") << "\n\n";
    }
}

void test_angle_bracket_logic() {
    std::cout << "Testing angle bracket logic in detail:\n\n";
    
    // Test the >> disambiguation specifically
    std::string test_code = "vector<map<string,int>> data;";
    std::cout << "Testing: " << test_code << "\n";
    
    auto span = Evidence2DAnalyzer::analyze_span(test_code);
    
    std::cout << "All confixes found:\n";
    for (const auto& confix : span.confixes) {
        std::cout << "  Type " << static_cast<int>(confix.type) 
                 << ": [" << confix.begin_pos << ", " << confix.end_pos << ")\n";
    }
    
    auto angles = span.get_confixes_of_type(ConfixType::ANGLE);
    std::cout << "\nAngle confixes: " << angles.size() << "\n";
    for (const auto& angle : angles) {
        char begin_char = test_code[angle.begin_pos];
        char end_char = test_code[angle.end_pos - 1];
        std::cout << "  Angle: '" << begin_char << "' at " << angle.begin_pos 
                 << " -> '" << end_char << "' at " << (angle.end_pos - 1) << "\n";
    }
}

int main() {
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Template Angle Bracket Disambiguation Test                 ║\n";
    std::cout << "║  2D Evidence System: Confix Type × Position                ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n\n";
    
    test_template_cases();
    test_angle_bracket_logic();
    
    return 0;
}
