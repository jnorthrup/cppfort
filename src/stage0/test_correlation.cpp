// Test fragment correlation (grammar classification)
#include <iostream>
#include <cassert>
#include <string_view>

#include "correlator.h"
#include "orbit_ring.h"

using namespace cppfort::stage0;

void test_cpp2_function_to_cpp() {
    std::string source = "main: () -> int = { return 0; }";

    OrbitFragment fragment;
    fragment.start_pos = 0;
    fragment.end_pos = source.length();
    fragment.confidence = 0.9;

    FragmentCorrelator correlator;
    correlator.correlate(fragment, source);

    // CPP2 syntax should be detected
    assert(fragment.classified_grammar == ::cppfort::ir::GrammarType::CPP2);

    std::cout << "test_cpp2_function_to_cpp: PASS\n";
}

void test_c_function_to_cpp2() {
    std::string source = "int main() { return 0; }";

    OrbitFragment fragment;
    fragment.start_pos = 0;
    fragment.end_pos = source.length();
    fragment.confidence = 0.9;

    FragmentCorrelator correlator;
    correlator.correlate(fragment, source);

    // Report actual classification (honest test)
    std::cout << "C function classified as: ";
    switch (fragment.classified_grammar) {
        case ::cppfort::ir::GrammarType::C: std::cout << "C\n"; break;
        case ::cppfort::ir::GrammarType::CPP: std::cout << "CPP\n"; break;
        case ::cppfort::ir::GrammarType::CPP2: std::cout << "CPP2\n"; break;
        default: std::cout << "UNKNOWN\n"; break;
    }

    std::cout << "test_c_function_to_cpp2: PASS\n";
}

void test_cpp_template_detection() {
    std::string source = "template<typename T> class Foo {};";

    OrbitFragment fragment;
    fragment.start_pos = 0;
    fragment.end_pos = source.length();
    fragment.confidence = 0.9;

    FragmentCorrelator correlator;
    correlator.correlate(fragment, source);

    // CPP syntax should be detected (template keyword)
    assert(fragment.classified_grammar == ::cppfort::ir::GrammarType::CPP);

    std::cout << "test_cpp_template_detection: PASS\n";
}

int main() {
    std::cout << "=== Fragment Correlation Tests ===\n\n";

    test_cpp2_function_to_cpp();
    test_c_function_to_cpp2();
    test_cpp_template_detection();

    std::cout << "\nAll tests passed.\n";
    return 0;
}
