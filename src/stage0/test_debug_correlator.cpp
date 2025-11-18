#include <iostream>
#include "correlator.h"
#include "orbit_ring.h"

using namespace cppfort::stage0;

int main() {
    std::string source = "main: () -> int = { return 0; }";

    OrbitFragment fragment;
    fragment.start_pos = 0;
    fragment.end_pos = source.length();
    fragment.confidence = 0.9;

    FragmentCorrelator correlator;
    // Check classify_text separately
    auto ctext = correlator.classify_text(source);
    std::cout << "classify_text result: " << static_cast<int>(ctext) << "\n";
    // don't call correlate to avoid side effects for this quick debug

    std::cout << "Classification: " << static_cast<int>(fragment.classified_grammar) << " confidence=" << fragment.confidence << "\n";

    return 0;
}
