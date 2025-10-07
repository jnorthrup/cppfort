#include "src/stage0/orbit_pipeline.h"
#include "src/stage0/wide_scanner.h"
#include "src/stage0/confix_orbit.h"
#include <iostream>

int main() {
    std::string source = "template<typename T> T add(T a, T b) { return a + b; }";
    
    cppfort::ir::WideScanner scanner;
    auto anchors = cppfort::ir::WideScanner::generateAlternatingAnchors(source);
    scanner.scanAnchorsWithOrbits(source, anchors);
    
    cppfort::stage0::OrbitPipeline pipeline;
    pipeline.load_patterns("tests/patterns/minimal.yaml");
    
    cppfort::stage0::OrbitIterator iterator(anchors.size());
    pipeline.populate_iterator(scanner.fragments(), iterator, source);
    
    int count = 0;
    for (auto* orbit = iterator.next(); orbit && count < 3; orbit = iterator.next(), count++) {
        std::cout << "\n=== Orbit " << count << " ===\n";
        std::cout << "Evidence count: " << orbit->evidence_count() << "\n";
        for (size_t i = 0; i < orbit->evidence_count(); ++i) {
            auto* ev = orbit->get_evidence(i);
            if (ev) {
                std::cout << "Evidence[" << i << "]: '" << ev->content.substr(0, 40) << "...'\n";
            }
        }
    }
    return 0;
}
