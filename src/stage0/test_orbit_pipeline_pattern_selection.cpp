#include "orbit_pipeline.h"
#include <cassert>
#include <iostream>

int main() {
    using namespace cppfort::stage0;

    OrbitPipeline pipeline;
    // Load our test patterns
    bool ok = pipeline.load_patterns("./src/stage0/test-patterns/function_test.yaml");
    assert(ok);

    std::string source = "prefix fn foo { int x; } suffix";
    // Create a single fragment spanning the function
    OrbitFragment fragment;
    fragment.start_pos = 7; // 'f' in 'fn'
    fragment.end_pos = 7 + 17; // to the closing brace
    fragment.confidence = 1.0;

    OrbitIterator iterator;
    std::vector<OrbitFragment> fragments{fragment};
    pipeline.populate_iterator(fragments, iterator, source);

    auto orbits = pipeline.orbits();
    assert(orbits.size() == 1);
    // The selected pattern should be function_test or be non-empty
    assert(!orbits[0]->selected_pattern().empty());

    std::cout << "test_orbit_pipeline_pattern_selection passed. Selected=" << orbits[0]->selected_pattern() << std::endl;
    return 0;
}
