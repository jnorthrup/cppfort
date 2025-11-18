#include "orbit_pipeline.h"
#include <cassert>
#include "debug_helpers.h"
#include <iostream>

int main() {
    cppfort::stage0::debug::install_signal_handlers();
    cppfort::stage0::debug::start_watchdog_from_env();
    using namespace cppfort::stage0;

    OrbitPipeline pipeline;
    // Load our test patterns
    std::string pattern_path = "/Users/jim/work/cppfort/src/stage0/test-patterns/function_test.yaml";
    std::cout << "Loading patterns from: " << pattern_path << std::endl;

    bool ok = pipeline.load_patterns(pattern_path);
    if (!ok) {
        std::cout << "Failed to load patterns from " << pattern_path << std::endl;
        std::cout << "Current working directory: ";
        system("pwd < /dev/null");
        std::cout << "File exists check: ";
        system("ls -la /Users/jim/work/cppfort/src/stage0/test-patterns/function_test.yaml < /dev/null");
        return 1;
    }

    std::cout << "Successfully loaded " << pipeline.pattern_count() << " patterns" << std::endl;

    std::string source = "prefix fn foo { int x; } suffix";
    // Create a single fragment spanning the function
    OrbitFragment fragment;
    fragment.start_pos = 7; // 'f' in 'fn'
    fragment.end_pos = 7 + 17; // to the closing brace
    fragment.confidence = 1.0;

    OrbitIterator iterator;
    std::vector<OrbitFragment> fragments{fragment};
    pipeline.populate_iterator(fragments, iterator, source);

    const auto& orbits = pipeline.orbits();
    if (orbits.size() != 1) {
        std::cout << "Expected 1 orbit, got " << orbits.size() << std::endl;
        return 1;
    }
    // The selected pattern should be function_test or be non-empty
    if (orbits[0]->selected_pattern().empty()) {
        std::cout << "Selected pattern is empty" << std::endl;
        return 1;
    }

    std::cout << "test_orbit_pipeline_pattern_selection passed. Selected=" << orbits[0]->selected_pattern() << std::endl;
    return 0;
}
