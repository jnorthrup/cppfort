#include "orbit_pipeline.h"
#include <iostream>
#include "debug_helpers.h"

int main() {
    cppfort::stage0::debug::install_signal_handlers();
    cppfort::stage0::debug::start_watchdog_from_env();
    using namespace cppfort::stage0;
    OrbitPipeline pipeline;
    OrbitFragment fragment;
    fragment.start_pos = 7; // in sample text used below
    fragment.end_pos = 24;
    fragment.confidence = 1.0;
    std::string source = "prefix fn foo { int x; } suffix";

    OrbitIterator iterator;
    std::vector<OrbitFragment> fragments{fragment};
    pipeline.populate_iterator(fragments, iterator, source);
    const auto& orbits = pipeline.orbits();
    if (orbits.empty()) {
        std::cerr << "Failed: no orbits produced by populate_iterator\n";
        return 1;
    }
    if (!orbits[0]->graph_node()) {
        std::cerr << "Failed: populate_iterator did not set graph_node for orbit\n";
        return 1;
    }
    std::cout << "OK: populate_iterator created and stored graph_node for orbit\n";
    return 0;
}
