#include "orbit_pipeline.h"
#include <iostream>
#include "debug_helpers.h"

int main() {
    cppfort::stage0::debug::install_signal_handlers();
    cppfort::stage0::debug::start_watchdog_from_env();
    using namespace cppfort::stage0;
    OrbitPipeline pipeline;
    OrbitFragment fragment;
    fragment.start_pos = 0;
    fragment.end_pos = 10;
    std::string source = "int main() {}";

    // For ownership of graph_node we want to verify that a newly constructed
    // ConfixOrbit does not contain an attached graph node (ownership should
    // be caller-managed by default). We directly construct a ConfixOrbit to
    // avoid calling private pipeline internals.
    auto base_orbit = std::make_unique<ConfixOrbit>('{', '}');
    base_orbit->start_pos = fragment.start_pos;
    base_orbit->end_pos = fragment.end_pos;
    if (!base_orbit) {
        std::cerr << "BUG: make_base_orbit returned null\n";
        return 1;
    }
    if (base_orbit->graph_node() != nullptr) {
        std::cerr << "BUG: make_base_orbit returned orbit with non-null graph_node (ownership should be caller-managed)\n";
        return 1;
    }
    std::cout << "OK: make_base_orbit returned orbit with null graph_node (ownership is caller responsibility)\n";
    return 0;
}
