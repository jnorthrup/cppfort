#include <iostream>
#include <cassert>
#include "pijul_orbit_builder.h"

using namespace cppfort;

int main() {
    pijul::OrbitMatchCollection source;
    pijul::OrbitMatchCollection transformed;

    // Build source: key 'A' in pattern 'pat1'
    pijul::OrbitMatchInfo sInfoA;
    sInfoA.key = "A";
    sInfoA.patternName = "pat1";
    sInfoA.context.start_pos = 0;
    sInfoA.context.end_pos = 1;
    source.byKey.emplace("A", sInfoA);
    source.byPattern["pat1"].push_back("A");

    // Build transformed: key 'B' in pattern 'pat2' matching source pat1
    pijul::OrbitMatchInfo tInfoB;
    tInfoB.key = "B";
    tInfoB.patternName = "pat2";
    tInfoB.context.start_pos = 0;
    tInfoB.context.end_pos = 1;
    transformed.byKey.emplace("B", tInfoB);
    // Add mapping from transformed pattern to candidate in source
    source.byPattern["pat2"].push_back("A");

    pijul::Patch patch;
    pijul::build_orbit_patch(source, transformed, "", "", "patch-01", patch);

    // We should have at least one NewNodesChange (for 'B') and one EdgesChange from 'A'->'B'
    bool saw_new_node = false;
    bool saw_edge = false;
    for (const auto& variant : patch.changes) {
        if (auto* new_nodes = std::get_if<pijul::NewNodesChange>(&variant)) {
            for (const auto& node : new_nodes->nodes) {
                if (node.find("B") != std::string::npos) saw_new_node = true;
            }
        } else if (auto* edges = std::get_if<pijul::EdgesChange>(&variant)) {
            for (const auto& e : edges->edges) {
                if (e.from.find("A") != std::string::npos && e.to.find("B") != std::string::npos) saw_edge = true;
            }
        }
    }

    assert(saw_new_node && "New node 'B' must be present in patch changes");
    // Edges are optional depending on pattern mapping; we expect an edge here
    assert(saw_edge && "Edge A->B must be present in patch changes");

    std::cout << "test_pijul_orbit_builder: OK\n";
    return 0;
}
