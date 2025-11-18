#include <iostream>
#include <cassert>
#include "pijul_parameter_graph.h"

using namespace cppfort::pijul;

int main() {
    OrbitMatchCollection source;
    OrbitMatchCollection transformed;

    OrbitMatchInfo sInfoA;
    sInfoA.key = "A";
    sInfoA.patternName = "patx";
    sInfoA.context.start_pos = 0;
    sInfoA.context.end_pos = 1;
    source.byKey.emplace("A", sInfoA);
    source.byPattern["patx"].push_back("A");

    OrbitMatchInfo tInfoB;
    tInfoB.key = "B";
    tInfoB.patternName = "patx"; // same pattern to create mapping
    tInfoB.context.start_pos = 0;
    tInfoB.context.end_pos = 1;
    transformed.byKey.emplace("B", tInfoB);
    transformed.byPattern["patx"].push_back("B");

    ParameterGraph graph;
    std::string srcCode = "A";
    std::string transformedCode = "B";
    populate_parameter_graph(graph, source, transformed, srcCode, transformedCode);

    // Both anchors should be present
    auto aOpt = graph.find("A");
    auto bOpt = graph.find("B");
    assert(aOpt.has_value());
    assert(bOpt.has_value());
    // There should be an edge candidate->B
    bool sawEdge = false;
    for (const auto& e : graph.edges()) {
        if (e.from == "A" && e.to == "B") sawEdge = true;
    }
    assert(sawEdge && "Graph must contain edge A->B");

    std::cout << "test_pijul_parameter_graph: OK\n";
    return 0;
}
