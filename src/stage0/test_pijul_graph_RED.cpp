#include "pijul_graph.h"
#include "heap_limiter.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <vector>

using namespace cppfort::pijul;

namespace {

ExternalKey make_key(std::uint8_t fill) {
    ExternalKey key(HASH_SIZE, fill);
    key.push_back(0);
    key.push_back(0);
    key.push_back(0);
    key.push_back(1);
    return key;
}

} // namespace

int main() {
    if (!cppfort::stage0::ensure_heap_limit(std::cerr)) {
        return 1;
    }

    Graph graph;
    auto node_a = graph.ensure_node(make_key(0xA1));
    auto node_b = graph.ensure_node(make_key(0xB2));
    auto node_c = graph.ensure_node(make_key(0xC3));

    graph.add_edge(node_a, node_b, EdgeFlag::Parent);
    graph.add_edge(node_a, node_c, EdgeFlag::Parent);
    graph.add_edge(node_b, node_c, EdgeFlag::Parent);

    bool ok = true;
    if (graph.fanout(node_a) != 2 || graph.fanout(node_b) != 1 || graph.fanout(node_c) != 0) {
        std::cerr << "Unexpected fanout distribution\n";
        ok = false;
    }
    if (graph.fanin(node_a) != 0 || graph.fanin(node_b) != 1 || graph.fanin(node_c) != 2) {
        std::cerr << "Unexpected fanin distribution\n";
        ok = false;
    }

    auto topo = graph.topological_order();
    if (topo.size() != 3 || topo[0] != node_a || topo[1] != node_b || topo[2] != node_c) {
        std::cerr << "Topological order incorrect\n";
        ok = false;
    }

    graph.add_edge(node_c, node_a, EdgeFlag::Parent);
    if (!graph.has_cycle()) {
        std::cerr << "Cycle should have been detected\n";
        ok = false;
    }

    auto contexts = graph.dependent_keys(node_c);
    if (contexts.size() != 2) {
        std::cerr << "Expected two inbound dependencies for node C\n";
        ok = false;
    }

    return ok ? 0 : 1;
}
