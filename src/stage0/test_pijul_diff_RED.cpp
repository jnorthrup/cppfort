#include "pijul_diff.h"
#include "heap_limiter.h"

#include <iostream>
#include <string>
#include <variant>

using namespace cppfort::pijul;

namespace {

std::string to_string(const std::vector<std::uint8_t>& bytes) {
    return std::string(bytes.begin(), bytes.end());
}

} // namespace

int main() {
    if (!cppfort::stage0::ensure_heap_limit(std::cerr)) {
        return 1;
    }

    const std::string source = "alpha\nbeta\n";
    const std::string target = "alpha\nbeta\ngamma\n";

    Patch patch = compute_line_patch("unit-test", source, target);

    bool ok = true;
    if (patch.name != "unit-test") {
        std::cerr << "Patch name mismatch\n";
        ok = false;
    }

    if (patch.changes.size() != 2) {
        std::cerr << "Expected 2 changes (new nodes + edges), got " << patch.changes.size() << "\n";
        ok = false;
    }

    const auto* new_nodes = std::get_if<NewNodesChange>(&patch.changes[0]);
    if (!new_nodes) {
        std::cerr << "First change should be NewNodes\n";
        ok = false;
    } else {
        if (new_nodes->nodes.size() != 1) {
            std::cerr << "Expected exactly one new node\n";
            ok = false;
        } else if (to_string(new_nodes->nodes[0]) != "gamma\n") {
            std::cerr << "New node contents mismatch: " << to_string(new_nodes->nodes[0]) << "\n";
            ok = false;
        }
        if (new_nodes->up_context.empty()) {
            std::cerr << "New node should have an up_context anchor\n";
            ok = false;
        }
        if (new_nodes->down_context.empty()) {
            std::cerr << "New node should have a down_context anchor (even if root)\n";
            ok = false;
        }
    }

    const auto* edges = (patch.changes.size() > 1) ? std::get_if<EdgesChange>(&patch.changes[1]) : nullptr;
    if (!edges) {
        std::cerr << "Second change should be Edges\n";
        ok = false;
    } else {
        if (edges->edges.size() != 2) {
            std::cerr << "Expected two edges tying the insertion into the graph\n";
            ok = false;
        }
        for (const auto& edge : edges->edges) {
            if (edge.from.empty()) {
                std::cerr << "Edge missing source key\n";
                ok = false;
            }
            if (edge.to.empty()) {
                std::cerr << "Edge missing destination key\n";
                ok = false;
            }
        }
    }

    if (patch.dependencies.empty()) {
        std::cerr << "Patch should advertise at least one dependency\n";
        ok = false;
    }

    return ok ? 0 : 1;
}
