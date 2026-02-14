#include "../include/mlir_cpp2_dialect.hpp"

using namespace cppfort::mlir_son;

bool CRDTGraph::apply_patch(const Patch& patch) {
    switch (patch.operation) {
        case Patch::Op::AddNode: {
            const Node& n = std::get<Node>(patch.data);
            auto it = nodes.find(n.id);
            if (it == nodes.end()) {
                nodes[n.id] = n;
                // ensure there's an outputs entry only when edges are added
                return true;
            } else {
                if (n.timestamp > it->second.timestamp) {
                    // Preserve existing edges and inputs
                    auto existing_edges = edges[n.id];
                    auto existing_inputs = it->second.inputs;
                    nodes[n.id] = n;
                    nodes[n.id].inputs = std::move(existing_inputs);
                    if (!existing_edges.empty()) edges[n.id] = std::move(existing_edges);
                    return true;
                } else {
                    return false; // Older or equal timestamp; reject
                }
            }
        }
        case Patch::Op::AddEdge: {
            auto [from, to] = std::get<std::pair<NodeID, NodeID>>(patch.data);
            // Ensure nodes exist (tests add nodes before edges, but be defensive)
            if (!nodes.contains(from) || !nodes.contains(to)) return false;
            edges[from].insert(to);
            auto& inputs = nodes[to].inputs;
            if (std::find(inputs.begin(), inputs.end(), from) == inputs.end()) inputs.push_back(from);
            return true;
        }
        case Patch::Op::RemoveEdge: {
            auto [from, to] = std::get<std::pair<NodeID, NodeID>>(patch.data);
            auto it = edges.find(from);
            if (it == edges.end() || !it->second.contains(to)) return false;
            it->second.erase(to);
            if (it->second.empty()) edges.erase(it);
            auto& inputs = nodes[to].inputs;
            inputs.erase(std::remove(inputs.begin(), inputs.end(), from), inputs.end());
            return true;
        }
        default:
            return false;
    }
}

void CRDTGraph::merge(const CRDTGraph& other) {
    // Merge nodes with Last-Writer-Wins
    for (const auto& [id, node] : other.nodes) {
        auto it = nodes.find(id);
        if (it == nodes.end()) {
            nodes[id] = node;
        } else {
            if (node.timestamp > it->second.timestamp) {
                // Preserve existing edges for id
                auto existing_edges = edges[id];
                nodes[id] = node;
                if (!existing_edges.empty()) edges[id] = std::move(existing_edges);
            }
        }
    }

    // Merge edges
    for (const auto& [from, outs] : other.edges) {
        for (auto to : outs) {
            edges[from].insert(to);
            auto& inputs = nodes[to].inputs;
            if (std::find(inputs.begin(), inputs.end(), from) == inputs.end()) inputs.push_back(from);
        }
    }
}
