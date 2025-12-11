#include "mlir_cpp2_dialect.hpp"
#include <algorithm>
#include <queue>
#include <sstream>

namespace cppfort::mlir_son {

// CRDT Graph implementation
bool CRDTGraph::apply_patch(const Patch& patch) {
    pending_patches.push_back(patch);

    switch (patch.operation) {
        case Patch::Op::AddNode: {
            auto& node = std::get<Node>(patch.data);
            auto it = nodes.find(node.id);
            if (it == nodes.end() || it->second.timestamp < node.timestamp) {
                nodes[node.id] = node;
                return true;
            }
            return false;
        }

        case Patch::Op::RemoveNode: {
            auto it = nodes.find(patch.target);
            if (it != nodes.end() && it->second.timestamp <= patch.target) {
                nodes.erase(it);
                edges.erase(patch.target);
                return true;
            }
            return false;
        }

        case Patch::Op::AddEdge: {
            auto [from, to] = std::get<std::pair<NodeID, NodeID>>(patch.data);
            edges[from].insert(to);

            // Update node outputs
            if (nodes.contains(from)) {
                auto& from_node = nodes[from];
                if (std::find(from_node.outputs.begin(), from_node.outputs.end(), to) == from_node.outputs.end()) {
                    from_node.outputs.push_back(to);
                }
            }

            // Update node inputs
            if (nodes.contains(to)) {
                auto& to_node = nodes[to];
                if (std::find(to_node.inputs.begin(), to_node.inputs.end(), from) == to_node.inputs.end()) {
                    to_node.inputs.push_back(from);
                }
            }
            return true;
        }

        case Patch::Op::RemoveEdge: {
            auto [from, to] = std::get<std::pair<NodeID, NodeID>>(patch.data);
            edges[from].erase(to);

            // Update node outputs
            if (nodes.contains(from)) {
                auto& from_node = nodes[from];
                from_node.outputs.erase(
                    std::remove(from_node.outputs.begin(), from_node.outputs.end(), to),
                    from_node.outputs.end()
                );
            }

            // Update node inputs
            if (nodes.contains(to)) {
                auto& to_node = nodes[to];
                to_node.inputs.erase(
                    std::remove(to_node.inputs.begin(), to_node.inputs.end(), from),
                    to_node.inputs.end()
                );
            }
            return true;
        }
    }
    return false;
}

void CRDTGraph::merge(const CRDTGraph& other) {
    // Merge nodes with LWW (Last-Writer-Wins) resolution
    for (const auto& [id, other_node] : other.nodes) {
        auto it = nodes.find(id);
        if (it == nodes.end() || it->second.timestamp < other_node.timestamp) {
            nodes[id] = other_node;
        }
    }

    // Union of edges
    for (const auto& [from, other_outs] : other.edges) {
        edges[from].insert(other_outs.begin(), other_outs.end());

        // Ensure node outputs are updated
        if (nodes.contains(from)) {
            auto& node = nodes[from];
            for (NodeID out : other_outs) {
                if (std::find(node.outputs.begin(), node.outputs.end(), out) == node.outputs.end()) {
                    node.outputs.push_back(out);
                }
            }
        }
    }
}

// Scheduler implementation
void Scheduler::schedule_early() {
    std::unordered_set<NodeID> visited;

    // Start from Stop node and work backwards
    for (const auto& [id, node] : graph.get_nodes()) {
        if (node.kind == Node::Kind::Stop) {
            schedule_early_dfs(id, visited);
        }
    }
}

void Scheduler::schedule_early_dfs(NodeID id, std::unordered_set<NodeID>& visited) {
    if (visited.contains(id)) return;
    visited.insert(id);

    const Node* node = graph.get_node(id);
    if (!node) return;

    // Visit all inputs first
    for (NodeID input_id : node->inputs) {
        schedule_early_dfs(input_id, visited);
    }

    // Only schedule floating data nodes
    if (is_floating_data_node(*node)) {
        // Find earliest block where all inputs are available
        NodeID earliest_block = find_earliest_dominator(*node);

        // Add control edge to schedule
        if (earliest_block != 0) {
            Patch control_patch;
            control_patch.target = id;
            control_patch.operation = Patch::Op::AddEdge;
            control_patch.data = std::make_pair(earliest_block, id);
            const_cast<CRDTGraph&>(graph).apply_patch(control_patch);
        }
    }
}

void Scheduler::schedule_late() {
    std::unordered_set<NodeID> visited;

    // Start from Start node and work forwards
    for (const auto& [id, node] : graph.get_nodes()) {
        if (node.kind == Node::Kind::Start) {
            schedule_late_dfs(id, visited);
        }
    }
}

void Scheduler::schedule_late_dfs(NodeID id, std::unordered_set<NodeID>& visited) {
    if (visited.contains(id)) return;
    visited.insert(id);

    const Node* node = graph.get_node(id);
    if (!node) return;

    // Visit all outputs
    const auto* outputs = graph.get_outputs(id);
    if (outputs) {
        for (NodeID output_id : *outputs) {
            schedule_late_dfs(output_id, visited);
        }
    }

    // Move floating nodes to latest valid position
    if (is_floating_data_node(*node)) {
        NodeID latest_block = find_latest_valid_position(*node);

        if (latest_block != 0) {
            // Move control edge to new position
            // (Implementation would remove old edge and add new one)
        }
    }
}

void Scheduler::insert_anti_dependencies() {
    // For each load, find stores to same alias class and add anti-deps
    std::unordered_map<NodeID, std::vector<NodeID>> loads_by_alias;

    // Collect all loads with their alias classes
    for (const auto& [id, node] : graph.get_nodes()) {
        if (node.kind == Node::Kind::Load) {
            AliasClass alias = get_alias_class(node);
            if (alias.id != 0) {
                loads_by_alias[alias.id].push_back(id);
            }
        }
    }

    // Find stores and create anti-dependencies
    for (const auto& [alias_id, loads] : loads_by_alias) {
        std::vector<NodeID> stores = find_stores_for_alias(alias_id);

        for (NodeID load_id : loads) {
            for (NodeID store_id : stores) {
                if (should_add_anti_dependency(load_id, store_id)) {
                    add_anti_dependency(load_id, store_id);
                }
            }
        }
    }
}

bool Scheduler::is_floating_data_node(const Node& node) {
    // Control nodes are fixed
    if (node.kind == Node::Kind::Start || node.kind == Node::Kind::Stop ||
        node.kind == Node::Kind::If || node.kind == Node::Kind::Region ||
        node.kind == Node::Kind::Loop || node.kind == Node::Kind::Return) {
        return false;
    }

    // Nodes without control input are floating
    for (NodeID input_id : node.inputs) {
        const Node* input = graph.get_node(input_id);
        if (input && input->kind == Node::Kind::Start) {
            return false; // Already has control
        }
    }

    return true;
}

NodeID Scheduler::find_earliest_dominator(const Node& node) {
    // Find first block where all inputs dominate this node
    std::unordered_set<NodeID> common_doms;

    bool first = true;
    for (NodeID input_id : node.inputs) {
        const Node* input = graph.get_node(input_id);
        if (!input) continue;

        std::unordered_set<NodeID> input_doms = find_dominators(input_id);

        if (first) {
            common_doms = input_doms;
            first = false;
        } else {
            // Intersection
            std::unordered_set<NodeID> intersection;
            for (NodeID dom : common_doms) {
                if (input_doms.contains(dom)) {
                    intersection.insert(dom);
                }
            }
            common_doms = std::move(intersection);
        }
    }

    // Return the deepest (most specific) common dominator
    NodeID deepest = 0;
    int max_depth = -1;
    for (NodeID dom : common_doms) {
        int depth = get_block_depth(dom);
        if (depth > max_depth) {
            max_depth = depth;
            deepest = dom;
        }
    }

    return deepest;
}

NodeID Scheduler::find_latest_valid_position(const Node& node) {
    // Find shallowest block that still dominates all uses
    NodeID latest = find_earliest_dominator(node);

    // Consider all uses and move towards them if possible
    const auto* outputs = graph.get_outputs(node.id);
    if (outputs) {
        for (NodeID use_id : *outputs) {
            NodeID use_block = find_block_for_node(use_id);
            if (use_block != 0 && dominates(latest, use_block)) {
                // Can move towards the use
                latest = move_towards_block(latest, use_block);
            }
        }
    }

    return latest;
}

// Utility methods
std::unordered_set<NodeID> Scheduler::find_dominators(NodeID node_id) {
    // Compute dominators restricted to control nodes (blocks)
    std::unordered_set<NodeID> control_nodes;
    for (const auto& [id, node] : graph.get_nodes()) {
        switch (node.kind) {
            case Node::Kind::Start:
            case Node::Kind::Stop:
            case Node::Kind::If:
            case Node::Kind::Region:
            case Node::Kind::Loop:
            case Node::Kind::Return:
                control_nodes.insert(id);
                break;
            default:
                break;
        }
    }

    // Build predecessor map among control nodes
    std::unordered_map<NodeID, std::unordered_set<NodeID>> preds;
    for (NodeID b : control_nodes) {
        auto pset = graph.get_predecessors(b);
        for (NodeID p : pset) {
            const Node* pn = graph.get_node(p);
            if (!pn) continue;
            if (control_nodes.contains(p)) preds[b].insert(p);
        }
    }

    // Find start nodes
    std::vector<NodeID> starts;
    for (const auto& [id, node] : graph.get_nodes()) {
        if (node.kind == Node::Kind::Start) starts.push_back(id);
    }

    // Initialize dominator sets
    std::unordered_map<NodeID, std::unordered_set<NodeID>> dom;
    for (NodeID n : control_nodes) {
        if (std::find(starts.begin(), starts.end(), n) != starts.end()) {
            dom[n] = {n};
        } else {
            dom[n] = control_nodes; // start with all control nodes
        }
    }

    // Iteratively refine
    bool changed = true;
    while (changed) {
        changed = false;
        for (NodeID n : control_nodes) {
            if (std::find(starts.begin(), starts.end(), n) != starts.end()) continue;

            // Intersection of dominators of predecessors
            std::unordered_set<NodeID> newDom;
            bool first = true;
            for (NodeID p : preds[n]) {
                if (first) {
                    newDom = dom[p];
                    first = false;
                } else {
                    std::unordered_set<NodeID> inter;
                    for (NodeID d : newDom) if (dom[p].contains(d)) inter.insert(d);
                    newDom = std::move(inter);
                }
            }
            // If no preds, dominator is only the node itself
            if (first) newDom = {};

            newDom.insert(n);

            if (newDom != dom[n]) {
                dom[n] = newDom;
                changed = true;
            }
        }
    }

    // If node_id is a control node return its doms, otherwise find its block
    const Node* node = graph.get_node(node_id);
    if (node && control_nodes.contains(node_id)) {
        return dom[node_id];
    }
    NodeID block = find_block_for_node(node_id);
    if (block != 0) return dom[block];
    // If no control block found, return empty set
    return {};
}

bool Scheduler::dominates(NodeID a, NodeID b) {
    // Check if block A dominates block B
    auto doms = find_dominators(b);
    return doms.contains(a);
}

NodeID Scheduler::find_block_for_node(NodeID node_id) {
    // BFS backward along inputs to find nearest control node
    std::queue<NodeID> q;
    std::unordered_set<NodeID> visited;

    const Node* start = graph.get_node(node_id);
    if (!start) return 0;

    for (NodeID in : start->inputs) {
        q.push(in);
        visited.insert(in);
    }

    while (!q.empty()) {
        NodeID cur = q.front(); q.pop();
        const Node* n = graph.get_node(cur);
        if (!n) continue;
        switch (n->kind) {
            case Node::Kind::Start:
            case Node::Kind::Stop:
            case Node::Kind::If:
            case Node::Kind::Region:
            case Node::Kind::Loop:
            case Node::Kind::Return:
                return cur;
            default:
                break;
        }
        for (NodeID in : n->inputs) {
            if (!visited.contains(in)) {
                visited.insert(in);
                q.push(in);
            }
        }
    }
    return 0;
}

NodeID Scheduler::get_parent_block(NodeID block_id) {
    // Return one control predecessor of the block
    auto preds = graph.get_predecessors(block_id);
    for (NodeID p : preds) {
        const Node* pn = graph.get_node(p);
        if (!pn) continue;
        switch (pn->kind) {
            case Node::Kind::Start:
            case Node::Kind::If:
            case Node::Kind::Region:
            case Node::Kind::Loop:
            case Node::Kind::Return:
            case Node::Kind::Stop:
                return p;
            default:
                break;
        }
    }
    return 0;
}

NodeID Scheduler::move_towards_block(NodeID from, NodeID to) {
    // For simplicity, return target block (a conservative move if dominance holds)
    (void)from;
    return to;
}

int Scheduler::get_block_depth(NodeID block_id) {
    // Calculate loop nesting depth
    int depth = 0;
    NodeID current = block_id;

    while (current != 0) {
        const Node* node = graph.get_node(current);
        if (!node) break;

        if (node->kind == Node::Kind::Loop) {
            depth++;
        }

        // Move to parent block
        current = get_parent_block(current);
    }

    return depth;
}

// Alias class utilities
AliasClass get_alias_class(const Node& node) {
    // Extract alias class from node based on field access pattern
    if (node.kind == Node::Kind::Load || node.kind == Node::Kind::Store) {
        // Parse field information from node
        // This would be populated during graph construction
    }
    return {0, "", ""};
}

std::vector<NodeID> find_stores_for_alias(NodeID alias_id) {
    // Find all store nodes that write to this alias class
    // Implementation would scan graph for Store nodes with matching alias
    return {};
}

bool should_add_anti_dependency(NodeID load, NodeID store) {
    // Determine if anti-dependency is needed based on control flow
    // and program ordering requirements
    return true;
}

void add_anti_dependency(NodeID load, NodeID store) {
    // Create anti-dependency edge from load to store
    // This ensures store happens after load in final schedule
}

} // namespace cppfort::mlir_son