#include "pijul_graph.h"

#include <algorithm>
#include <queue>

namespace cppfort::pijul {

namespace {

bool edge_exists(const std::vector<GraphEdge>& edges, Graph::NodeId target) {
    return std::any_of(edges.begin(), edges.end(), [&](const GraphEdge& edge) {
        return edge.target == target;
    });
}

} // namespace

Graph::Graph() = default;

Graph::NodeId Graph::add_node(const ExternalKey& key) {
    GraphNode node;
    node.key = key;
    m_nodes.push_back(node);
    return m_nodes.size() - 1;
}

Graph::NodeId Graph::ensure_node(const ExternalKey& key) {
    auto it = m_index.find(key);
    if (it != m_index.end()) {
        return it->second;
    }

    NodeId id = add_node(key);
    m_index.emplace(key, id);
    return id;
}

void Graph::add_edge(NodeId from, NodeId to, EdgeFlag flag) {
    if (from >= m_nodes.size() || to >= m_nodes.size()) {
        return;
    }
    if (edge_exists(m_nodes[from].outgoing, to)) {
        return;
    }

    m_nodes[from].outgoing.push_back({to, flag});
    m_nodes[to].incoming.push_back({from, flag});
}

std::size_t Graph::fanout(NodeId node) const {
    if (node >= m_nodes.size()) {
        return 0;
    }
    return m_nodes[node].outgoing.size();
}

std::size_t Graph::fanin(NodeId node) const {
    if (node >= m_nodes.size()) {
        return 0;
    }
    return m_nodes[node].incoming.size();
}

bool Graph::has_edge(NodeId from, NodeId to) const {
    if (from >= m_nodes.size()) {
        return false;
    }
    return edge_exists(m_nodes[from].outgoing, to);
}

bool Graph::has_cycle_from(NodeId node, std::vector<int>& state) const {
    state[node] = 1;
    for (const auto& edge : m_nodes[node].outgoing) {
        if (state[edge.target] == 1) {
            return true;
        }
        if (state[edge.target] == 0 && has_cycle_from(edge.target, state)) {
            return true;
        }
    }
    state[node] = 2;
    return false;
}

bool Graph::has_cycle() const {
    std::vector<int> state(m_nodes.size(), 0);
    for (NodeId node = 0; node < m_nodes.size(); ++node) {
        if (state[node] == 0 && has_cycle_from(node, state)) {
            return true;
        }
    }
    return false;
}

std::vector<Graph::NodeId> Graph::topological_order() const {
    std::vector<NodeId> order;
    order.reserve(m_nodes.size());

    std::vector<std::size_t> indegree(m_nodes.size(), 0);
    for (const auto& node : m_nodes) {
        for (const auto& edge : node.outgoing) {
            indegree[edge.target]++;
        }
    }

    std::queue<NodeId> q;
    for (NodeId node = 0; node < m_nodes.size(); ++node) {
        if (indegree[node] == 0) {
            q.push(node);
        }
    }

    while (!q.empty()) {
        NodeId current = q.front();
        q.pop();
        order.push_back(current);
        for (const auto& edge : m_nodes[current].outgoing) {
            if (--indegree[edge.target] == 0) {
                q.push(edge.target);
            }
        }
    }

    return order;
}

std::vector<ExternalKey> Graph::dependent_keys(NodeId node) const {
    std::vector<ExternalKey> keys;
    if (node >= m_nodes.size()) {
        return keys;
    }

    for (const auto& edge : m_nodes[node].incoming) {
        keys.push_back(m_nodes[edge.target].key);
    }
    return keys;
}

} // namespace cppfort::pijul

