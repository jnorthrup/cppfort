#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <vector>

#include "pijul_patch.h"

namespace cppfort::pijul {

enum class EdgeFlag : std::uint8_t {
    None = 0,
    Pseudo = 1,
    Folder = 2,
    Parent = 4,
    Deleted = 8
};

struct GraphEdge {
    std::size_t target = 0;
    EdgeFlag flag = EdgeFlag::None;
};

struct GraphNode {
    ExternalKey key;
    std::vector<GraphEdge> outgoing;
    std::vector<GraphEdge> incoming;
};

class Graph {
public:
    using NodeId = std::size_t;

    Graph();

    NodeId ensure_node(const ExternalKey& key);
    void add_edge(NodeId from, NodeId to, EdgeFlag flag);

    std::size_t fanout(NodeId node) const;
    std::size_t fanin(NodeId node) const;

    bool has_edge(NodeId from, NodeId to) const;
    bool has_cycle() const;

    std::vector<NodeId> topological_order() const;
    std::vector<ExternalKey> dependent_keys(NodeId node) const;

private:
    NodeId add_node(const ExternalKey& key);
    bool has_cycle_from(NodeId node, std::vector<int>& state) const;

    std::vector<GraphNode> m_nodes;
    std::map<ExternalKey, NodeId> m_index;
};

} // namespace cppfort::pijul

