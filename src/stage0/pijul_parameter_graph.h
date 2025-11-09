#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pijul_types.h"
#include "pijul_orbit_builder.h"

namespace cppfort::pijul {

struct ParameterAnchor {
    std::string signature;
    std::string pattern;
    std::size_t depth = 0;
    std::unordered_map<std::string, std::string> parameters;
    NodeContext context;
    std::string description;
    std::string source_fragment;
    std::string transformed_fragment;
    std::vector<std::string> source_tokens;
    std::vector<std::string> transformed_tokens;
};

struct ParameterEdge {
    std::string from;
    std::string to;
    std::string reason;
};

class ParameterGraph {
public:
    ParameterGraph() = default;

    const ParameterAnchor& add_anchor(const ParameterAnchor& anchor);
    void add_edge(const std::string& from,
                  const std::string& to,
                  const std::string& reason);

    std::optional<ParameterAnchor> find(const std::string& signature) const;
    const std::vector<ParameterEdge>& edges() const { return m_edges; }
    const std::vector<ParameterAnchor>& anchors() const { return m_anchors; }

private:
    std::vector<ParameterAnchor> m_anchors;
    std::unordered_map<std::string, std::size_t> m_index;
    std::vector<ParameterEdge> m_edges;
};

ParameterAnchor make_anchor(const OrbitMatchInfo& match,
                            std::string_view source_fragment,
                            std::string_view transformed_fragment,
                            const std::unordered_map<std::string, std::string>& params,
                            const std::string& description);

void populate_parameter_graph(ParameterGraph& graph,
                              const OrbitMatchCollection& source,
                              const OrbitMatchCollection& transformed,
                              const std::string& source_code,
                              const std::string& transformed_code);

} // namespace cppfort::pijul
