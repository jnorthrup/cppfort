#pragma once

#include "graph_node.h"
#include <vector>
#include <optional>

namespace cppfort::stage0 {

class SubgraphMatcher {
public:
    SubgraphMatcher() = default;

    // Attempt to match the pattern graph rooted at 'pattern' into 'root'
    // Returns a list of matched nodes (one per pattern node) or empty if not matched
    std::optional<std::vector<GraphNode*>> match(GraphNode* root, const GraphNode* pattern) const;
};

} // namespace
