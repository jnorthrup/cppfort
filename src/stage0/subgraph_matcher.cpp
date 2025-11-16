#include "subgraph_matcher.h"

#include <algorithm>
#include <iostream>

namespace cppfort::stage0 {

namespace {
    bool payload_matches(const GraphPayload& src, const GraphPayload& pat) {
        // Basic checks: if either is monostate, accept
        if (std::holds_alternative<std::monostate>(pat)) return true;
        if (pat.index() == src.index()) return true;
        // Further heuristics could be added here
        return false;
    }

    bool match_tree(GraphNode* src, const GraphNode* pat, std::vector<GraphNode*>& out) {
        if (!src || !pat) return false;
        if (src->type != pat->type) return false;
        if (!payload_matches(src->payload, pat->payload)) return false;

        // Simplistic child count check
        if (src->children.size() < pat->children.size()) return false;

        out.push_back(src);
        // Greedy match: match pat children against the prefix of src children
        for (size_t i = 0; i < pat->children.size(); ++i) {
            if (!match_tree(src->children[i].get(), pat->children[i].get(), out)) {
                return false;
            }
        }
        return true;
    }
}

std::optional<std::vector<GraphNode*>> SubgraphMatcher::match(GraphNode* root, const GraphNode* pattern) const {
    if (!root || !pattern) return std::nullopt;
    std::vector<GraphNode*> matched;
    if (match_tree(root, pattern, matched)) {
        return matched;
    }
    // Otherwise search children
    for (auto& c : root->children) {
        auto res = match(c.get(), pattern);
        if (res) return res;
    }
    return std::nullopt;
}

} // namespace cppfort::stage0
