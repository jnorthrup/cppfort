#pragma once

#include <cstddef>
#include <string>
#include <vector>
#include <unordered_map>
#include <variant>
#include <memory>

namespace cppfort::stage0 {

enum class GraphNodeType {
    CONFIX,
    PATTERN,
    EVIDENCE,
    FRAGMENT,
    MLIR_OP
};

struct ConfixPayload { char open; char close; };
struct PatternPayload { std::string pattern_name; };
struct EvidencePayload { std::string text; };
struct MLIRPayload { std::string op_name; };

using GraphPayload = std::variant<std::monostate, ConfixPayload, PatternPayload, EvidencePayload, MLIRPayload>;

struct GraphNode {
    GraphNode(GraphNodeType t) : type(t) {}
    GraphNodeType type;
    std::string id;
    size_t start_pos = 0;
    size_t end_pos = 0;
    double confidence = 0.0;
    GraphPayload payload;
    std::vector<std::unique_ptr<GraphNode>> children;
    GraphNode* parent = nullptr;
    std::unordered_map<std::string, GraphNode*> edges;
    int grammar = 0; // grammar type (C/CPP/CPP2) as int

    // Build helper
    GraphNode* addChild(std::unique_ptr<GraphNode> child) {
        child->parent = this;
        children.push_back(std::move(child));
        return children.back().get();
    }
};

} // namespace cppfort::stage0
