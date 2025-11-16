#include "graph_serde.h"
#include "graph_node.h"
#include <yaml-cpp/yaml.h>
#include <sstream>

#ifdef HAVE_NLOHMANN_JSON
#include <nlohmann/json.hpp>
using nlohmann::json;
#endif

namespace cppfort::stage0 {

static const char* graphNodeTypeToString(GraphNodeType t) {
    switch (t) {
        case GraphNodeType::CONFIX: return "CONFIX";
        case GraphNodeType::PATTERN: return "PATTERN";
        case GraphNodeType::EVIDENCE: return "EVIDENCE";
        case GraphNodeType::FRAGMENT: return "FRAGMENT";
        case GraphNodeType::MLIR_OP: return "MLIR_OP";
        default: return "UNKNOWN";
    }
}

static GraphNodeType stringToGraphNodeType(const std::string& s) {
    if (s == "CONFIX") return GraphNodeType::CONFIX;
    if (s == "PATTERN") return GraphNodeType::PATTERN;
    if (s == "EVIDENCE") return GraphNodeType::EVIDENCE;
    if (s == "FRAGMENT") return GraphNodeType::FRAGMENT;
    if (s == "MLIR_OP") return GraphNodeType::MLIR_OP;
    return GraphNodeType::CONFIX;
}

YAML::Node graphNodeToYaml(const GraphNode& node) {
    YAML::Node n;
    n["type"] = graphNodeTypeToString(node.type);
    n["id"] = node.id;
    n["start_pos"] = node.start_pos;
    n["end_pos"] = node.end_pos;
    n["confidence"] = node.confidence;
    n["grammar"] = node.grammar;

    // payload
    if (std::holds_alternative<ConfixPayload>(node.payload)) {
        const auto& p = std::get<ConfixPayload>(node.payload);
        n["payload"]["kind"] = "ConfixPayload";
        n["payload"]["open"] = std::string(1, p.open);
        n["payload"]["close"] = std::string(1, p.close);
    } else if (std::holds_alternative<PatternPayload>(node.payload)) {
        const auto& p = std::get<PatternPayload>(node.payload);
        n["payload"]["kind"] = "PatternPayload";
        n["payload"]["pattern_name"] = p.pattern_name;
    } else if (std::holds_alternative<EvidencePayload>(node.payload)) {
        const auto& p = std::get<EvidencePayload>(node.payload);
        n["payload"]["kind"] = "EvidencePayload";
        n["payload"]["text"] = p.text;
    } else if (std::holds_alternative<MLIRPayload>(node.payload)) {
        const auto& p = std::get<MLIRPayload>(node.payload);
        n["payload"]["kind"] = "MLIRPayload";
        n["payload"]["op_name"] = p.op_name;
    }

    // children
    if (!node.children.empty()) {
        for (const auto& child : node.children) {
            n["children"].push_back(graphNodeToYaml(*child));
        }
    }

    // edges: store as mapping of name -> child id (if available)
    if (!node.edges.empty()) {
        for (const auto& edge : node.edges) {
            if (edge.second) {
                n["edges"][edge.first] = edge.second->id;
            }
        }
    }

    return n;
}

std::unique_ptr<GraphNode> yamlToGraphNode(const YAML::Node& n) {
    if (!n.IsMap()) return nullptr;
    std::string type_str = n["type"].as<std::string>("CONFIX");
    GraphNodeType t = stringToGraphNodeType(type_str);
    auto node = std::make_unique<GraphNode>(t);
    node->id = n["id"] ? n["id"].as<std::string>() : std::string();
    node->start_pos = n["start_pos"] ? n["start_pos"].as<size_t>() : 0;
    node->end_pos = n["end_pos"] ? n["end_pos"].as<size_t>() : 0;
    node->confidence = n["confidence"] ? n["confidence"].as<double>() : 0.0;
    node->grammar = n["grammar"] ? n["grammar"].as<int>() : 0;

    if (n["payload"] && n["payload"].IsMap()) {
        auto kind = n["payload"]["kind"].as<std::string>("");
        if (kind == "ConfixPayload") {
            std::string open = n["payload"]["open"].as<std::string>("{");
            std::string close = n["payload"]["close"].as<std::string>("}");
            node->payload = ConfixPayload{open[0], close[0]};
        } else if (kind == "PatternPayload") {
            node->payload = PatternPayload{n["payload"]["pattern_name"].as<std::string>("")};
        } else if (kind == "EvidencePayload") {
            node->payload = EvidencePayload{n["payload"]["text"].as<std::string>("")};
        } else if (kind == "MLIRPayload") {
            node->payload = MLIRPayload{n["payload"]["op_name"].as<std::string>("")};
        }
    }

    // children
    if (n["children"] && n["children"].IsSequence()) {
        for (const auto& c : n["children"]) {
            auto child_node = yamlToGraphNode(c);
            if (child_node) node->addChild(std::move(child_node));
        }
    }

    // edges: We can't resolve to child pointer without additional context, so store edge names mapping to id in 'edges' map with nullptr when not resolvable
    if (n["edges"] && n["edges"].IsMap()) {
        for (const auto& kv : n["edges"]) {
            std::string key = kv.first.as<std::string>();
            // We only store id here; resolving to pointers will need separate pass
            node->edges[key] = nullptr;
        }
    }

    return node;
}

#ifdef HAVE_NLOHMANN_JSON
json graphNodeToJson(const GraphNode& node) {
    json j;
    j["type"] = graphNodeTypeToString(node.type);
    j["id"] = node.id;
    j["start_pos"] = node.start_pos;
    j["end_pos"] = node.end_pos;
    j["confidence"] = node.confidence;
    j["grammar"] = node.grammar;

    if (std::holds_alternative<ConfixPayload>(node.payload)) {
        const auto& p = std::get<ConfixPayload>(node.payload);
        j["payload"] = { {"kind", "ConfixPayload"}, {"open", std::string(1, p.open)}, {"close", std::string(1, p.close)} };
    } else if (std::holds_alternative<PatternPayload>(node.payload)) {
        const auto& p = std::get<PatternPayload>(node.payload);
        j["payload"] = { {"kind", "PatternPayload"}, {"pattern_name", p.pattern_name} };
    } else if (std::holds_alternative<EvidencePayload>(node.payload)) {
        const auto& p = std::get<EvidencePayload>(node.payload);
        j["payload"] = { {"kind", "EvidencePayload"}, {"text", p.text} };
    } else if (std::holds_alternative<MLIRPayload>(node.payload)) {
        const auto& p = std::get<MLIRPayload>(node.payload);
        j["payload"] = { {"kind", "MLIRPayload"}, {"op_name", p.op_name} };
    }

    // children
    if (!node.children.empty()) {
        for (const auto& child : node.children) {
            j["children"].push_back(graphNodeToJson(*child));
        }
    }

    // edges
    if (!node.edges.empty()) {
        for (const auto& edge : node.edges) {
            if (edge.second) j["edges"][edge.first] = edge.second->id;
            else j["edges"][edge.first] = nullptr;
        }
    }

    return j;
}

std::unique_ptr<GraphNode> jsonToGraphNode(const json& j) {
    if (!j.is_object()) return nullptr;
    std::string type_str = j.value("type", "CONFIX");
    GraphNodeType t = stringToGraphNodeType(type_str);
    auto node = std::make_unique<GraphNode>(t);
    node->id = j.value("id", std::string());
    node->start_pos = j.value("start_pos", size_t{0});
    node->end_pos = j.value("end_pos", size_t{0});
    node->confidence = j.value("confidence", 0.0);
    node->grammar = j.value("grammar", 0);

    if (j.contains("payload") && j["payload"].is_object()) {
        std::string kind = j["payload"].value("kind", std::string());
        if (kind == "ConfixPayload") {
            std::string open = j["payload"].value("open", std::string("{"));
            std::string close = j["payload"].value("close", std::string("}"));
            node->payload = ConfixPayload{open[0], close[0]};
        } else if (kind == "PatternPayload") {
            node->payload = PatternPayload{j["payload"].value("pattern_name", std::string())};
        } else if (kind == "EvidencePayload") {
            node->payload = EvidencePayload{j["payload"].value("text", std::string())};
        } else if (kind == "MLIRPayload") {
            node->payload = MLIRPayload{j["payload"].value("op_name", std::string())};
        }
    }

    // children
    if (j.contains("children") && j["children"].is_array()) {
        for (const auto& child : j["children"]) {
            auto child_node = jsonToGraphNode(child);
            if (child_node) node->addChild(std::move(child_node));
        }
    }

    // edges
    if (j.contains("edges") && j["edges"].is_object()) {
        for (auto it = j["edges"].begin(); it != j["edges"].end(); ++it) {
            node->edges[it.key()] = nullptr;
        }
    }

    return node;
}
#endif

} // namespace cppfort::stage0
