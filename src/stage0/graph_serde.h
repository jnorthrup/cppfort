#pragma once

#include "graph_node.h"
#include <yaml-cpp/yaml.h>
#include <memory>
#include <string>

#ifdef HAVE_NLOHMANN_JSON
#include <nlohmann/json.hpp>
using nlohmann::json;
#endif

namespace cppfort::stage0 {

// Serialize GraphNode to a YAML::Node
YAML::Node graphNodeToYaml(const GraphNode& node);

// Deserialize YAML::Node to a GraphNode
std::unique_ptr<GraphNode> yamlToGraphNode(const YAML::Node& node);

#ifdef HAVE_NLOHMANN_JSON
// Serialize GraphNode to JSON object
json graphNodeToJson(const GraphNode& node);

// Deserialize JSON object to GraphNode
std::unique_ptr<GraphNode> jsonToGraphNode(const json& j);
#endif

} // namespace cppfort::stage0
