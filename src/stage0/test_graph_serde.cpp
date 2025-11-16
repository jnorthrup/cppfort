#include <iostream>
#include <cassert>
#include "graph_node.h"
#include "graph_serde.h"
#include <yaml-cpp/yaml.h>

#ifdef HAVE_NLOHMANN_JSON
#include <nlohmann/json.hpp>
using nlohmann::json;
#endif

using namespace cppfort::stage0;

int main() {
    // Build a simple graph node hierarchy
    auto root = std::make_unique<GraphNode>(GraphNodeType::CONFIX);
    root->id = "root";
    root->start_pos = 0;
    root->end_pos = 100;
    root->confidence = 0.75;
    root->payload = ConfixPayload{'{', '}'};

    auto child1 = std::make_unique<GraphNode>(GraphNodeType::EVIDENCE);
    child1->id = "child1";
    child1->payload = EvidencePayload{"int a = 0;"};

    root->addChild(std::move(child1));

    // Serialize to YAML
    YAML::Node y = graphNodeToYaml(*root);
    std::string emitted = YAML::Dump(y);
    std::cout << "YAML: " << emitted << "\n";

    // Deserialize back
    auto parsed = yamlToGraphNode(y);
    assert(parsed && parsed->id == "root");
    assert(!parsed->children.empty());
    assert(parsed->children[0]->id == "child1");

#ifdef HAVE_NLOHMANN_JSON
    auto j = graphNodeToJson(*root);
    std::string jstr = j.dump(2);
    std::cout << "JSON: " << jstr << "\n";
    auto parsed2 = jsonToGraphNode(j);
    assert(parsed2 && parsed2->id == "root");
    assert(parsed2->children.size() == 1);
#endif

    std::cout << "Graph serde roundtrip OK\n";
    return 0;
}
