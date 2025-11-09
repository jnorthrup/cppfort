#ifndef CPPFORT_GRAPH_SEMANTIC_TRANSFORMER_H
#define CPPFORT_GRAPH_SEMANTIC_TRANSFORMER_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <sstream>
#include <list>
#include <queue>
#include <algorithm>

#include "node.h"
#include "type.h"
#include "orbit_mask.h"
#include "semantic_transformer.h"

namespace cppfort::ir {

/**
 * @brief Represents a transformation node in the semantic transformation graph
 */
struct TransformationNode {
    std::string id;
    std::string cpp2_construct;           // The Cpp2 construct being transformed
    std::string cpp_equivalent;           // The C++ equivalent
    std::vector<std::string> dependencies; // Other nodes this node depends on
    std::vector<std::string> dependents;   // Nodes that depend on this one
    int priority;                         // Transformation priority
    bool processed;                       // Whether this node has been processed
    
    TransformationNode(const std::string& node_id, const std::string& cpp2, 
                      const std::string& cpp, int prio = 0) 
        : id(node_id), cpp2_construct(cpp2), cpp_equivalent(cpp), 
          priority(prio), processed(false) {}
};

/**
 * @brief Represents a transformation edge in the semantic transformation graph
 */
struct TransformationEdge {
    std::string from_node;
    std::string to_node;
    std::string transformation_type;  // type of transformation (e.g., "function", "parameter", "type")
    
    TransformationEdge(const std::string& from, const std::string& to, 
                      const std::string& type)
        : from_node(from), to_node(to), transformation_type(type) {}
};

/**
 * @brief Core graph-based semantic transformation engine
 * 
 * This class implements the graph-based flexibility system for semantic transformations,
 * allowing for bidirectional transformation and systematic correlation between
 * Cpp2 and C++ constructs.
 */
class GraphSemanticTransformer {
private:
    std::vector<TransformationNode> m_nodes;
    std::vector<TransformationEdge> m_edges;
    std::unordered_map<std::string, std::string> m_type_mappings;
    std::unordered_map<std::string, int> m_node_indices;
    std::vector<std::string> m_required_includes;
    
    // Bidirectional mapping
    std::unordered_map<std::string, std::string> m_cpp2_to_cpp;
    std::unordered_map<std::string, std::string> m_cpp_to_cpp2;
    
public:
    GraphSemanticTransformer();
    
    /**
     * @brief Add a transformation node to the graph
     */
    void addTransformationNode(const TransformationNode& node);
    
    /**
     * @brief Add a transformation edge between nodes
     */
    void addTransformationEdge(const TransformationEdge& edge);
    
    /**
     * @brief Add a bidirectional transformation mapping
     */
    void addBidirectionalMapping(const std::string& cpp2_construct, 
                                const std::string& cpp_equivalent);
    
    /**
     * @brief Transform Cpp2 code to C++ using the transformation graph
     */
    std::string transformCpp2ToCpp(const std::string& cpp2_code);
    
    /**
     * @brief Transform C++ code to Cpp2 using the transformation graph (reverse transformation)
     */
    std::string transformCppToCpp2(const std::string& cpp_code);
    
    /**
     * @brief Add a required include for the transformation
     */
    void addRequiredInclude(const std::string& include);
    
    /**
     * @brief Get all required includes for the transformations
     */
    const std::vector<std::string>& getRequiredIncludes() const;
    
    /**
     * @brief Topologically sort the transformation nodes based on dependencies
     */
    std::vector<std::string> topologicalSort();
    
    /**
     * @brief Apply transformations in topological order
     */
    std::string applyTransformations(const std::string& input, 
                                    const std::vector<std::string>& order);
    
    /**
     * @brief Validate the transformation graph for cycles and completeness
     */
    bool validateGraph();
    
    /**
     * @brief Get the transformation path for a specific construct
     */
    std::vector<std::string> getTransformationPath(const std::string& construct);
    
    /**
     * @brief Correlate transformations between Cpp2 and C++ constructs
     */
    void correlateTransformations(const std::string& cpp2_construct, 
                                 const std::string& cpp_construct);
    
private:
    /**
     * @brief Initialize with common transformation mappings
     */
    void initializeCommonMappings();
    
    /**
     * @brief Apply a single transformation node to the code
     */
    std::string applyNodeTransformation(const TransformationNode& node, 
                                       const std::string& code);
};

inline GraphSemanticTransformer::GraphSemanticTransformer() {
    // Initialize with common mappings
    initializeCommonMappings();
}

inline void GraphSemanticTransformer::addTransformationNode(const TransformationNode& node) {
    m_nodes.push_back(node);
    m_node_indices[node.id] = m_nodes.size() - 1;
}

inline void GraphSemanticTransformer::addTransformationEdge(const TransformationEdge& edge) {
    m_edges.push_back(edge);
    
    // Update dependency relationships
    auto from_it = m_node_indices.find(edge.from_node);
    auto to_it = m_node_indices.find(edge.to_node);
    
    if (from_it != m_node_indices.end() && to_it != m_node_indices.end()) {
        size_t from_idx = from_it->second;
        size_t to_idx = to_it->second;
        
        // Add forward dependency
        m_nodes[from_idx].dependents.push_back(edge.to_node);
        
        // Add backward dependency
        m_nodes[to_idx].dependencies.push_back(edge.from_node);
    }
}

inline void GraphSemanticTransformer::addBidirectionalMapping(
    const std::string& cpp2_construct, 
    const std::string& cpp_equivalent) {
    
    m_cpp2_to_cpp[cpp2_construct] = cpp_equivalent;
    m_cpp_to_cpp2[cpp_equivalent] = cpp2_construct;
}

inline std::string GraphSemanticTransformer::transformCpp2ToCpp(const std::string& cpp2_code) {
    // Add required includes
    std::string includes;
    for (const auto& inc : m_required_includes) {
        includes += "#include <" + inc + ">\n";
    }
    if (!m_required_includes.empty()) {
        includes += "\n";
    }
    
    // Get topological order of transformations
    std::vector<std::string> order = topologicalSort();
    
    // Apply transformations in order
    std::string result = includes + cpp2_code;
    result = applyTransformations(result, order);
    
    return result;
}

inline std::string GraphSemanticTransformer::transformCppToCpp2(const std::string& cpp_code) {
    // For reverse transformation, we iterate through the reverse mappings
    std::string result = cpp_code;
    
    for (const auto& mapping : m_cpp_to_cpp2) {
        size_t pos = 0;
        while ((pos = result.find(mapping.first, pos)) != std::string::npos) {
            result.replace(pos, mapping.first.length(), mapping.second);
            pos += mapping.second.length();
        }
    }
    
    return result;
}

inline void GraphSemanticTransformer::addRequiredInclude(const std::string& include) {
    if (std::find(m_required_includes.begin(), m_required_includes.end(), include) == 
        m_required_includes.end()) {
        m_required_includes.push_back(include);
    }
}

inline const std::vector<std::string>& GraphSemanticTransformer::getRequiredIncludes() const {
    return m_required_includes;
}

inline std::vector<std::string> GraphSemanticTransformer::topologicalSort() {
    std::vector<std::string> result;
    std::unordered_set<std::string> visited;
    std::list<std::string> temp_list;
    
    // Helper function for DFS
    std::function<void(const std::string&)> dfs = [&](const std::string& node_id) {
        if (visited.count(node_id)) return;
        
        visited.insert(node_id);
        
        auto it = m_node_indices.find(node_id);
        if (it != m_node_indices.end()) {
            size_t idx = it->second;
            for (const auto& dep : m_nodes[idx].dependencies) {
                dfs(dep);
            }
        }
        
        temp_list.push_front(node_id);
    };
    
    // Perform DFS for each unvisited node
    for (const auto& node : m_nodes) {
        if (!visited.count(node.id)) {
            dfs(node.id);
        }
    }
    
    // Convert list to vector
    for (const auto& node_id : temp_list) {
        result.push_back(node_id);
    }
    
    return result;
}

inline std::string GraphSemanticTransformer::applyTransformations(
    const std::string& input, 
    const std::vector<std::string>& order) {
    
    std::string result = input;
    
    for (const auto& node_id : order) {
        auto it = m_node_indices.find(node_id);
        if (it != m_node_indices.end()) {
            size_t idx = it->second;
            if (!m_nodes[idx].processed) {
                result = applyNodeTransformation(m_nodes[idx], result);
                m_nodes[idx].processed = true;
            }
        }
    }
    
    // Reset processed flags for next use
    for (auto& node : m_nodes) {
        node.processed = false;
    }
    
    return result;
}

inline bool GraphSemanticTransformer::validateGraph() {
    // Check for cycles using topological sort
    std::vector<std::string> order = topologicalSort();
    
    // If all nodes are in the topological order, there are no cycles
    return order.size() == m_nodes.size();
}

inline std::vector<std::string> GraphSemanticTransformer::getTransformationPath(
    const std::string& construct) {
    // Find nodes that match the construct
    std::vector<std::string> path;
    
    for (const auto& node : m_nodes) {
        if (node.cpp2_construct.find(construct) != std::string::npos ||
            node.cpp_equivalent.find(construct) != std::string::npos) {
            
            // Build path from dependencies
            std::vector<std::string> current_path;
            std::unordered_set<std::string> visited;
            
            std::function<void(const std::string&)> buildPath = 
                [&](const std::string& node_id) {
                    if (visited.count(node_id)) return;
                    
                    visited.insert(node_id);
                    current_path.push_back(node_id);
                    
                    auto it = m_node_indices.find(node_id);
                    if (it != m_node_indices.end()) {
                        size_t idx = it->second;
                        for (const auto& dep : m_nodes[idx].dependencies) {
                            buildPath(dep);
                        }
                    }
                };
            
            buildPath(node.id);
            
            // Reverse to get the right order
            std::reverse(current_path.begin(), current_path.end());
            
            if (current_path.size() > path.size()) {
                path = current_path;
            }
        }
    }
    
    return path;
}

inline void GraphSemanticTransformer::correlateTransformations(
    const std::string& cpp2_construct, 
    const std::string& cpp_construct) {
    // Create the bidirectional mapping
    addBidirectionalMapping(cpp2_construct, cpp_construct);
    
    // Also add a transformation node to track the correlation
    std::string node_id = "corr_" + std::to_string(m_nodes.size());
    TransformationNode node(node_id, cpp2_construct, cpp_construct, 100);
    addTransformationNode(node);
}

inline void GraphSemanticTransformer::initializeCommonMappings() {
    // Basic function mapping: main: () -> int = { } -> int main() { }
    addBidirectionalMapping("main: () -> int = {", "int main() {");
    addRequiredInclude("iostream");
    
    // Variable declaration: x: int = 5; -> int x = 5;
    addBidirectionalMapping(": =", " = ");
    
    // Auto variable: x := value -> auto x = value
    addBidirectionalMapping(" := ", "auto  = ");
    
    // Parameter modes
    addBidirectionalMapping("inout ", "& ");
    addBidirectionalMapping("move ", "&& ");
    
    // Type mappings
    addBidirectionalMapping("i32", "std::int32_t");
    addBidirectionalMapping("i64", "std::int64_t");
    addBidirectionalMapping("f64", "double");
    addBidirectionalMapping("f32", "float");
    
    // Add transformation nodes for complex constructs
    TransformationNode func_node("func_decl", "function declaration", "C++ function declaration", 90);
    addTransformationNode(func_node);
    
    TransformationNode param_node("param_decl", "parameter declaration", "C++ parameter declaration", 85);
    addTransformationNode(param_node);
    
    TransformationNode var_node("var_decl", "variable declaration", "C++ variable declaration", 80);
    addTransformationNode(var_node);
}

inline std::string GraphSemanticTransformer::applyNodeTransformation(
    const TransformationNode& node, 
    const std::string& code) {
    
    std::string result = code;
    
    // This is a simplified transformation - a full implementation would be more sophisticated
    if (!node.cpp2_construct.empty() && !node.cpp_equivalent.empty()) {
        size_t pos = 0;
        while ((pos = result.find(node.cpp2_construct, pos)) != std::string::npos) {
            result.replace(pos, node.cpp2_construct.length(), node.cpp_equivalent);
            pos += node.cpp_equivalent.length();
        }
    }
    
    return result;
}

} // namespace cppfort::ir

#endif // CPPFORT_GRAPH_SEMANTIC_TRANSFORMER_H