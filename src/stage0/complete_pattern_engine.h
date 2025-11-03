#ifndef CPPFORT_COMPLETE_PATTERN_ENGINE_H
#define CPPFORT_COMPLETE_PATTERN_ENGINE_H

#include <yaml-cpp/yaml.h>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <regex>
#include <functional>
#include <iostream>
#include <sstream>
#include "node.h"
#include "type.h"

namespace cppfort {

/**
 * @brief Enum representing different Cpp2 grammar types for classification
 */
enum class GrammarType {
    CPP2,  // C++2 syntax
    CPP,   // C++ syntax  
    C,     // C syntax
    UNKNOWN
};

/**
 * @brief Represents a single pattern for Cpp2 to C++ transformation
 */
struct SemanticPattern {
    std::string name;
    bool use_alternating;
    std::vector<std::string> alternating_anchors;
    int grammar_modes;
    std::vector<std::string> evidence_types;
    int priority;
    std::map<int, std::string> transformation_templates;
    
    // Additional semantic information
    std::string description;
    std::vector<std::string> required_includes;
    std::function<bool(const std::vector<std::string>&)> validation_fn;
};

/**
 * @brief Complete pattern engine that manages semantic transformations
 */
class CompletePatternEngine {
private:
    std::vector<SemanticPattern> m_patterns;
    std::map<std::string, std::string> m_type_mappings;
    std::vector<std::string> m_required_includes;
    
    // Graph-based pattern matching components
    std::vector<std::unique_ptr<ir::Node>> m_transformation_graph;
    
public:
    CompletePatternEngine();
    ~CompletePatternEngine() = default;
    
    /**
     * @brief Load patterns from YAML file
     */
    bool loadPatterns(const std::string& patternFile);
    
    /**
     * @brief Apply all applicable patterns to the input
     */
    std::string applyTransformations(const std::string& input);
    
    /**
     * @brief Apply a specific pattern to an input segment
     */
    std::string applyPattern(const SemanticPattern& pattern, const std::vector<std::string>& matches);
    
    /**
     * @brief Perform contextual analysis to select best transformation
     */
    const SemanticPattern* selectBestPattern(const std::string& input, 
                                           const std::vector<SemanticPattern*>& candidates);
    
    /**
     * @brief Add type mapping (Cpp2 type to C++ equivalent)
     */
    void addTypeMapping(const std::string& cpp2_type, const std::string& cpp_type);
    
    /**
     * @brief Transform Cpp2 type to C++ type
     */
    std::string transformType(const std::string& cpp2_type);
    
    /**
     * @brief Detect grammar type of input
     */
    GrammarType detectGrammar(const std::string& input);
    
    /**
     * @brief Get all required includes for transformations
     */
    const std::vector<std::string>& getRequiredIncludes() const;
    
    /**
     * @brief Perform semantic validation of transformed output
     */
    bool validateTransformation(const std::string& original, const std::string& transformed);
    
    /**
     * @brief Build transformation graph for complex multi-step transformations
     */
    void buildTransformationGraph();
    
    /**
     * @brief Apply transformations using the graph representation
     */
    std::string applyGraphTransformations(const std::string& input);
};

/**
 * @brief Implementation of CompletePatternEngine
 */
inline CompletePatternEngine::CompletePatternEngine() {
    // Initialize common type mappings
    addTypeMapping("i8", "std::int8_t");
    addTypeMapping("i16", "std::int16_t");
    addTypeMapping("i32", "std::int32_t");
    addTypeMapping("i64", "std::int64_t");
    addTypeMapping("u8", "std::uint8_t");
    addTypeMapping("u16", "std::uint16_t");
    addTypeMapping("u32", "std::uint32_t");
    addTypeMapping("u64", "std::uint64_t");
    addTypeMapping("f32", "float");
    addTypeMapping("f64", "double");
    addTypeMapping("_", "auto");
}

inline bool CompletePatternEngine::loadPatterns(const std::string& patternFile) {
    try {
        YAML::Node patternDoc = YAML::LoadFile(patternFile);
        
        for (const auto& patternNode : patternDoc) {
            SemanticPattern pattern;
            
            // Required fields
            pattern.name = patternNode["name"].as<std::string>();
            pattern.use_alternating = patternNode["use_alternating"].as<bool>();
            pattern.grammar_modes = patternNode["grammar_modes"] ? 
                                   patternNode["grammar_modes"].as<int>() : 7;
            pattern.priority = patternNode["priority"] ? 
                              patternNode["priority"].as<int>() : 100;
            
            // Anchors
            if (patternNode["alternating_anchors"]) {
                for (const auto& anchor : patternNode["alternating_anchors"]) {
                    pattern.alternating_anchors.push_back(anchor.as<std::string>());
                }
            }
            
            // Evidence types
            if (patternNode["evidence_types"]) {
                for (const auto& evidence : patternNode["evidence_types"]) {
                    pattern.evidence_types.push_back(evidence.as<std::string>());
                }
            }
            
            // Transformation templates
            if (patternNode["transformation_templates"]) {
                for (const auto& tpl : patternNode["transformation_templates"]) {
                    int key = tpl.first.as<int>();
                    std::string value = tpl.second.as<std::string>();
                    pattern.transformation_templates[key] = value;
                }
            }
            
            m_patterns.push_back(pattern);
        }
        
        // Sort patterns by priority (higher priority first)
        std::sort(m_patterns.begin(), m_patterns.end(), 
                 [](const SemanticPattern& a, const SemanticPattern& b) {
                     return a.priority > b.priority;
                 });
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading patterns: " << e.what() << std::endl;
        return false;
    }
}

inline void CompletePatternEngine::addTypeMapping(const std::string& cpp2_type, const std::string& cpp_type) {
    m_type_mappings[cpp2_type] = cpp_type;
}

inline std::string CompletePatternEngine::transformType(const std::string& cpp2_type) {
    auto it = m_type_mappings.find(cpp2_type);
    if (it != m_type_mappings.end()) {
        return it->second;
    }
    // If not found, return the original type (might be a standard C++ type)
    return cpp2_type;
}

inline GrammarType CompletePatternEngine::detectGrammar(const std::string& input) {
    // Simple heuristics for grammar detection
    if (input.find(": (") != std::string::npos && input.find(") -> ") != std::string::npos) {
        return GrammarType::CPP2;  // Function definition pattern
    } else if (input.find("auto") != std::string::npos && input.find("->") != std::string::npos) {
        return GrammarType::CPP;   // Trailing return type
    } else if (input.find("#include") != std::string::npos) {
        return GrammarType::C;     // C-style include
    }
    return GrammarType::UNKNOWN;
}

inline std::string CompletePatternEngine::applyPattern(const SemanticPattern& pattern, 
                                                      const std::vector<std::string>& matches) {
    // Find the transformation template with the right number of matches
    auto tpl_it = pattern.transformation_templates.find(static_cast<int>(matches.size()));
    if (tpl_it != pattern.transformation_templates.end()) {
        std::string result = tpl_it->second;
        
        // Replace placeholders $1, $2, etc. with actual matches
        for (size_t i = 0; i < matches.size(); ++i) {
            std::string placeholder = "$" + std::to_string(i + 1);
            size_t pos = result.find(placeholder);
            while (pos != std::string::npos) {
                result.replace(pos, placeholder.length(), matches[i]);
                pos = result.find(placeholder, pos + matches[i].length());
            }
        }
        
        return result;
    }
    
    // If no template matches the number of matches, return original
    if (!matches.empty()) {
        return matches[0];  // Return first match as fallback
    }
    return "";
}

inline const SemanticPattern* CompletePatternEngine::selectBestPattern(
    const std::string& input, 
    const std::vector<SemanticPattern*>& candidates) {
    
    if (candidates.empty()) {
        return nullptr;
    }
    
    // For now, return the highest priority pattern
    // In a more sophisticated system, this would consider context
    const SemanticPattern* best = candidates[0];
    for (const auto* pattern : candidates) {
        if (pattern->priority > best->priority) {
            best = pattern;
        }
    }
    
    return best;
}

inline std::string CompletePatternEngine::applyTransformations(const std::string& input) {
    std::string result = input;
    
    // Apply transformations based on the loaded patterns
    for (const auto& pattern : m_patterns) {
        if (pattern.use_alternating && !pattern.alternating_anchors.empty()) {
            // For alternating patterns, look for anchor sequences
            std::string current = result;
            std::string transformed;
            
            // This is a simplified approach - a real implementation would be more sophisticated
            if (pattern.alternating_anchors.size() >= 2) {
                std::string anchor1 = pattern.alternating_anchors[0];
                std::string anchor2 = pattern.alternating_anchors[1];
                
                size_t pos1 = current.find(anchor1);
                while (pos1 != std::string::npos) {
                    size_t pos2 = current.find(anchor2, pos1 + anchor1.length());
                    if (pos2 != std::string::npos) {
                        // Extract segments between anchors
                        std::string before = current.substr(0, pos1);
                        std::string segment1 = current.substr(pos1 + anchor1.length(), 
                                                            pos2 - pos1 - anchor1.length());
                        std::string after = current.substr(pos2 + anchor2.length());
                        
                        // Apply transformation with extracted segments
                        std::vector<std::string> segments = {before, segment1, after};
                        std::string replacement = applyPattern(pattern, segments);
                        
                        result = before + replacement + after;
                        
                        // Continue from after the replacement
                        pos1 = result.find(anchor1, before.length() + replacement.length());
                    } else {
                        break;  // No more pairs found
                    }
                }
            }
        } 
    }
    
    return result;
}

inline const std::vector<std::string>& CompletePatternEngine::getRequiredIncludes() const {
    return m_required_includes;
}

inline bool CompletePatternEngine::validateTransformation(const std::string& original, 
                                                         const std::string& transformed) {
    // Basic validation: ensure we're not producing empty results when we shouldn't
    if (original.empty() && transformed.empty()) {
        return true;
    }
    
    if (!original.empty() && transformed.empty()) {
        return false;  // Input was not empty but output is
    }
    
    // More sophisticated validation would go here
    // For example, checking syntax validity, type consistency, etc.
    return true;
}

inline void CompletePatternEngine::buildTransformationGraph() {
    // Initialize the transformation graph with nodes representing
    // different stages of the transformation process
    // This would connect semantic patterns in dependency order
}

inline std::string CompletePatternEngine::applyGraphTransformations(const std::string& input) {
    // For now, fall back to pattern-based transformation
    // A full graph implementation would traverse the transformation graph
    return applyTransformations(input);
}

} // namespace cppfort

#endif // CPPFORT_COMPLETE_PATTERN_ENGINE_H