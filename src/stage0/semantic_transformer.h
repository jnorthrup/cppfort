#ifndef CPPFORT_SEMANTIC_TRANSFORMER_H
#define CPPFORT_SEMANTIC_TRANSFORMER_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>
#include <sstream>

#include "node.h"
#include "type.h"
#include "orbit_mask.h"
#include "unified_pattern_matcher.h"

namespace cppfort::ir {

/**
 * @brief Represents a semantic transformation rule from Cpp2 to C++
 */
struct SemanticTransformRule {
    std::string cpp2_pattern;      // Pattern to match in Cpp2
    std::string cpp_pattern;       // Output pattern in C++
    std::function<std::string(const std::vector<std::string>&)> transform_fn;
    int priority;                  // Priority for rule selection (higher = more specific)
    std::string description;       // Human-readable description
    
    SemanticTransformRule(
        const std::string& cpp2_pat,
        const std::string& cpp_pat,
        std::function<std::string(const std::vector<std::string>&)> fn,
        int prio,
        const std::string& desc
    ) : cpp2_pattern(cpp2_pat), cpp_pattern(cpp_pat), transform_fn(fn), 
        priority(prio), description(desc) {}
};

/**
 * @brief Core semantic transformation engine
 * 
 * This class handles the transformation of Cpp2 constructs to equivalent C++ constructs
 * with proper semantic preservation and context awareness.
 */
class SemanticTransformer {
private:
    std::vector<SemanticTransformRule> m_rules;
    std::unordered_map<std::string, std::string> m_type_mappings;
    std::vector<std::string> m_required_includes;
    
    // Pattern matching for complex transformations
    std::unique_ptr<UnifiedPatternMatcher> m_pattern_matcher;

public:
    SemanticTransformer();
    
    /**
     * @brief Add a semantic transformation rule
     */
    void addRule(const SemanticTransformRule& rule);
    
    /**
     * @brief Add a type mapping (Cpp2 type to C++ equivalent)
     */
    void addTypeMapping(const std::string& cpp2_type, const std::string& cpp_type);
    
    /**
     * @brief Transform a Cpp2 code segment to C++
     */
    std::string transform(const std::string& cpp2_code);
    
    /**
     * @brief Transform a node in the AST
     */
    std::string transformNode(Node* node);
    
    /**
     * @brief Add a required include for the transformation
     */
    void addRequiredInclude(const std::string& include);
    
    /**
     * @brief Get all required includes for the transformations
     */
    const std::vector<std::string>& getRequiredIncludes() const;
    
    /**
     * @brief Validate that the transformation preserves semantics
     */
    bool validate(const std::string& original, const std::string& transformed);
    
private:
    /**
     * @brief Apply the most appropriate transformation rule to the input
     */
    std::string applyBestRule(const std::string& input);
    
    /**
     * @brief Initialize with common transformation rules
     */
    void initializeCommonTransforms();
    
    /**
     * @brief Transform function declarations
     */
    std::string transformFunctionDeclaration(const std::string& func_decl);
    
    /**
     * @brief Transform parameter declarations with modes
     */
    std::string transformParameter(const std::string& param_decl);
    
    /**
     * @brief Transform variable declarations
     */
    std::string transformVariable(const std::string& var_decl);
    
    /**
     * @brief Transform type aliases
     */
    std::string transformTypeAlias(const std::string& type_alias);
    
    /**
     * @brief Transform function calls with UFCS
     */
    std::string transformUfcsCall(const std::string& call_expr);
    
    /**
     * @brief Transform inspect expressions to equivalent C++ constructs
     */
    std::string transformInspectExpression(const std::string& inspect_expr);
    
    /**
     * @brief Transform as/cast expressions
     */
    std::string transformAsExpression(const std::string& as_expr);
};

inline SemanticTransformer::SemanticTransformer() : m_pattern_matcher(nullptr) {
    // Initialize with common type mappings
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
    
    // Initialize common transformations
    initializeCommonTransforms();
}

inline void SemanticTransformer::addRule(const SemanticTransformRule& rule) {
    m_rules.push_back(rule);
    // Keep rules sorted by priority (highest first)
    std::sort(m_rules.begin(), m_rules.end(), 
              [](const SemanticTransformRule& a, const SemanticTransformRule& b) {
                  return a.priority > b.priority;
              });
}

inline void SemanticTransformer::addTypeMapping(const std::string& cpp2_type, const std::string& cpp_type) {
    m_type_mappings[cpp2_type] = cpp_type;
}

inline std::string SemanticTransformer::transform(const std::string& cpp2_code) {
    std::string result = cpp2_code;
    
    // Add required includes at the top
    std::string includes;
    for (const auto& inc : m_required_includes) {
        includes += "#include <" + inc + ">\n";
    }
    if (!m_required_includes.empty()) {
        includes += "\n";
    }
    
    // Apply transformations in order of priority
    std::string current = result;
    for (const auto& rule : m_rules) {
        // This is a simplified approach - a full implementation would use more sophisticated pattern matching
        std::string transformed = applyBestRule(current);
        if (!transformed.empty() && transformed != current) {
            current = transformed;
        }
    }
    
    return includes + current;
}

inline std::string SemanticTransformer::transformNode(Node* node) {
    if (!node) return "";
    
    // Route to the appropriate transformation based on node kind
    switch (node->getKind()) {
        case NodeKind::FUNCTION:
            // Handle function transformation
            return "/* Function transformation */";
        case NodeKind::PARAMETER:
            // Handle parameter transformation
            return "/* Parameter transformation */";
        case NodeKind::CONSTANT:
            // Handle constant transformation
            return std::to_string(dynamic_cast<ConstantNode*>(node)->_value);
        case NodeKind::ADD:
            // Handle binary operation
            return "/* Binary operation transformation */";
        case NodeKind::RETURN:
            // Handle return transformation
            return "/* Return transformation */";
        default:
            // Default: just return node label
            return node->label();
    }
}

inline void SemanticTransformer::addRequiredInclude(const std::string& include) {
    // Avoid duplicates
    if (std::find(m_required_includes.begin(), m_required_includes.end(), include) == 
        m_required_includes.end()) {
        m_required_includes.push_back(include);
    }
}

inline const std::vector<std::string>& SemanticTransformer::getRequiredIncludes() const {
    return m_required_includes;
}

inline bool SemanticTransformer::validate(const std::string& original, const std::string& transformed) {
    // Basic validation: ensure non-empty transformation where expected
    if (!original.empty() && transformed.empty()) {
        return false;
    }
    
    // More sophisticated validation would go here
    return true;
}

inline std::string SemanticTransformer::applyBestRule(const std::string& input) {
    for (const auto& rule : m_rules) {
        // This is a simplified pattern matching approach
        // A complete implementation would use more sophisticated pattern matching
        if (input.find(rule.cpp2_pattern) != std::string::npos) {
            // Apply the transformation function
            if (rule.transform_fn) {
                // Extract segments and pass to transformation function
                std::vector<std::string> segments = {input};
                return rule.transform_fn(segments);
            }
        }
    }
    
    // If no rule matches, return original
    return input;
}

inline void SemanticTransformer::initializeCommonTransforms() {
    // Function declaration transformation: main: () -> int = { }
    addRule(SemanticTransformRule(
        "main: () -> int = {",
        "int main() {",
        [](const std::vector<std::string>& segments) -> std::string {
            std::string result = segments[0];
            size_t pos = result.find("main: () -> int = {");
            if (pos != std::string::npos) {
                result.replace(pos, 21, "int main() {");  // 21 is length of "main: () -> int = {"
            }
            return result;
        },
        100,  // High priority for main function
        "Transform main function declaration"
    ));
    
    // Variable declaration with explicit type: x: int = 5;
    addRule(SemanticTransformRule(
        ": =",
        " =",
        [](const std::vector<std::string>& segments) -> std::string {
            std::string result = segments[0];
            size_t pos = result.find(": ");
            while (pos != std::string::npos) {
                size_t equals_pos = result.find(" =", pos);
                if (equals_pos != std::string::npos) {
                    size_t type_start = pos + 2;  // After ": "
                    size_t var_len = equals_pos - type_start;
                    std::string var_name = result.substr(0, pos);
                    // Find actual variable name by extracting the last identifier before ":"
                    size_t var_end = pos;
                    size_t var_start = var_end;
                    while (var_start > 0 && result[var_start-1] != ' ' && result[var_start-1] != '\t' && result[var_start-1] != '\n') {
                        var_start--;
                    }
                    std::string var = result.substr(var_start, var_end - var_start);
                    
                    // This is a simplified approach - a full implementation would use proper parsing
                    std::string modified = result;
                    size_t colon_pos = modified.find(": ");
                    if (colon_pos != std::string::npos) {
                        size_t next_space = modified.find(" ", colon_pos + 2);
                        if (next_space != std::string::npos) {
                            std::string type = modified.substr(colon_pos + 2, next_space - (colon_pos + 2));
                            modified.replace(colon_pos - var.length(), type.length() + 3, type + " " + var); // +3 for ": "
                        }
                    }
                    return modified;
                }
                pos = result.find(": ", pos + 1);
            }
            return result;
        },
        80,  // High priority for variable declarations
        "Transform variable declarations (x: type = value)"
    ));
    
    // Auto variable declaration: x := value
    addRule(SemanticTransformRule(
        " := ",
        " = ",
        [](const std::vector<std::string>& segments) -> std::string {
            std::string result = segments[0];
            // Replace "identifier := value" with "auto identifier = value"
            size_t pos = result.find(" := ");
            while (pos != std::string::npos) {
                // Find the start of the identifier
                size_t start = pos;
                while (start > 0 && result[start-1] != ' ' && result[start-1] != '\t' && result[start-1] != '\n') {
                    start--;
                }
                std::string var_name = result.substr(start, pos - start);
                
                // Replace with auto declaration
                std::string prefix = result.substr(0, start);
                std::string suffix = result.substr(pos + 4); // +4 for " := "
                result = prefix + "auto " + var_name + " = " + suffix;
                
                pos = result.find(" := ", pos + 10); // +10 to skip the replacement
            }
            return result;
        },
        85,  // High priority for auto declarations
        "Transform auto variable declarations (x := value)"
    ));
    
    // Parameter mode transformations - inout to reference
    addRule(SemanticTransformRule(
        "inout ",
        "& ",
        [](const std::vector<std::string>& segments) -> std::string {
            std::string result = segments[0];
            size_t pos = result.find("inout ");
            while (pos != std::string::npos) {
                // Find the parameter name after 'inout'
                size_t start = pos + 6; // After 'inout '
                while (start < result.length() && (result[start] == ' ' || result[start] == '\t')) {
                    start++;
                }
                
                size_t end = start;
                while (end < result.length() && result[end] != ':' && result[end] != ',' && result[end] != ')') {
                    end++;
                }
                
                std::string param_name = result.substr(start, end - start);
                
                // Find the type after ':'
                if (end < result.length() && result[end] == ':') {
                    size_t type_start = end + 1;
                    while (type_start < result.length() && (result[type_start] == ' ' || result[type_start] == '\t')) {
                        type_start++;
                    }
                    
                    size_t type_end = type_start;
                    while (type_end < result.length() && result[type_end] != ',' && result[type_end] != ')') {
                        type_end++;
                    }
                    
                    std::string param_type = result.substr(type_start, type_end - type_start);
                    
                    // Replace 'inout param_name: type' with 'type& param_name'
                    std::string prefix = result.substr(0, pos);
                    std::string param_part = param_type + "& " + param_name;
                    std::string suffix = result.substr(type_end);
                    
                    result = prefix + param_part + suffix;
                }
                
                pos = result.find("inout ", pos + 1);
            }
            return result;
        },
        90,  // High priority for parameter transformations
        "Transform inout parameters to reference parameters"
    ));
}

} // namespace cppfort::ir

#endif // CPPFORT_SEMANTIC_TRANSFORMER_H