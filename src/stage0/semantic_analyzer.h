#ifndef CPPFORT_SEMANTIC_ANALYZER_H
#define CPPFORT_SEMANTIC_ANALYZER_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <sstream>
#include <list>

#include "node.h"
#include "type.h"
#include "orbit_mask.h"
#include "graph_semantic_transformer.h"

namespace cppfort::ir {

/**
 * @brief Represents a semantic construct in Cpp2 code
 */
struct SemanticConstruct {
    std::string type;                    // Type of construct (function, variable, type, etc.)
    std::string name;                    // Name of the construct (if applicable)
    std::string signature;               // Full signature
    std::vector<std::string> parameters; // Parameters (for functions)
    std::string returnType;              // Return type (for functions)
    std::string definition;              // Full definition
    size_t line;                         // Line number in source
    size_t column;                       // Column number in source
    std::string scope;                   // Scope where defined
    
    SemanticConstruct(const std::string& t, const std::string& n, const std::string& sig, size_t l, size_t c)
        : type(t), name(n), signature(sig), line(l), column(c) {}
};

/**
 * @brief Semantic Analysis for Cpp2 code
 * 
 * This class performs semantic analysis of Cpp2 code to identify constructs
 * and their relationships, which is necessary for proper transformation.
 */
class SemanticAnalyzer {
private:
    std::vector<SemanticConstruct> m_constructs;
    std::unordered_map<std::string, std::vector<size_t>> m_construct_indices; // name -> indices in m_constructs
    std::unordered_map<std::string, std::string> m_type_mappings;
    std::unordered_map<std::string, std::string> m_scope_hierarchy;
    
public:
    SemanticAnalyzer();
    
    /**
     * @brief Analyze Cpp2 source code and identify semantic constructs
     */
    std::vector<SemanticConstruct> analyze(const std::string& source);
    
    /**
     * @brief Add a type mapping for semantic analysis
     */
    void addTypeMapping(const std::string& cpp2_type, const std::string& cpp_type);
    
    /**
     * @brief Get all identified constructs
     */
    const std::vector<SemanticConstruct>& getConstructs() const;
    
    /**
     * @brief Find constructs by type
     */
    std::vector<SemanticConstruct> findConstructsByType(const std::string& type) const;
    
    /**
     * @brief Find constructs by name
     */
    std::vector<SemanticConstruct> findConstructsByName(const std::string& name) const;
    
    /**
     * @brief Get constructs in a specific scope
     */
    std::vector<SemanticConstruct> getConstructsInScope(const std::string& scope) const;
    
    /**
     * @brief Analyze function declarations specifically
     */
    std::vector<SemanticConstruct> analyzeFunctions(const std::string& source);
    
    /**
     * @brief Analyze variable declarations specifically
     */
    std::vector<SemanticConstruct> analyzeVariables(const std::string& source);
    
    /**
     * @brief Analyze type definitions specifically
     */
    std::vector<SemanticConstruct> analyzeTypes(const std::string& source);
    
    /**
     * @brief Check for semantic errors in the code
     */
    std::vector<std::string> checkSemantics(const std::string& source);
    
    /**
     * @brief Resolve types in the context of the analyzed constructs
     */
    std::string resolveType(const std::string& type_name, const std::string& context = "") const;
    
private:
    /**
     * @brief Parse a function declaration from source text
     */
    std::vector<SemanticConstruct> parseFunctionDeclarations(const std::string& source);
    
    /**
     * @brief Parse variable declarations from source text
     */
    std::vector<SemanticConstruct> parseVariableDeclarations(const std::string& source);
    
    /**
     * @brief Parse type definitions from source text
     */
    std::vector<SemanticConstruct> parseTypeDefinitions(const std::string& source);
    
    /**
     * @brief Extract tokens from source for analysis
     */
    std::vector<std::string> tokenize(const std::string& source);
    
    /**
     * @brief Determine the scope of a construct based on context
     */
    std::string determineScope(size_t line, size_t column, const std::string& source);
};

inline SemanticAnalyzer::SemanticAnalyzer() {
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
}

inline void SemanticAnalyzer::addTypeMapping(const std::string& cpp2_type, const std::string& cpp_type) {
    m_type_mappings[cpp2_type] = cpp_type;
}

inline std::vector<SemanticConstruct> SemanticAnalyzer::analyze(const std::string& source) {
    m_constructs.clear();
    
    // Parse different types of constructs
    auto functions = parseFunctionDeclarations(source);
    auto variables = parseVariableDeclarations(source);
    auto types = parseTypeDefinitions(source);
    
    // Combine all constructs
    m_constructs.insert(m_constructs.end(), functions.begin(), functions.end());
    m_constructs.insert(m_constructs.end(), variables.begin(), variables.end());
    m_constructs.insert(m_constructs.end(), types.begin(), types.end());
    
    // Sort by line and column for consistent processing
    std::sort(m_constructs.begin(), m_constructs.end(), 
              [](const SemanticConstruct& a, const SemanticConstruct& b) {
                  if (a.line != b.line) return a.line < b.line;
                  return a.column < b.column;
              });
    
    return m_constructs;
}

inline const std::vector<SemanticConstruct>& SemanticAnalyzer::getConstructs() const {
    return m_constructs;
}

inline std::vector<SemanticConstruct> SemanticAnalyzer::findConstructsByType(const std::string& type) const {
    std::vector<SemanticConstruct> result;
    for (const auto& construct : m_constructs) {
        if (construct.type == type) {
            result.push_back(construct);
        }
    }
    return result;
}

inline std::vector<SemanticConstruct> SemanticAnalyzer::findConstructsByName(const std::string& name) const {
    std::vector<SemanticConstruct> result;
    for (const auto& construct : m_constructs) {
        if (construct.name == name) {
            result.push_back(construct);
        }
    }
    return result;
}

inline std::vector<SemanticConstruct> SemanticAnalyzer::getConstructsInScope(const std::string& scope) const {
    std::vector<SemanticConstruct> result;
    for (const auto& construct : m_constructs) {
        if (construct.scope == scope) {
            result.push_back(construct);
        }
    }
    return result;
}

inline std::vector<SemanticConstruct> SemanticAnalyzer::analyzeFunctions(const std::string& source) {
    return parseFunctionDeclarations(source);
}

inline std::vector<SemanticConstruct> SemanticAnalyzer::analyzeVariables(const std::string& source) {
    return parseVariableDeclarations(source);
}

inline std::vector<SemanticConstruct> SemanticAnalyzer::analyzeTypes(const std::string& source) {
    return parseTypeDefinitions(source);
}

inline std::vector<std::string> SemanticAnalyzer::checkSemantics(const std::string& source) {
    std::vector<std::string> errors;
    
    // Perform various semantic checks
    for (const auto& construct : m_constructs) {
        // Check for undefined types
        if (construct.type == "variable" || construct.type == "function") {
            if (m_type_mappings.find(construct.returnType) == m_type_mappings.end() && 
                construct.returnType != "auto") {
                errors.push_back("Undefined type: " + construct.returnType + 
                               " at line " + std::to_string(construct.line));
            }
        }
    }
    
    // Add more semantic checks as needed
    
    return errors;
}

inline std::string SemanticAnalyzer::resolveType(const std::string& type_name, const std::string& context) const {
    auto it = m_type_mappings.find(type_name);
    if (it != m_type_mappings.end()) {
        return it->second;
    }
    // If not found in mappings, return the original type name
    return type_name;
}

inline std::vector<SemanticConstruct> SemanticAnalyzer::parseFunctionDeclarations(const std::string& source) {
    std::vector<SemanticConstruct> functions;
    size_t line_num = 1;
    
    size_t pos = 0;
    while (pos < source.length()) {
        // Look for function pattern: name: (params) -> return_type = { ... }
        size_t func_pos = source.find(": (", pos);
        if (func_pos == std::string::npos) break;
        
        // Find the function name (go backwards from ": (")
        size_t name_start = func_pos;
        while (name_start > 0 && source[name_start - 1] != ' ' && 
               source[name_start - 1] != '\t' && source[name_start - 1] != '\n') {
            name_start--;
        }
        
        std::string name = source.substr(name_start, func_pos - name_start);
        
        // Find the end of parameters and return type
        size_t return_pos = source.find(") -> ", func_pos);
        if (return_pos != std::string::npos) {
            size_t return_end = source.find(" = {", return_pos);
            if (return_end != std::string::npos) {
                std::string return_type = source.substr(return_pos + 5, return_end - (return_pos + 5));
                
                // Count newlines to determine line number
                size_t line_count = 1;
                for (size_t i = 0; i < name_start; ++i) {
                    if (source[i] == '\n') line_count++;
                }
                
                SemanticConstruct func("function", name, 
                                     source.substr(name_start, return_end + 4 - name_start),
                                     line_count, name_start);
                func.returnType = return_type;
                func.scope = determineScope(line_count, name_start, source);
                
                functions.push_back(func);
            }
        }
        
        pos = func_pos + 1;
    }
    
    return functions;
}

inline std::vector<SemanticConstruct> SemanticAnalyzer::parseVariableDeclarations(const std::string& source) {
    std::vector<SemanticConstruct> variables;
    
    size_t pos = 0;
    while (pos < source.length()) {
        // Look for variable pattern: name: type = value; or name := value;
        size_t colon_pos = source.find(": ", pos);
        size_t auto_pos = source.find(" := ", pos);
        
        size_t var_pos = std::string::npos;
        bool is_auto = false;
        
        if (colon_pos != std::string::npos && (auto_pos == std::string::npos || colon_pos < auto_pos)) {
            var_pos = colon_pos;
            is_auto = false;
        } else if (auto_pos != std::string::npos) {
            var_pos = auto_pos;
            is_auto = true;
        }
        
        if (var_pos == std::string::npos) break;
        
        // Find the variable name
        size_t name_start = var_pos;
        while (name_start > 0 && source[name_start - 1] != ' ' && 
               source[name_start - 1] != '\t' && source[name_start - 1] != '\n') {
            name_start--;
        }
        
        std::string name = source.substr(name_start, var_pos - name_start);
        
        // Count newlines to determine line number
        size_t line_count = 1;
        for (size_t i = 0; i < name_start; ++i) {
            if (source[i] == '\n') line_count++;
        }
        
        // Determine the type/initialization
        std::string definition;
        if (!is_auto) {
            // Find the end of the type declaration
            size_t equals_pos = source.find(" = ", var_pos);
            if (equals_pos != std::string::npos) {
                size_t end_pos = source.find(";", equals_pos);
                if (end_pos != std::string::npos) {
                    definition = source.substr(name_start, end_pos + 1 - name_start);
                }
            }
        } else {
            // For auto variables, find the end
            size_t end_pos = source.find(";", auto_pos);
            if (end_pos != std::string::npos) {
                definition = source.substr(name_start, end_pos + 1 - name_start);
            }
        }
        
        SemanticConstruct var("variable", name, definition, line_count, name_start);
        var.scope = determineScope(line_count, name_start, source);
        
        variables.push_back(var);
        pos = var_pos + 1;
    }
    
    return variables;
}

inline std::vector<SemanticConstruct> SemanticAnalyzer::parseTypeDefinitions(const std::string& source) {
    std::vector<SemanticConstruct> types;
    
    // This is a simplified implementation
    // A full implementation would need more sophisticated parsing
    
    size_t pos = 0;
    while (pos < source.length()) {
        // Look for type pattern: Name: type = { ... }
        size_t type_pos = source.find(": type = {", pos);
        if (type_pos == std::string::npos) break;
        
        // Find the type name (go backwards from ": type = {")
        size_t name_start = type_pos;
        while (name_start > 0 && source[name_start - 1] != ' ' && 
               source[name_start - 1] != '\t' && source[name_start - 1] != '\n') {
            name_start--;
        }
        
        std::string name = source.substr(name_start, type_pos - name_start);
        
        // Count newlines to determine line number
        size_t line_count = 1;
        for (size_t i = 0; i < name_start; ++i) {
            if (source[i] == '\n') line_count++;
        }
        
        // Find the end of the type definition
        size_t brace_count = 1;
        size_t end_pos = type_pos + 10; // After ": type = {"
        while (end_pos < source.length() && brace_count > 0) {
            if (source[end_pos] == '{') brace_count++;
            else if (source[end_pos] == '}') brace_count--;
            end_pos++;
        }
        
        if (brace_count == 0) {
            std::string definition = source.substr(name_start, end_pos - name_start);
            
            SemanticConstruct type("type", name, definition, line_count, name_start);
            type.scope = determineScope(line_count, name_start, source);
            
            types.push_back(type);
        }
        
        pos = type_pos + 1;
    }
    
    return types;
}

inline std::vector<std::string> SemanticAnalyzer::tokenize(const std::string& source) {
    std::vector<std::string> tokens;
    std::string current_token;
    
    for (char c : source) {
        if (std::isalnum(c) || c == '_') {
            current_token += c;
        } else {
            if (!current_token.empty()) {
                tokens.push_back(current_token);
                current_token.clear();
            }
            if (!std::isspace(c)) {
                tokens.push_back(std::string(1, c));
            }
        }
    }
    
    if (!current_token.empty()) {
        tokens.push_back(current_token);
    }
    
    return tokens;
}

inline std::string SemanticAnalyzer::determineScope(size_t line, size_t column, const std::string& source) {
    // Simplified scope determination
    // A full implementation would track scope hierarchies properly
    return "global";
}

} // namespace cppfort::ir

#endif // CPPFORT_SEMANTIC_ANALYZER_H