#ifndef CPPFORT_UNIFIED_SEMANTIC_TRANSFORMATION_ENGINE_H
#define CPPFORT_UNIFIED_SEMANTIC_TRANSFORMATION_ENGINE_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>
#include <sstream>
#include <iostream>
#include <fstream>

#include "node.h"
#include "type.h"
#include "orbit_mask.h"
#include "semantic_transformer.h"
#include "graph_semantic_transformer.h"
#include "semantic_analyzer.h"

namespace cppfort::ir {

/**
 * @brief Result of a semantic transformation operation
 */
struct TransformationResult {
    std::string transformed_code;
    std::vector<std::string> warnings;
    std::vector<std::string> errors;
    bool success;
    
    TransformationResult() : success(false) {}
};

/**
 * @brief Unified Semantic Transformation Engine
 * 
 * This is the main orchestrator for the complete end-to-end semantic transformation system.
 * It integrates semantic analysis, graph-based transformations, and validation to provide
 * a comprehensive transformation pipeline from Cpp2 to C++.
 */
class UnifiedSemanticTransformationEngine {
private:
    std::unique_ptr<SemanticAnalyzer> m_analyzer;
    std::unique_ptr<GraphSemanticTransformer> m_transformer;
    std::vector<std::string> m_required_includes;
    std::unordered_map<std::string, std::string> m_type_mappings;
    
public:
    UnifiedSemanticTransformationEngine();
    ~UnifiedSemanticTransformationEngine() = default;
    
    /**
     * @brief Transform Cpp2 source code to C++
     */
    TransformationResult transformCpp2ToCpp(const std::string& cpp2_source);
    
    /**
     * @brief Transform C++ source code back to Cpp2 (for verification)
     */
    TransformationResult transformCppToCpp2(const std::string& cpp_source);
    
    /**
     * @brief Analyze Cpp2 source for semantic constructs
     */
    std::vector<SemanticConstruct> analyzeSource(const std::string& source);
    
    /**
     * @brief Add a type mapping to the transformation system
     */
    void addTypeMapping(const std::string& cpp2_type, const std::string& cpp_type);
    
    /**
     * @brief Add a required include for the transformations
     */
    void addRequiredInclude(const std::string& include);
    
    /**
     * @brief Get all required includes for the transformations
     */
    const std::vector<std::string>& getRequiredIncludes() const;
    
    /**
     * @brief Validate that a transformation preserves semantics
     */
    bool validateTransformation(const std::string& original, const std::string& transformed);
    
    /**
     * @brief Get transformation statistics
     */
    std::unordered_map<std::string, size_t> getTransformationStats() const;
    
    /**
     * @brief Process a complete Cpp2 file to C++
     */
    bool processFile(const std::string& input_path, const std::string& output_path);
    
private:
    /**
     * @brief Add default type mappings
     */
    void initializeTypeMappings();
    
    /**
     * @brief Generate required includes for the output
     */
    std::string generateIncludes() const;
    
    /**
     * @brief Apply analysis-based transformations
     */
    std::string applyAnalysisBasedTransformations(
        const std::string& source,
        const std::vector<SemanticConstruct>& constructs);
};

inline UnifiedSemanticTransformationEngine::UnifiedSemanticTransformationEngine() {
    m_analyzer = std::make_unique<SemanticAnalyzer>();
    m_transformer = std::make_unique<GraphSemanticTransformer>();
    
    // Initialize with default mappings
    initializeTypeMappings();
}

inline TransformationResult UnifiedSemanticTransformationEngine::transformCpp2ToCpp(
    const std::string& cpp2_source) {
    
    TransformationResult result;
    
    try {
        // Step 1: Analyze the source for semantic constructs
        auto constructs = analyzeSource(cpp2_source);
        
        // Step 2: Apply graph-based transformations
        std::string transformed = m_transformer->transformCpp2ToCpp(cpp2_source);
        
        // Step 3: Apply analysis-based transformations
        transformed = applyAnalysisBasedTransformations(transformed, constructs);
        
        // Step 4: Validate the transformation
        if (validateTransformation(cpp2_source, transformed)) {
            result.transformed_code = transformed;
            result.success = true;
        } else {
            result.errors.push_back("Transformation validation failed");
            result.success = false;
        }
    } catch (const std::exception& e) {
        result.errors.push_back(std::string("Transformation error: ") + e.what());
        result.success = false;
    }
    
    return result;
}

inline TransformationResult UnifiedSemanticTransformationEngine::transformCppToCpp2(
    const std::string& cpp_source) {
    
    TransformationResult result;
    
    try {
        // Apply reverse transformation
        std::string transformed = m_transformer->transformCppToCpp2(cpp_source);
        
        result.transformed_code = transformed;
        result.success = true;
    } catch (const std::exception& e) {
        result.errors.push_back(std::string("Reverse transformation error: ") + e.what());
        result.success = false;
    }
    
    return result;
}

inline std::vector<SemanticConstruct> UnifiedSemanticTransformationEngine::analyzeSource(
    const std::string& source) {
    return m_analyzer->analyze(source);
}

inline void UnifiedSemanticTransformationEngine::addTypeMapping(
    const std::string& cpp2_type, 
    const std::string& cpp_type) {
    m_type_mappings[cpp2_type] = cpp_type;
    m_analyzer->addTypeMapping(cpp2_type, cpp_type);
}

inline void UnifiedSemanticTransformationEngine::addRequiredInclude(const std::string& include) {
    if (std::find(m_required_includes.begin(), m_required_includes.end(), include) == 
        m_required_includes.end()) {
        m_required_includes.push_back(include);
    }
    m_transformer->addRequiredInclude(include);
}

inline const std::vector<std::string>& UnifiedSemanticTransformationEngine::getRequiredIncludes() const {
    return m_required_includes;
}

inline bool UnifiedSemanticTransformationEngine::validateTransformation(
    const std::string& original, 
    const std::string& transformed) {
    
    // Basic validation: ensure non-empty result where expected
    if (!original.empty() && transformed.empty()) {
        return false;
    }
    
    // More sophisticated validation would go here
    // For example, checking that functions have proper signatures,
    // variables have proper types, etc.
    
    return true;
}

inline std::unordered_map<std::string, size_t> UnifiedSemanticTransformationEngine::getTransformationStats() const {
    std::unordered_map<std::string, size_t> stats;
    
    // This would include statistics about:
    // - Number of functions transformed
    // - Number of variables transformed
    // - Number of types transformed
    // - etc.
    
    return stats;
}

inline bool UnifiedSemanticTransformationEngine::processFile(
    const std::string& input_path, 
    const std::string& output_path) {
    
    // Read input file
    std::ifstream input_file(input_path);
    if (!input_file.is_open()) {
        std::cerr << "Error: Could not open input file " << input_path << std::endl;
        return false;
    }
    
    std::string source((std::istreambuf_iterator<char>(input_file)),
                       std::istreambuf_iterator<char>());
    input_file.close();
    
    // Transform the source
    auto result = transformCpp2ToCpp(source);
    
    if (!result.success) {
        std::cerr << "Transformation failed:" << std::endl;
        for (const auto& error : result.errors) {
            std::cerr << "  " << error << std::endl;
        }
        return false;
    }
    
    // Write output file
    std::ofstream output_file(output_path);
    if (!output_file.is_open()) {
        std::cerr << "Error: Could not open output file " << output_path << std::endl;
        return false;
    }
    
    output_file << result.transformed_code;
    output_file.close();
    
    return true;
}

inline void UnifiedSemanticTransformationEngine::initializeTypeMappings() {
    // Initialize with common Cpp2 to C++ type mappings
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
    
    // Add required includes
    addRequiredInclude("cstdint");
    addRequiredInclude("string");
    addRequiredInclude("vector");
    addRequiredInclude("memory");
}

inline std::string UnifiedSemanticTransformationEngine::generateIncludes() const {
    std::string includes;
    for (const auto& inc : m_required_includes) {
        includes += "#include <" + inc + ">\n";
    }
    if (!m_required_includes.empty()) {
        includes += "\n";
    }
    return includes;
}

inline std::string UnifiedSemanticTransformationEngine::applyAnalysisBasedTransformations(
    const std::string& source,
    const std::vector<SemanticConstruct>& constructs) {
    
    std::string result = source;
    
    // Apply transformations based on semantic analysis
    for (const auto& construct : constructs) {
        switch (construct.type[0]) {  // Using first char as a simple way to categorize
            case 'f': // function
                // Apply function-specific transformations
                break;
            case 'v': // variable
                // Apply variable-specific transformations
                break;
            case 't': // type
                // Apply type-specific transformations
                break;
            default:
                break;
        }
    }
    
    return result;
}

} // namespace cppfort::ir

#endif // CPPFORT_UNIFIED_SEMANTIC_TRANSFORMATION_ENGINE_H