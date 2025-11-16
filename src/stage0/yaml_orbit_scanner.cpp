#include "yaml_orbit_scanner.h"
#include <fstream>
#include <sstream>

namespace cppfort {
namespace ir {

YamlOrbitScanner::YamlOrbitScanner(const std::filesystem::path& yamlPatternsPath)
    : m_baseScanner(OrbitScannerConfig(yamlPatternsPath.parent_path())),
      m_config(yamlPatternsPath) {
}

YamlOrbitScanner::~YamlOrbitScanner() = default;

bool YamlOrbitScanner::initialize() {
    // Initialize base scanner first
    if (!m_baseScanner.initialize()) {
        return false;
    }
    
    // Load YAML-specific patterns if available
    if (std::filesystem::exists(m_config.yamlPatternsPath)) {
        // Additional YAML-specific initialization can go here
    }
    
    return true;
}

DetectionResult YamlOrbitScanner::scanYaml(const std::string& yamlContent) const {
    // Handle multi-document YAML if enabled
    std::string content = yamlContent;
    if (m_config.handleMultiDocument) {
        auto doc_boundaries = m_yamlScanner.findDocumentBoundaries(yamlContent);
        if (doc_boundaries.size() > 1) {
            // For multi-document, analyze first document for pattern detection
            if (doc_boundaries[0] < yamlContent.length()) {
                size_t first_doc_end = (doc_boundaries.size() > 1) ? doc_boundaries[1] : yamlContent.length();
                content = yamlContent.substr(doc_boundaries[0], first_doc_end - doc_boundaries[0]);
            }
        }
    }
    
    // Perform basic YAML structure validation
    if (m_config.validateSyntax) {
        auto structure = m_yamlScanner.scanYamlStructure(content);
        if (structure.empty() && !content.empty()) {
            DetectionResult result;
            result.confidence = 0.0;
            result.reasoning = "Failed to parse YAML structure";
            return result;
        }
    }
    
    // Apply orbit pattern detection
    DetectionResult result = m_baseScanner.scan(content);
    
    // Boost confidence for valid YAML structures
    if (result.confidence > 0 && m_config.inferTypes) {
        auto values = extractYamlStructure(content);
        if (!values.empty()) {
            result.confidence = std::min(1.0, result.confidence + 0.10);
        }
    }
    
    return result;
}

DetectionResult YamlOrbitScanner::scanYamlFile(const std::filesystem::path& yamlFile) const {
    if (!std::filesystem::exists(yamlFile)) {
        DetectionResult result;
        result.confidence = 0.0;
        result.reasoning = "File not found: " + yamlFile.string();
        return result;
    }
    
    std::ifstream file(yamlFile);
    std::stringstream buffer;
    buffer << file.rdbuf();
    
    return scanYaml(buffer.str());
}

std::vector<cppfort::stage0::YamlTypedValue> YamlOrbitScanner::extractYamlStructure(
    const std::string& yamlContent) const {
    std::vector<cppfort::stage0::YamlTypedValue> result;
    
    try {
        // Try to extract as mapping first (most common)
        auto mapping = m_yamlScanner.extractMapping(yamlContent);
        
        for (const auto& [key, value] : mapping) {
            result.push_back(value);
        }
        
        // If no mapping found, try sequence
        if (result.empty()) {
            auto sequence = m_yamlScanner.extractSequence(yamlContent);
            result = sequence;
        }
    } catch (...) {
        // If parsing fails, return empty result
    }
    
    return result;
}

DetectionResult YamlOrbitScanner::validateJsonYamlConversion(
    const std::string& jsonContent,
    const std::string& yamlContent) const {
    
    DetectionResult result;
    
    // Extract structures from both formats
    auto json_scanner = JsonOrbitScanner(m_config.yamlPatternsPath.parent_path());
    auto json_values = json_scanner.extractJsonStructure(jsonContent);
    auto yaml_values = extractYamlStructure(yamlContent);
    
    // Compare structure counts as basic validation
    if (json_values.size() == yaml_values.size()) {
        result.confidence = std::max(0.85, 1.0 - (std::abs((int)json_values.size() - (int)yaml_values.size()) * 0.1));
        result.reasoning = "Structure count matches between JSON and YAML";
    } else {
        result.confidence = 0.5;
        result.reasoning = "Structure count mismatch: JSON has " + 
                          std::to_string(json_values.size()) + 
                          " values, YAML has " + 
                          std::to_string(yaml_values.size());
    }
    
    return result;
}

// Standalone function
DetectionResult detectYamlPatterns(const std::string& yamlContent,
                                   const std::filesystem::path& patternsDir) {
    YamlOrbitScanner scanner(patternsDir);
    if (!scanner.initialize()) {
        DetectionResult result;
        result.confidence = 0.0;
        result.reasoning = "Failed to initialize scanner";
        return result;
    }
    return scanner.scanYaml(yamlContent);
}

// JsonYamlValidationScanner implementation
JsonYamlValidationScanner::JsonYamlValidationScanner(const std::filesystem::path& jsonPatternsPath,
                                                     const std::filesystem::path& yamlPatternsPath)
    : m_jsonScanner(jsonPatternsPath),
      m_yamlScanner(yamlPatternsPath) {
}

DetectionResult JsonYamlValidationScanner::validateEquivalence(
    const std::string& jsonContent,
    const std::string& yamlContent) const {
    
    return m_yamlScanner.validateJsonYamlConversion(jsonContent, yamlContent);
}

} // namespace ir
} // namespace cppfort
