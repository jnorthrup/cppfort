#include "json_orbit_scanner.h"
#include <fstream>
#include <sstream>

namespace cppfort {
namespace ir {

JsonOrbitScanner::JsonOrbitScanner(const std::filesystem::path& jsonPatternsPath)
        : m_baseScanner(OrbitScannerConfig(jsonPatternsPath.parent_path())),
            m_jsonScanner(std::string_view("")),
            m_config(jsonPatternsPath) {
}

JsonOrbitScanner::~JsonOrbitScanner() = default;

bool JsonOrbitScanner::initialize() {
    // Initialize base scanner first
    if (!m_baseScanner.initialize()) {
        return false;
    }
    
    // Load JSON-specific patterns if available
    if (std::filesystem::exists(m_config.jsonPatternsPath)) {
        // Additional JSON-specific initialization can go here
    }
    
    return true;
}

DetectionResult JsonOrbitScanner::scanJson(const std::string& jsonContent) const {
    // First perform basic JSON structure validation
    if (m_config.validateSyntax) {
        // Use a temporary JsonScanner to analyze the given content
        cppfort::stage0::JsonScanner scanner(jsonContent);
        auto doc = scanner.scan();
        if (doc.first.first.empty() && !jsonContent.empty()) {
            DetectionResult result;
            result.confidence = 0.0;
            result.reasoning = "Failed to parse JSON structure";
            return result;
        }
    }
    
    // Then apply orbit pattern detection
    DetectionResult result = m_baseScanner.scan(jsonContent);
    
    // Boost confidence for valid JSON structures
    if (result.confidence > 0 && m_config.inferTypes) {
        auto values = extractJsonStructure(jsonContent);
        if (!values.empty()) {
            result.confidence = std::min(1.0, result.confidence + 0.15);
        }
    }
    
    return result;
}

DetectionResult JsonOrbitScanner::scanJsonFile(const std::filesystem::path& jsonFile) const {
    if (!std::filesystem::exists(jsonFile)) {
        DetectionResult result;
        result.confidence = 0.0;
        result.reasoning = "File not found: " + jsonFile.string();
        return result;
    }
    
    std::ifstream file(jsonFile);
    std::stringstream buffer;
    buffer << file.rdbuf();
    
    return scanJson(buffer.str());
}

std::vector<cppfort::stage0::JsonTypedValue> JsonOrbitScanner::extractJsonStructure(
    const std::string& jsonContent) const {
    std::vector<cppfort::stage0::JsonTypedValue> result;
    
    // Create document for parsing
    cppfort::stage0::JsonDocument document;
        cppfort::stage0::JsonScanner scanner(jsonContent);
        auto doc = scanner.scan();
        // Convert to JsonTypedValue set by extracting values at each index position
        auto& index_positions = doc.second.first;
        for (auto pos : index_positions) {
            cppfort::stage0::JsonTypedValue v;
            v.evidence = cppfort::stage0::JsonTypeEvidence();
            v.value = scanner.extract_value(static_cast<size_t>(pos));
            result.push_back(v);
        }
    
    // Add orbit confidence to each value
    for (auto& value : result) {
        value.evidence.depth = static_cast<int32_t>(m_config.jsonConfidenceThreshold * 100.0);
    }
    
    return result;
}

// Standalone function
DetectionResult detectJsonPatterns(const std::string& jsonContent,
                                   const std::filesystem::path& patternsDir) {
    JsonOrbitScanner scanner(patternsDir);
    if (!scanner.initialize()) {
        DetectionResult result;
        result.confidence = 0.0;
        result.reasoning = "Failed to initialize scanner";
        return result;
    }
    return scanner.scanJson(jsonContent);
}

} // namespace ir
} // namespace cppfort
