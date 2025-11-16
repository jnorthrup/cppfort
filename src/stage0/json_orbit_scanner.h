#pragma once

#include <string>
#include <vector>
#include <memory>
#include <filesystem>
#include "orbit_scanner.h"
#include "json_scanner.h"

namespace cppfort {
namespace ir {

/**
 * Orbit scanner specialized for JSON pattern detection
 * 
 * This scanner uses the orbit-based pattern matching system
 * to detect and validate JSON structures in pattern files.
 */
class JsonOrbitScanner {
private:
    OrbitScanner m_baseScanner;
    cppfort::stage0::JsonScanner m_jsonScanner;
    
    struct JsonOrbitConfig {
        std::filesystem::path jsonPatternsPath;
        double jsonConfidenceThreshold = 0.75;
        bool validateSyntax = true;
        bool inferTypes = true;
        
        JsonOrbitConfig() = default;
        JsonOrbitConfig(const std::filesystem::path& path)
            : jsonPatternsPath(path) {}
    };
    
    JsonOrbitConfig m_config;
    
public:
    /**
     * Constructor - uses the same base scanner but with JSON patterns
     * @param jsonPatternsPath Directory containing JSON-specific orbit patterns
     */
    explicit JsonOrbitScanner(const std::filesystem::path& jsonPatternsPath);
    
    /**
     * Destructor
     */
    ~JsonOrbitScanner();
    
    /**
     * Initialize with JSON-specific patterns
     * @return true if successful
     */
    bool initialize();
    
    /**
     * Scan JSON content using orbit patterns
     * @param jsonContent JSON text to scan
     * @return Detection result with grammar type and confidence
     */
    DetectionResult scanJson(const std::string& jsonContent) const;
    
    /**
     * Scan a JSON file and validate against orbit patterns
     * @param jsonFile Path to JSON file
     * @return Detection result with validation info
     */
    DetectionResult scanJsonFile(const std::filesystem::path& jsonFile) const;
    
    /**
     * Extract JSON structure with orbit evidence
     * @param jsonContent JSON text to analyze
     * @return Vector of JsonTypedValue with orbit metadata
     */
    std::vector<cppfort::stage0::JsonTypedValue> extractJsonStructure(
        const std::string& jsonContent) const;
    
    /**
     * Get base scanner for advanced operations
     */
    const OrbitScanner& getBaseScanner() const { return m_baseScanner; }
    
    /**
     * Get configuration
     */
    const JsonOrbitConfig& getConfig() const { return m_config; }
    
    /**
     * Update configuration
     */
    void updateConfig(const JsonOrbitConfig& config) { m_config = config; }
};

/**
 * Standalone function for quick JSON pattern detection
 * Useful for integration with other scanners
 */
DetectionResult detectJsonPatterns(const std::string& jsonContent,
                                   const std::filesystem::path& patternsDir);

} // namespace ir
} // namespace cppfort
