#pragma once

#include <string>
#include <vector>
#include <memory>
#include <filesystem>
#include "orbit_scanner.h"
#include "yaml_scanner.h"

namespace cppfort {
namespace ir {

/**
 * Orbit scanner specialized for YAML pattern detection
 * 
 * This scanner uses the orbit-based pattern matching system
 * to detect and validate YAML structures in pattern files and configuration files.
 */
class YamlOrbitScanner {
private:
    OrbitScanner m_baseScanner;
    cppfort::stage0::YamlScanner m_yamlScanner;
    
    struct YamlOrbitConfig {
        std::filesystem::path yamlPatternsPath;
        double yamlConfidenceThreshold = 0.70;
        bool validateSyntax = true;
        bool inferTypes = true;
        bool handleMultiDocument = false;
        
        YamlOrbitConfig() = default;
        YamlOrbitConfig(const std::filesystem::path& path)
            : yamlPatternsPath(path) {}
    };
    
    YamlOrbitConfig m_config;
    
public:
    /**
     * Constructor - uses the same base scanner but with YAML patterns
     * @param yamlPatternsPath Directory containing YAML-specific orbit patterns
     */
    explicit YamlOrbitScanner(const std::filesystem::path& yamlPatternsPath);
    
    /**
     * Destructor
     */
    ~YamlOrbitScanner();
    
    /**
     * Initialize with YAML-specific patterns
     * @return true if successful
     */
    bool initialize();
    
    /**
     * Scan YAML content using orbit patterns
     * @param yamlContent YAML text to scan
     * @return Detection result with grammar type and confidence
     */
    DetectionResult scanYaml(const std::string& yamlContent) const;
    
    /**
     * Scan a YAML file and validate against orbit patterns
     * @param yamlFile Path to YAML file
     * @return Detection result with validation info
     */
    DetectionResult scanYamlFile(const std::filesystem::path& yamlFile) const;
    
    /**
     * Extract YAML structure with orbit evidence
     * @param yamlContent YAML text to analyze
     * @return Vector of YamlTypedValue with orbit metadata
     */
    std::vector<cppfort::stage0::YamlTypedValue> extractYamlStructure(
        const std::string& yamlContent) const;
    
    /**
     * Validate JSON-to-YAML conversion using orbit patterns
     * @param jsonContent Original JSON
     * @param yamlContent Converted YAML
     * @return Validation result with confidence scores
     */
    DetectionResult validateJsonYamlConversion(
        const std::string& jsonContent,
        const std::string& yamlContent) const;
    
    /**
     * Get base scanner for advanced operations
     */
    const OrbitScanner& getBaseScanner() const { return m_baseScanner; }
    
    /**
     * Get configuration
     */
    const YamlOrbitConfig& getConfig() const { return m_config; }
    
    /**
     * Update configuration
     */
    void updateConfig(const YamlOrbitConfig& config) { m_config = config; }
};

/**
 * Standalone function for quick YAML pattern detection
 * Useful for integration with other scanners
 */
DetectionResult detectYamlPatterns(const std::string& yamlContent,
                                   const std::filesystem::path& patternsDir);

/**
 * Combined scanner that validates JSON↔YAML conversions
 * Used by the two-way converter to ensure semantic preservation
 */
class JsonYamlValidationScanner {
private:
    JsonOrbitScanner m_jsonScanner;
    YamlOrbitScanner m_yamlScanner;
    
public:
    JsonYamlValidationScanner(const std::filesystem::path& jsonPatternsPath,
                             const std::filesystem::path& yamlPatternsPath);
    
    /**
     * Validate that JSON and YAML represent the same structure
     * @param jsonContent JSON text
     * @param yamlContent YAML text
     * @return Validation result with confidence and differences
     */
    DetectionResult validateEquivalence(const std::string& jsonContent,
                                       const std::string& yamlContent) const;
};

} // namespace ir
} // namespace cppfort
