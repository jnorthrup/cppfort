#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <filesystem>

namespace cppfort::ir {

/**
 * Chapter 19: Orbit Pattern Structure
 *
 * Defines patterns for language/protocol detection.
 * Based on professional_orbit_scanner.cpp2 reference implementation.
 */
struct OrbitPattern {
    ::std::string name;                           // Pattern name ("cpp20_concepts")
    uint32_t orbit_id;                          // Orbit identifier
    ::std::vector<::std::string> signature_patterns; // Keywords/markers
    ::std::vector<::std::string> protocol_indicators; // Protocol hints
    ::std::vector<::std::string> version_patterns;   // Version strings
    double weight;                              // Pattern importance (0.0-1.0)

    // Depth context for match validity: -1 = any depth, >=0 = specific total depth
    int expected_depth = -1;
    // Required confix context: which delimiter must be active (empty = none)
    ::std::string required_confix;  // "{", "(", "[", "<", "\"" or ""

    OrbitPattern() = default;
    OrbitPattern(const ::std::string& n, uint32_t id, double w = 1.0)
        : name(n), orbit_id(id), weight(w) {}
};

/**
 * Chapter 19: Pattern Database
 *
 * Stores and queries orbit patterns for language detection.
 * Supports YAML-based pattern loading and TableGen export.
 * Based on orbit patterns from ororoboros-couchduck.
 */
class PatternDatabase {
private:
    ::std::unordered_map<::std::string, OrbitPattern> _patterns;
    ::std::unordered_map<uint32_t, ::std::vector<OrbitPattern>> _patternsByOrbit;

    /**
     * Load patterns from YAML file.
     */
    bool loadYamlFile(const ::std::string& filepath);

    /**
     * Parse single pattern from YAML node.
     */
    ::std::optional<OrbitPattern> parsePattern(const ::std::string& yamlContent);

public:
    /**
     * Load patterns from directory containing YAML files.
     * Expected structure: patterns/c_patterns.yaml, patterns/cpp_patterns.yaml, etc.
     */
    bool loadFromDirectory(const ::std::string& directoryPath);

    /**
     * Load patterns from single YAML file.
     */
      bool loadFromYaml(const ::std::filesystem::path& filePath);

    /**
     * Get pattern by name.
     */
    ::std::optional<OrbitPattern> getPattern(const ::std::string& name) const;

    /**
     * Get all patterns for specific orbit ID.
     */
    ::std::vector<OrbitPattern> getPatternsForOrbit(uint32_t orbitId) const;

    /**
     * Get all pattern names.
     */
    ::std::vector<::std::string> getPatternNames() const;

    /**
     * Check if pattern exists.
     */
    bool hasPattern(const ::std::string& name) const;

    /**
     * Export patterns to TableGen format for MLIR dialect generation.
     */
    ::std::string exportToTableGen(const ::std::string& dialectName) const;

    /**
     * Get all patterns.
     */
    ::std::vector<OrbitPattern> getPatterns() const;

    /**
     * Get number of patterns.
     */
    size_t size() const;

    /**
     * Clear all patterns.
     */
    void clear();
};

} // namespace cppfort::ir