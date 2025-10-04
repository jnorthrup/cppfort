#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <filesystem>

namespace cppfort {
namespace ir {

/**
 * Chapter 19: Orbit Pattern Structure
 *
 * Defines patterns for language/protocol detection.
 * Based on professional_orbit_scanner.cpp2 reference implementation.
 */
struct OrbitPattern {
    ::std::string name;                           // Pattern name ("cpp20_concepts")
    uint32_t orbit_id;                            // Orbit identifier
    ::std::vector<::std::string> signature_patterns; // Keywords/markers
    ::std::vector<::std::string> protocol_indicators; // Protocol hints
    ::std::vector<::std::string> version_patterns;   // Version strings
    double weight = 1.0;                          // Pattern importance (0.0-1.0)

    // Depth context for match validity: -1 = any depth, >=0 = specific total depth
    int expected_depth = -1;

    // Legacy: Required confix context as simple string (keeps backward compatibility)
    // Valid values: "{", "(", "[", "<", "\"" or empty = none
    ::std::string required_confix;

    // Confix visibility mask - controls where this pattern is allowed to match.
    // Bitfield uses ConfixMask values below. Default allows all confixes (visible everywhere).
    enum ConfixMask : uint8_t {
        TopLevel  = 1 << 0,  // depth 0 (no open delimiters)
        InBrace   = 1 << 1,  // inside {}
        InParen   = 1 << 2,  // inside ()
        InAngle   = 1 << 3,  // inside <>
        InBracket = 1 << 4,  // inside []
        InQuote   = 1 << 5,  // inside ""
    };
    uint8_t confix_mask = 0x3F; // Default: all six bits set

    // N-way grammar support: multiple language modes for concurrent orbit detection
    enum GrammarMode : uint8_t {
        C     = 1 << 0,  // 0x01 - ANSI C / C99 / C11
        CPP   = 1 << 1,  // 0x02 - C++ (any standard)
        CPP2  = 1 << 2,  // 0x04 - CPP2 pure mode
    };
    uint8_t grammar_modes = 0x07;  // Default: all modes (C | CPP | CPP2)

    // Lattice integration: byte-level class filter for pre-filtering
    // Uses LatticeClasses bitmask (16-bit from lattice_classes.h)
    // 0xFFFF = all classes (no filtering), 0x0000 = never matches
    uint16_t lattice_filter = 0xFFFF;

    // Context windows for disambiguation
    ::std::vector<::std::string> prev_tokens;      // Lookbehind (token types expected before)
    ::std::vector<::std::string> next_tokens;      // Lookahead (token types expected after)
    ::std::string scope_requirement;               // Required scope: "function_body", "struct_body", "class_body", "global", "any"

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

} // namespace ir
} // namespace cppfort