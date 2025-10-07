#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <filesystem>

#include "orbit_mask.h"
#include "cpp2_key_resolver.h"
#include "unified_orbit_patterns.h"
#include "multi_grammar_loader.h"

namespace cppfort {
namespace ir {

// Forward declarations
// (none needed - all included)
class RabinKarp;
class OrbitContext;
class RBCursiveScanner;

/**
 * Configuration for the orbit scanner
 */
struct OrbitScannerConfig {
    std::filesystem::path patternsDir;  // Directory containing pattern files
    double patternThreshold = 0.5;      // Minimum confidence for pattern matches
    size_t maxMatches = 1000;           // Maximum number of matches to return
    size_t maxDepth = 100;              // Maximum orbit depth to track

    OrbitScannerConfig() = default;
    OrbitScannerConfig(const std::filesystem::path& dir)
        : patternsDir(dir) {}
};

/**
 * Type alias for vector of orbit matches
 */
using MatchResults = std::vector<OrbitMatch>;

/**
 * Result of grammar detection analysis
 */
struct DetectionResult {
    GrammarType detectedGrammar = GrammarType::UNKNOWN;
    double confidence = 0.0;
    std::string reasoning;
    std::unordered_map<GrammarType, double> grammarScores;
    MatchResults matches;

    DetectionResult() = default;
};

/**
 * Orbit Scanner: Multi-grammar pattern detection with orbit-based analysis
 *
 * This scanner detects language patterns using hierarchical orbit structures
 * and supports n-way grammar disambiguation (C, C++, CPP2).
 */
class OrbitScanner {
private:
    OrbitScannerConfig m_config;
    std::unique_ptr<RabinKarp> m_rabinKarp;
    std::unique_ptr<OrbitContext> m_context;
    std::unique_ptr<MultiGrammarLoader> m_loader;
    std::unique_ptr<cppfort::stage0::CPP2KeyResolver> m_cpp2Resolver;
    std::unique_ptr<UnifiedOrbitDatabase> m_unifiedDatabase;
    // Centralized scanner/lexer utility (private asset, replacing ad-hoc scanners)
    std::unique_ptr<RBCursiveScanner> m_rbcursive;

    // Validation methods
    bool validateConfig() const;
    bool validateInitialization() const;

    // Core scanning methods
    MatchResults findMatches(const std::string& code,
                           const std::vector<OrbitPattern>& patterns) const;
    MatchResults findUnifiedMatches(const std::string& code,
                                   const std::vector<UnifiedOrbitPattern>& patterns) const;
    MatchResults applyCPP2KeyResolution(const std::string& code,
                                       const MatchResults& candidates) const;
    double detectOrbitPattern(GrammarType grammar,
                            const std::array<size_t, 6>& orbitCounts,
                            size_t pos, const std::string& code) const;

    // Analysis methods
    DetectionResult analyzeMatches(const MatchResults& matches) const;
    double calculateGrammarConfidence(GrammarType grammar,
                                    const MatchResults& matches) const;
    GrammarType determineBestGrammar(const std::unordered_map<GrammarType, double>& scores) const;
    // Helper methods for CPP2 key resolution
    std::string determine_scope_type(const std::string& context) const;
    uint16_t determine_lattice_mask(const std::string& context) const;
    std::string generateReasoning(const DetectionResult& result) const;

public:
    /**
     * Constructor
     * @param config Scanner configuration
     * @param loader Grammar loader (optional, will create default if null)
     */
    OrbitScanner(const OrbitScannerConfig& config,
                 std::unique_ptr<MultiGrammarLoader> loader = nullptr);

    /**
     * Destructor
     */
    ~OrbitScanner();

    /**
     * Initialize the scanner with patterns
     * @return true if initialization successful
     */
    bool initialize();

    /**
     * Scan code and detect grammar type
     * @param code Source code to scan
     * @return Detection result with grammar type and confidence
     */
    DetectionResult scan(const std::string& code) const;

    /**
     * Scan code with specific patterns
     * @param code Source code to scan
     * @param patterns Patterns to use for detection
     * @return Detection result
     */
    DetectionResult scan(const std::string& code,
                        const std::vector<OrbitPattern>& patterns) const;

    /**
     * Scan code with both legacy and unified patterns
     * @param code Source code to scan
     * @param legacyPatterns Legacy orbit patterns
     * @param unifiedPatterns Unified orbit patterns
     * @return Detection result
     */
    DetectionResult scan(const std::string& code,
                        const std::vector<OrbitPattern>& legacyPatterns,
                        const std::vector<UnifiedOrbitPattern>& unifiedPatterns) const;

    /**
     * Get current configuration
     */
    const OrbitScannerConfig& getConfig() const;

    /**
     * Update configuration
     * @param config New configuration
     */
    void updateConfig(const OrbitScannerConfig& config);

    /**
     * Get number of loaded patterns
     */
    size_t getPatternCount() const;

    /**
     * Get list of loaded grammars
     */
    std::vector<GrammarType> getLoadedGrammars() const;
};

} // namespace ir
} // namespace cppfort
