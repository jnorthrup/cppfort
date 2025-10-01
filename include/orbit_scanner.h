#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <filesystem>

#include "orbit_mask.h"
#include "tblgen_patterns.h"

namespace cppfort {
namespace ir {

// Forward declarations
class RabinKarp;
class OrbitContext;
class MultiGrammarLoader;
struct OrbitPattern;
struct OrbitMatch;

// Orbit scanner configuration
struct OrbitScannerConfig {
  // Rabin-Karp settings
  uint32_t prime = 31;
  ::std::vector<size_t> windowSizes = {1, 2, 4, 8, 16, 32, 64};

  // Context tracking settings
  size_t maxDepth = 100;
  double minConfidence = 0.5;

  // Pattern matching settings
  size_t maxMatches = 1000;
  double patternThreshold = 0.7;

  // Multi-grammar settings
  ::std::filesystem::path patternsDir = "patterns";
  bool enableAllGrammars = true;
};

// Detection result for a code segment
struct DetectionResult {
  GrammarType detectedGrammar = GrammarType::UNKNOWN;
  double confidence = 0.0;
  ::std::vector<OrbitMatch> matches;
  ::std::unordered_map<GrammarType, double> grammarScores;
  ::std::string reasoning;
};

// Main orbit scanner class
class OrbitScanner {
public:
  // Constructor
  OrbitScanner(const OrbitScannerConfig& config = OrbitScannerConfig{});

  // Destructor
  ~OrbitScanner();

  // Initialize the scanner with patterns
  bool initialize();

  // Scan a code segment and detect grammar
  DetectionResult scan(const ::std::string& code) const;

  // Scan with custom patterns
  DetectionResult scan(const ::std::string& code, const ::std::vector<OrbitPattern>& patterns) const;

  // Get scanner configuration
  const OrbitScannerConfig& getConfig() const;

  // Update configuration
  void updateConfig(const OrbitScannerConfig& config);

  // Get loaded patterns count
  size_t getPatternCount() const;

  // Get loaded grammars
  ::std::vector<GrammarType> getLoadedGrammars() const;

private:
  // Configuration
  OrbitScannerConfig m_config;

  // Core components
  ::std::unique_ptr<RabinKarp> m_rabinKarp;
  ::std::unique_ptr<OrbitContext> m_context;
  ::std::unique_ptr<MultiGrammarLoader> m_loader;

  // Helper methods
  ::std::vector<OrbitMatch> findMatches(const ::std::string& code,
                                     const ::std::vector<OrbitPattern>& patterns) const;

  DetectionResult analyzeMatches(const ::std::vector<OrbitMatch>& matches) const;

  double calculateGrammarConfidence(GrammarType grammar,
                                   const ::std::vector<OrbitMatch>& matches) const;

  GrammarType determineBestGrammar(const ::std::unordered_map<GrammarType, double>& scores) const;

  ::std::string generateReasoning(const DetectionResult& result) const;

  double detectOrbitPattern(GrammarType grammar, const ::std::array<size_t, 6>& orbitCounts,
                           size_t pos, const ::std::string& code) const;

  // Validation
  bool validateConfig() const;
  bool validateInitialization() const;
};

// Utility functions
::std::string detectionResultToString(const DetectionResult& result);
void printDetectionResult(const DetectionResult& result);

} // namespace ir
} // namespace cppfort