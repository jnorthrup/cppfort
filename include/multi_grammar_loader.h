#pragma once

#include "orbit_scanner.h"

namespace cppfort {
namespace ir {

// Forward declarations
struct OrbitPattern;
class PatternDatabase;

// Multi-grammar loader for orbit scanner
class MultiGrammarLoader {
public:
  // Constructor
  MultiGrammarLoader();

  // Destructor
  ~MultiGrammarLoader();

  // Load all grammar patterns from the patterns directory
  bool loadAllGrammars(const ::std::filesystem::path& patternsDir);

  // Load specific grammar patterns
  bool loadGrammar(GrammarType type, const ::std::filesystem::path& patternFile);

  // Get patterns for a specific grammar
  ::std::vector<OrbitPattern> getPatterns(GrammarType type) const;

  // Get all loaded patterns across all grammars
  ::std::vector<OrbitPattern> getAllPatterns() const;

  // Check if a grammar is loaded
  bool isGrammarLoaded(GrammarType type) const;

  // Get supported grammar types
  ::std::vector<GrammarType> getLoadedGrammars() const;

  // Clear all loaded patterns
  void clear();

  // Get loading statistics
  struct LoadStats {
    size_t totalPatterns = 0;
    ::std::unordered_map<GrammarType, size_t> patternsByGrammar;
    ::std::vector<::std::string> errors;
  };

  LoadStats getLoadStats() const;

private:
  // Pattern databases for each grammar type
  ::std::unordered_map<GrammarType, ::std::unique_ptr<PatternDatabase>> m_databases;

  // Loading statistics
  LoadStats m_stats;

  // Helper methods
  GrammarType detectGrammarType(const ::std::filesystem::path& filename) const;
  ::std::filesystem::path getDefaultPatternPath(GrammarType type) const;
  bool validatePatternFile(const ::std::filesystem::path& file) const;
};

// Utility functions
::std::string grammarTypeToString(GrammarType type);
GrammarType stringToGrammarType(const ::std::string& str);

} // namespace ir
} // namespace cppfort