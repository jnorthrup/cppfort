#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "orbit_mask.h"
#include "tblgen_patterns.h"

namespace cppfort {
namespace ir {

class PatternDatabase;

// Interface for supplying grammar patterns to the orbit scanner.
class IMultiGrammarLoader {
public:
  struct LoadStats {
    size_t totalPatterns = 0;
    ::std::unordered_map<GrammarType, size_t> patternsByGrammar;
    ::std::vector<::std::string> errors;
  };

  virtual ~IMultiGrammarLoader() = default;

  virtual bool loadAllGrammars(const ::std::filesystem::path& patternsDir) = 0;
  virtual bool loadGrammar(GrammarType type, const ::std::filesystem::path& patternFile) = 0;
  virtual ::std::vector<OrbitPattern> getPatterns(GrammarType type) const = 0;
  virtual ::std::vector<OrbitPattern> getAllPatterns() const = 0;
  virtual bool isGrammarLoaded(GrammarType type) const = 0;
  virtual ::std::vector<GrammarType> getLoadedGrammars() const = 0;
  virtual void clear() = 0;
  virtual LoadStats getLoadStats() const = 0;
};

// Multi-grammar loader for orbit scanner
class MultiGrammarLoader : public IMultiGrammarLoader {
public:
  // Constructor
  MultiGrammarLoader();

  // Destructor
  ~MultiGrammarLoader();

  // Load all grammar patterns from the patterns directory
  bool loadAllGrammars(const ::std::filesystem::path& patternsDir) override;

  // Load specific grammar patterns
  bool loadGrammar(GrammarType type, const ::std::filesystem::path& patternFile) override;

  // Get patterns for a specific grammar
  ::std::vector<OrbitPattern> getPatterns(GrammarType type) const override;

  // Get all loaded patterns across all grammars
  ::std::vector<OrbitPattern> getAllPatterns() const override;

  // Check if a grammar is loaded
  bool isGrammarLoaded(GrammarType type) const override;

  // Get supported grammar types
  ::std::vector<GrammarType> getLoadedGrammars() const override;

  // Clear all loaded patterns
  void clear() override;

  // Get loading statistics
  using LoadStats = IMultiGrammarLoader::LoadStats;

  LoadStats getLoadStats() const override;

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
