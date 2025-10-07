#include "multi_grammar_loader.h"
#include <iostream>
#include <algorithm>
#include <fstream>
#include <filesystem>

namespace cppfort {
namespace ir {

MultiGrammarLoader::MultiGrammarLoader() = default;

MultiGrammarLoader::~MultiGrammarLoader() = default;

bool MultiGrammarLoader::loadAllGrammars(const ::std::filesystem::path& patternsDir) {
  if (!::std::filesystem::exists(patternsDir) || !::std::filesystem::is_directory(patternsDir)) {
    m_stats.errors.push_back("Patterns directory does not exist: " + patternsDir.string());
    return false;
  }

  bool success = true;
  size_t loadedCount = 0;

  // Load each grammar type
  for (const auto& entry : ::std::filesystem::directory_iterator(patternsDir)) {
    if (!entry.is_regular_file()) continue;

    const auto& path = entry.path();
    if (path.extension() != ".yaml") continue;

    GrammarType type = detectGrammarType(path.filename().string());
    if (type != GrammarType::UNKNOWN) {
      if (loadGrammar(type, path)) {
        loadedCount++;
      } else {
        success = false;
      }
    }
  }

  if (loadedCount == 0) {
    m_stats.errors.push_back("No valid grammar pattern files found in: " + patternsDir.string());
    return false;
  }

  return success;
}

bool MultiGrammarLoader::loadGrammar(GrammarType type, const ::std::filesystem::path& patternFile) {
  if (!validatePatternFile(patternFile)) {
    m_stats.errors.push_back("Invalid pattern file: " + patternFile.string());
    return false;
  }

  // Create pattern database for this grammar
  auto database = ::std::make_unique<PatternDatabase>();

  if (!database->loadFromYaml(patternFile)) {
    m_stats.errors.push_back("Failed to load patterns from: " + patternFile.string());
    return false;
  }

  // Store the database
  m_databases[type] = ::std::move(database);

  // Update statistics
  const auto& patterns = m_databases[type]->getPatterns();
  m_stats.patternsByGrammar[type] = patterns.size();
  m_stats.totalPatterns += patterns.size();

  return true;
}

::std::vector<OrbitPattern> MultiGrammarLoader::getPatterns(GrammarType type) const {
  auto it = m_databases.find(type);
  if (it != m_databases.end()) {
    return it->second->getPatterns();
  }

  return {};
}

::std::vector<OrbitPattern> MultiGrammarLoader::getAllPatterns() const {
  ::std::vector<OrbitPattern> allPatterns;

  for (const auto& [type, database] : m_databases) {
    const auto& patterns = database->getPatterns();
    allPatterns.insert(allPatterns.end(), patterns.begin(), patterns.end());
  }

  return allPatterns;
}

bool MultiGrammarLoader::isGrammarLoaded(GrammarType type) const {
  return m_databases.find(type) != m_databases.end();
}

::std::vector<GrammarType> MultiGrammarLoader::getLoadedGrammars() const {
  ::std::vector<GrammarType> loaded;

  for (const auto& [type, database] : m_databases) {
    loaded.push_back(type);
  }

  return loaded;
}

void MultiGrammarLoader::clear() {
  m_databases.clear();
  m_stats = LoadStats{};
}

MultiGrammarLoader::LoadStats MultiGrammarLoader::getLoadStats() const {
  return m_stats;
}

GrammarType MultiGrammarLoader::detectGrammarType(const ::std::filesystem::path& filename) const {
  ::std::string name = filename.stem().string();

  if (name == "c_patterns") return GrammarType::C;
  if (name == "cpp_patterns") return GrammarType::CPP;
  if (name == "cpp2_patterns") return GrammarType::CPP2;

  return GrammarType::UNKNOWN;
}

::std::filesystem::path MultiGrammarLoader::getDefaultPatternPath(GrammarType type) const {
  ::std::string filename;

  switch (type) {
    case GrammarType::C: filename = "c_patterns.yaml"; break;
    case GrammarType::CPP: filename = "cpp_patterns.yaml"; break;
    case GrammarType::CPP2: filename = "cpp2_patterns.yaml"; break;
    default: return "";
  }

  return ::std::filesystem::path("patterns") / filename;
}

bool MultiGrammarLoader::validatePatternFile(const ::std::filesystem::path& file) const {
  if (!::std::filesystem::exists(file)) return false;
  if (!::std::filesystem::is_regular_file(file)) return false;

  // Basic validation - check if file is readable
  ::std::ifstream testFile(file);
  return testFile.good();
}

::std::string grammarTypeToString(GrammarType type) {
  switch (type) {
    case GrammarType::C: return "C";
    case GrammarType::CPP: return "C++";
    case GrammarType::CPP2: return "CPP2";
    default: return "UNKNOWN";
  }
}

GrammarType stringToGrammarType(const ::std::string& str) {
  if (str == "C") return GrammarType::C;
  if (str == "C++" || str == "CPP") return GrammarType::CPP;
  if (str == "CPP2") return GrammarType::CPP2;
  return GrammarType::UNKNOWN;
}

} // namespace ir
} // namespace cppfort