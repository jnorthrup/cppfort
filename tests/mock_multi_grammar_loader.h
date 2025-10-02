#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "multi_grammar_loader.h"

namespace cppfort::ir {

// Lightweight in-memory loader used to mock grammar loading during tests.
class MockMultiGrammarLoader : public IMultiGrammarLoader {
public:
  using LoadStats = IMultiGrammarLoader::LoadStats;

  void setPatterns(GrammarType type, ::std::vector<OrbitPattern> patterns) {
    m_patternStore[type] = ::std::move(patterns);
  }

  void setLoadAllSuccess(bool allow) { m_allowLoadAllSuccess = allow; }
  void setLoadGrammarSuccess(bool allow) { m_allowLoadGrammarSuccess = allow; }

  size_t loadAllCallCount() const { return m_loadAllCalls; }
  size_t loadGrammarCallCount() const { return m_loadGrammarCalls; }
  const ::std::filesystem::path& lastPatternsDir() const { return m_lastPatternsDir; }

  bool loadAllGrammars(const ::std::filesystem::path& patternsDir) override {
    ++m_loadAllCalls;
    m_lastPatternsDir = patternsDir;
    resetStats();

    if (!m_allowLoadAllSuccess) {
      m_stats.errors.push_back("Mock loadAllGrammars forced failure");
      return false;
    }

    if (m_patternStore.empty()) {
      m_stats.errors.push_back("Mock loader has no configured patterns");
      return false;
    }

    m_loaded.clear();
    for (const auto& [grammar, patterns] : m_patternStore) {
      if (!patterns.empty()) {
        m_loaded.insert(grammar);
      }
    }

    if (m_loaded.empty()) {
      m_stats.errors.push_back("Mock loader patterns are empty");
      return false;
    }

    refreshStats();
    return true;
  }

  bool loadGrammar(GrammarType type, const ::std::filesystem::path& patternFile) override {
    ++m_loadGrammarCalls;
    m_lastPatternFiles[type] = patternFile;

    if (!m_allowLoadGrammarSuccess) {
      m_stats.errors.push_back("Mock loadGrammar forced failure");
      return false;
    }

    auto it = m_patternStore.find(type);
    if (it == m_patternStore.end() || it->second.empty()) {
      m_stats.errors.push_back("No mock patterns configured for requested grammar");
      return false;
    }

    m_loaded.insert(type);
    refreshStats();
    return true;
  }

  ::std::vector<OrbitPattern> getPatterns(GrammarType type) const override {
    if (!isGrammarLoaded(type)) {
      return {};
    }

    auto it = m_patternStore.find(type);
    if (it == m_patternStore.end()) {
      return {};
    }

    return it->second;
  }

  ::std::vector<OrbitPattern> getAllPatterns() const override {
    ::std::vector<OrbitPattern> all;
    for (auto grammar : m_loaded) {
      auto it = m_patternStore.find(grammar);
      if (it != m_patternStore.end()) {
        all.insert(all.end(), it->second.begin(), it->second.end());
      }
    }
    return all;
  }

  bool isGrammarLoaded(GrammarType type) const override {
    return m_loaded.find(type) != m_loaded.end();
  }

  ::std::vector<GrammarType> getLoadedGrammars() const override {
    return ::std::vector<GrammarType>(m_loaded.begin(), m_loaded.end());
  }

  void clear() override {
    m_loaded.clear();
    resetStats();
  }

  LoadStats getLoadStats() const override {
    return m_stats;
  }

private:
  void resetStats() {
    m_stats.totalPatterns = 0;
    m_stats.patternsByGrammar.clear();
    m_stats.errors.clear();
  }

  void refreshStats() {
    m_stats.totalPatterns = 0;
    m_stats.patternsByGrammar.clear();
    for (auto grammar : m_loaded) {
      auto it = m_patternStore.find(grammar);
      if (it != m_patternStore.end()) {
        m_stats.patternsByGrammar[grammar] = it->second.size();
        m_stats.totalPatterns += it->second.size();
      }
    }
    if (m_stats.totalPatterns == 0) {
      m_stats.errors.push_back("Mock loader loaded grammars without patterns");
    }
  }

  ::std::unordered_map<GrammarType, ::std::vector<OrbitPattern>> m_patternStore;
  ::std::unordered_set<GrammarType> m_loaded;
  LoadStats m_stats;

  bool m_allowLoadAllSuccess = true;
  bool m_allowLoadGrammarSuccess = true;

  size_t m_loadAllCalls = 0;
  size_t m_loadGrammarCalls = 0;
  ::std::filesystem::path m_lastPatternsDir;
  ::std::unordered_map<GrammarType, ::std::filesystem::path> m_lastPatternFiles;
};

} // namespace cppfort::ir

