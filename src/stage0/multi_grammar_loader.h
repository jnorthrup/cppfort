#pragma once

#include <filesystem>
#include <vector>
#include <string>
#include <unordered_map>

namespace cppfort {
namespace ir {

struct GrammarStats {
    size_t loaded = 0;
    std::vector<std::string> errors;
};

class IMultiGrammarLoader {
public:
    virtual ~IMultiGrammarLoader() = default;
    virtual bool loadAllGrammars(const std::filesystem::path& patternsDir) = 0;
    virtual bool loadGrammar(GrammarType type, const std::filesystem::path& patternFile) = 0;
    virtual const GrammarStats& getStats() const = 0;
};

class MultiGrammarLoader : public IMultiGrammarLoader {
public:
    MultiGrammarLoader();
    ~MultiGrammarLoader() override;

    bool loadAllGrammars(const std::filesystem::path& patternsDir) override;
    bool loadGrammar(GrammarType type, const std::filesystem::path& patternFile) override;
    const GrammarStats& getStats() const override { return m_stats; }

private:
    GrammarStats m_stats;
    std::unordered_map<GrammarType, std::vector<std::string>> m_patterns;

    GrammarType detectGrammarType(const std::string& filename) const;
    bool validatePatternFile(const std::filesystem::path& file) const;
};

} // namespace ir
} // namespace cppfort