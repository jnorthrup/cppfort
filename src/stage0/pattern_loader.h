#pragma once

#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "orbit_mask.h"

namespace cppfort::stage0 {

struct PatternData {
    std::string name;
    std::string regex;
    std::string category;
    std::vector<std::string> unified_signatures;
    std::map<::cppfort::ir::GrammarType, std::string> grammar_variants;
};

class PatternLoader {
public:
    PatternLoader() = default;

    bool load_yaml(const std::string& path);

    const std::vector<PatternData>& patterns() const { return patterns_; }
    std::vector<PatternData>& patterns() { return patterns_; }

    size_t pattern_count() const { return patterns_.size(); }

private:
    std::vector<PatternData> patterns_;
};

} // namespace cppfort::stage0

