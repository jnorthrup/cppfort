#pragma once

#include <string>
#include <vector>
#include <optional>

namespace cppfort::stage0 {

// Match CPP2 input against tblgen pattern and extract segments
class TblgenPatternMatcher {
public:
    // Match pattern like "$0: ($1) -> $2 = $3" against input
    // Returns segments [$0, $1, $2, $3] if match successful
    static std::optional<std::vector<std::string>> match(
        const std::string& pattern,
        const std::string& input
    );

private:
    // Convert pattern to regex, replacing $N with capture groups
    static std::string pattern_to_regex(const std::string& pattern);
};

} // namespace cppfort::stage0
