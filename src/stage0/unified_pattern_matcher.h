#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <optional>
#include <functional>
#include <unordered_map>
#include "pattern_loader.h"

namespace cppfort::stage0 {

// Unified pattern match result combining all matcher features
struct UnifiedPatternMatch {
    size_t start_pos;
    size_t end_pos;
    const PatternData* pattern;
    std::vector<std::string> segments;
    int depth = 0;  // Optional depth tracking

    // For IR lowering compatibility (simplified - no IR dependency)
    std::string emitted_code;
    int target_language = 0;  // 0=C, 1=CPP, 2=CPP2
};

/**
 * Unified pattern matcher consolidating:
 * - TblgenPatternMatcher (segment extraction)
 * - DepthPatternMatcher (depth-aware matching)
 * - PatternMatcher (IR lowering patterns)
 */
class UnifiedPatternMatcher {
public:
    // Core matching function - finds all matches in text
    static std::vector<UnifiedPatternMatch> find_matches(
        std::string_view text,
        const std::vector<PatternData>& patterns,
        bool track_depth = false,
        int max_depth = 5
    );

    // Segment extraction (from TblgenPatternMatcher)
    static std::optional<std::vector<std::string>> extract_segments(
        const std::string& pattern,
        const std::string& input
    );

    // IR lowering support (simplified)
    using RewriteFunc = std::function<std::string(const UnifiedPatternMatch&)>;

    void register_rewrite(
        const std::string& pattern_name,
        int target_language,  // 0=C, 1=CPP, 2=CPP2
        RewriteFunc rewrite,
        int priority = 0
    );

    std::optional<std::string> apply_rewrite(
        const UnifiedPatternMatch& match,
        int target_language
    ) const;

private:
    // Single internal implementation for all matching
    static std::optional<UnifiedPatternMatch> try_match_at(
        std::string_view text,
        size_t pos,
        const PatternData& pattern,
        const std::vector<int>* depth_map = nullptr
    );

    // Rewrite registry for IR lowering
    struct RewriteEntry {
        RewriteFunc func;
        int priority;
    };
    std::unordered_map<std::string, std::unordered_map<int, RewriteEntry>> rewrites_;
};

} // namespace cppfort::stage0