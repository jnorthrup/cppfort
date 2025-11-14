#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "speculation.h"
#include "orbit_ring.h"
#include "packrat_cache.h"
#include "pattern_loader.h"

namespace cppfort {
namespace ir {

class SimdScanner {
public:
    virtual ~SimdScanner() = default;
    virtual std::vector<std::size_t> scanBytes(std::span<const std::uint8_t> data,
                                               std::span<const std::uint8_t> targets) const = 0;
};

class ScalarScanner final : public SimdScanner {
public:
    std::vector<std::size_t> scanBytes(std::span<const std::uint8_t> data,
                                       std::span<const std::uint8_t> targets) const override;
};

// Minimal, internal scanning facade inspired by Litebike rbcursive.
// This is a private asset of the orbit scanner. It intentionally starts
// simple and can be extended with SIMD and protocol combinators later.
class RBCursiveScanner {
public:
    enum class PatternType { Glob, Regex };

    struct Match {
        std::size_t start = 0;
        std::size_t end = 0; // one-past-end
    };

    struct Capabilities {
        bool glob = true;
        bool regex = true;
        bool simd = false; // placeholder until SIMD paths are ported
    };

    RBCursiveScanner() = default;

    // Returns true if the whole text matches the glob pattern
    bool matchGlob(std::string_view text, std::string_view pattern) const;

    // Returns the first regex full-match if any
    bool matchRegex(std::string_view text, std::string_view pattern) const;

    // Scans and returns non-overlapping matches by pattern type
    std::vector<Match> scanWithPattern(std::string_view data,
                                       std::string_view pattern,
                                       PatternType type) const;

    // Speculative matching against all known patterns
    void speculate(std::string_view text);

    // Speculative matching against all known patterns across fragment boundaries
    void speculate_across_fragments(const std::vector< ::cppfort::stage0::OrbitFragment>& fragments, std::string_view source);

    // Speculative matching using alternating anchor/evidence pattern
    void speculate_alternating(const ::cppfort::stage0::PatternData& pattern, std::string_view text);

    // Experimental: Anchor tuple + terminal-span backchain thinning using TypeEvidence
    void speculate_backchain(std::string_view text);

    // Orbit-based anchor position detection (NO string.find())
    std::vector<std::size_t> find_anchor_positions_orbit(std::string_view text, 
                                                         std::string_view anchor,
                                                         const std::vector<int>* depth_map = nullptr) const;

    // Get the best speculative match (longest match, then highest confidence)
    const ::cppfort::stage0::SpeculativeMatch* get_best_match() const;

    Capabilities patternCapabilities() const { return {}; }

    const SimdScanner& scalarScanner() const { return m_scalarScanner; }

    // Set patterns for speculation
    void set_patterns(const std::vector< ::cppfort::stage0::PatternData>& patterns) { patterns_ = &patterns; }

    // Set packrat cache for memoization
    void set_packrat_cache( ::cppfort::stage0::PackratCache* cache) { packrat_cache_ = cache; }

private:
    ScalarScanner m_scalarScanner;
    std::vector< ::cppfort::stage0::SpeculativeMatch> matches_;
    const std::vector< ::cppfort::stage0::PatternData>* patterns_ = nullptr;
    ::cppfort::stage0::PackratCache* packrat_cache_ = nullptr;
    
    // Validate evidence type for alternating patterns
    bool validate_evidence_type(const std::string& type, std::string_view evidence) const;
    // Simple glob matcher supporting '*', '?'
    static bool globMatch(std::string_view text, std::string_view pattern);
};

class CombinatorPool {
public:
    explicit CombinatorPool(std::size_t initial_size = 0);

    RBCursiveScanner* allocate();
    void release(RBCursiveScanner* scanner);
    std::size_t available() const;

private:
    std::vector<RBCursiveScanner> pool_;
    std::vector<bool> used_;
};

} // namespace ir
} // namespace cppfort
