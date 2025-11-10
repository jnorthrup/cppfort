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
namespace stage0 {

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
    void speculate_across_fragments(const std::vector<OrbitFragment>& fragments, std::string_view source);

    // Speculative matching using alternating anchor/evidence pattern
    void speculate_alternating(const PatternData& pattern, std::string_view text);

    // Experimental: Anchor tuple + terminal-span backchain thinning using TypeEvidence
    void speculate_backchain(std::string_view text);

    // Get the best speculative match (longest match, then highest confidence)
    const SpeculativeMatch* get_best_match() const;

    Capabilities patternCapabilities() const { return {}; }

    const SimdScanner& scalarScanner() const { return m_scalarScanner; }

    // Set patterns for speculation
    void set_patterns(const std::vector<PatternData>& patterns) { patterns_ = &patterns; }

    // Set packrat cache for memoization
    void set_packrat_cache(PackratCache* cache) { packrat_cache_ = cache; }

    // Enable semantic trace capture mode
    void enable_trace_capture(bool enable) { capture_traces_ = enable; }
    
    // Get collected semantic traces (for regression testing)
    const std::vector<SemanticTrace>& get_semantic_traces() const { return semantic_traces_; }
    
    // Clear collected traces
    void clear_traces() { semantic_traces_.clear(); }

private:
    ScalarScanner m_scalarScanner;
    std::vector<SpeculativeMatch> matches_;
    const std::vector<PatternData>* patterns_ = nullptr;
    PackratCache* packrat_cache_ = nullptr;
    bool capture_traces_ = false;  // NEW: enable trace capture mode
    std::vector<SemanticTrace> semantic_traces_;  // NEW: collected traces
    
    // Validate evidence type for alternating patterns
    bool validate_evidence_type(const std::string& type, std::string_view evidence) const;
    // Simple glob matcher supporting '*', '?'
    static bool globMatch(std::string_view text, std::string_view pattern);
};

// Configuration for combinator pool growth and limits
struct PoolConfig {
    size_t initial_size = 16;
    size_t max_size = 128;  // Conservative limit for RBCursiveScanner objects
    float growth_factor = 1.5f;  // Exponential growth factor
    bool allow_growth = true;
    bool log_growth = false;  // For debugging growth events
};

// Metrics for monitoring pool behavior
struct PoolMetrics {
    size_t current_size = 0;
    size_t peak_size = 0;
    size_t total_allocations = 0;
    size_t allocation_failures = 0;
    size_t growth_events = 0;
    
    void record_allocation(bool success) {
        total_allocations++;
        if (!success) {
            allocation_failures++;
        }
    }
    
    void record_growth(size_t new_size) {
        growth_events++;
        current_size = new_size;
        if (new_size > peak_size) {
            peak_size = new_size;
        }
    }
};

class CombinatorPool {
public:
    explicit CombinatorPool(const PoolConfig& config = {});
    explicit CombinatorPool(std::size_t initial_size);  // Legacy constructor

    RBCursiveScanner* allocate();
    void release(RBCursiveScanner* scanner);
    std::size_t available() const;
    std::size_t capacity() const { return pool_.size(); }
    
    // Metrics access
    const PoolMetrics& metrics() const { return metrics_; }
    PoolMetrics& metrics() { return metrics_; }
    
    // Manual resizing
    bool resize(std::size_t new_size);

private:
    void grow_pool();
    
    PoolConfig config_;
    PoolMetrics metrics_;
    std::vector<RBCursiveScanner> pool_;
    std::vector<bool> used_;
};

} // namespace cppfort
} // namespace stage0
