#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <span>
#include <optional>
#include <map>
#include <any>
#include <chrono>

#include "orbit_mask.h"
#include "pattern_matcher.h"

namespace cppfort::stage0 {

// ===== PATTERN DETECTION RESULT TYPES =====

/**
 * Pattern detection result with confidence scoring
 * Inspired by fizzbuzz_wide_scanner_ccek_final.cpp2
 */
struct PatternResult {
    uint32_t orbit_id {0};           // Protocol orbit identifier
    uint64_t pattern_mask {0};       // Bitmask of detected patterns
    bool is_active {false};          // Whether pattern is currently active
    double confidence {0.0};         // Detection confidence score (0.0-1.0)
    uint64_t timestamp {0};          // Detection timestamp
    size_t position {0};             // Position in source where pattern was detected
    std::string pattern_name;        // Human-readable pattern name
    std::map<std::string, std::any> metadata; // Extended metadata
};

/**
 * CCEK (Coroutine Context Element Key) for pattern injection
 * Allows patterns to be stored and retrieved in coroutine context
 */
struct PatternCCEK {
    std::string pattern_key {"pattern_context"};
    std::vector<PatternResult> orbit_context;

    // Inject pattern context
    void inject_pattern(const PatternResult& result) {
        orbit_context.push_back(result);
    }

    // Extract pattern by orbit ID
    std::optional<PatternResult> extract_orbit(uint32_t orbit_id) const {
        for (const auto& result : orbit_context) {
            if (result.orbit_id == orbit_id) {
                return result;
            }
        }
        return std::nullopt;
    }

    // Clear context
    void clear() {
        orbit_context.clear();
    }
};

// ===== PATTERN SCANNER =====

/**
 * FizzBuzz-inspired pattern scanner for multi-grammar detection
 * 
 * Uses modulo-based detection strategy where different languages/protocols
 * are detected at different "frequencies" (like FizzBuzz divisibility):
 * - C patterns (divisible by 3): sizeof, typedef, struct
 * - C++ patterns (divisible by 5): template, namespace, class
 * - CPP2 patterns (divisible by 7): :=, ->, inspect
 * - Combined patterns (divisible by combinations): mixed syntax
 * 
 * This approach allows efficient, incremental pattern detection with
 * confidence scoring and orbit-based classification.
 */
class PatternScanner {
public:
    PatternScanner();
    explicit PatternScanner(const cppfort::ir::PatternDatabase& db);

    // Main scanning interface
    std::vector<PatternResult> scan(std::string_view source);
    std::vector<PatternResult> scanWithCCEK(std::string_view source);

    // Sliding window scanning for large sources
    std::vector<PatternResult> scanWindowed(std::string_view source, size_t window_size = 1024);

    // Get current CCEK context
    const PatternCCEK& getCCEK() const { return m_ccek; }
    PatternCCEK& getCCEK() { return m_ccek; }

    // Generate orbit mask from results
    std::string generateOrbitMask(const std::vector<PatternResult>& results) const;

    // Calculate pattern confidence for a position
    double calculateConfidence(std::string_view source, size_t position, 
                              const cppfort::ir::OrbitPattern& pattern) const;

private:
    cppfort::ir::PatternDatabase m_pattern_db;
    PatternCCEK m_ccek;

    // FizzBuzz-style detection helpers
    bool detectCPattern(std::string_view source, size_t position) const;
    bool detectCppPattern(std::string_view source, size_t position) const;
    bool detectCpp2Pattern(std::string_view source, size_t position) const;

    // Pattern matching utilities
    bool matchesSignature(std::string_view source, size_t position,
                         const std::vector<std::string>& patterns) const;
    
    double calculateDensity(std::string_view source, size_t position,
                           const std::vector<std::string>& patterns, 
                           size_t window = 20) const;

    // Timestamp utility
    static uint64_t getCurrentTimestamp() {
        return std::chrono::high_resolution_clock::now()
            .time_since_epoch()
            .count();
    }
};

// ===== ORBIT MASK UTILITIES =====

/**
 * Orbit mask bits for different language/protocol patterns
 * Allows compact representation of detected patterns
 */
namespace OrbitMasks {
    constexpr uint64_t C_PATTERNS       = 0x0000'0000'0000'0001ULL;
    constexpr uint64_t CPP_PATTERNS     = 0x0000'0000'0000'0002ULL;
    constexpr uint64_t CPP2_PATTERNS    = 0x0000'0000'0000'0004ULL;
    constexpr uint64_t TEMPLATE_ORBIT   = 0x1000'0000'0000'0000ULL;
    constexpr uint64_t FUNCTION_ORBIT   = 0x2000'0000'0000'0000ULL;
    constexpr uint64_t CLASS_ORBIT      = 0x4000'0000'0000'0000ULL;
    constexpr uint64_t DEFAULT_ORBIT    = 0x8000'0000'0000'0000ULL;
}

} // namespace cppfort::stage0
