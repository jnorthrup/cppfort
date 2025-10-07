#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <array>

namespace cppfort {
namespace ir {

// Grammar types supported by the orbit scanner
enum class GrammarType {
  C,
  CPP,
  CPP2,
  UNKNOWN
};

/**
 * Chapter 19: Orbit Type Enumeration
 *
 * Defines the 8 basic orbit types for structural delimiters.
 * Based on ororoboros-couchduck orbit scanner patterns.
 */
enum class OrbitType {
    None = 0,
    OpenBrace, CloseBrace,        // { }
    OpenBracket, CloseBracket,    // [ ]
    OpenAngle, CloseAngle,        // < >
    OpenParen, CloseParen,        // ( )
    Quote,                        // "
    NumberStart, NumberEnd,       // Numeric literals
    Unknown
};

/**
 * Densified Orbit Context - SIMD-friendly packed counters
 * 32 bytes total, fits in two cache lines, enables vectorized operations
 */
struct DenseOrbitContext {
    // Packed depth counters (8 bytes) - enables SIMD comparison
    uint8_t brace_depth;      // { } balance (0-255)
    uint8_t bracket_depth;    // [ ] balance
    uint8_t angle_depth;      // < > balance
    uint8_t paren_depth;      // ( ) balance
    uint8_t quote_depth;      // " balance
    uint8_t number_depth;     // Numeric literal balance
    uint8_t reserved1;        // Alignment padding
    uint8_t reserved2;        // Alignment padding

    // Packed position tracking (8 bytes)
    uint32_t last_open_pos;   // Position of last opening delimiter
    uint16_t current_depth;   // Sum of all depths
    uint16_t max_depth;       // Maximum allowed depth

    // Packed state flags (8 bytes) - bitfield for fast masking
    uint32_t state_flags;     // Bit flags for various states
    uint32_t reserved_flags;  // Future expansion

    // Packed hash accumulators (8 bytes) - for Rabin-Karp orbit hashing
    uint32_t hash_brace;      // Rolling hash for brace sequences
    uint32_t hash_bracket;    // Rolling hash for bracket sequences

    // SIMD-friendly accessors
    uint8_t getDepth(uint8_t delim_type) const {
        switch (delim_type) {
            case 0: return brace_depth;
            case 1: return bracket_depth;
            case 2: return angle_depth;
            case 3: return paren_depth;
            case 4: return quote_depth;
            case 5: return number_depth;
            default: return 0;
        }
    }

    void setDepth(uint8_t delim_type, uint8_t depth) {
        switch (delim_type) {
            case 0: brace_depth = depth; break;
            case 1: bracket_depth = depth; break;
            case 2: angle_depth = depth; break;
            case 3: paren_depth = depth; break;
            case 4: quote_depth = depth; break;
            case 5: number_depth = depth; break;
        }
    }

    // Check if context is balanced (SIMD-friendly)
    bool isBalanced() const {
        return (brace_depth | bracket_depth | angle_depth | paren_depth | quote_depth | number_depth) == 0;
    }

    // Get confix mask for pattern matching
    uint8_t confixMask() const {
        uint8_t mask = 0;
        if (brace_depth > 0) mask |= (1 << 1);
        if (paren_depth > 0) mask |= (1 << 2);
        if (angle_depth > 0) mask |= (1 << 3);
        if (bracket_depth > 0) mask |= (1 << 4);
        if (quote_depth > 0) mask |= (1 << 5);
        return mask;
    }

    // Reset context
    void reset() {
        brace_depth = bracket_depth = angle_depth = paren_depth = quote_depth = number_depth = 0;
        last_open_pos = 0;
        current_depth = 0;
        state_flags = 0;
        hash_brace = hash_bracket = 0;
    }
};

/**
 * Chapter 19: Orbit Match Structure
 *
 * Represents a detected orbit delimiter with position and confidence.
 * Based on orbit_scanner_test.cpp reference implementation.
 */
struct OrbitMatch {
    ::std::string patternName;   // Name of matched pattern
    GrammarType grammarType;     // Grammar type (C, CPP, CPP2)
    size_t position;             // Start position in code (alias for startPos)
    size_t length;               // Length of match (computed from endPos - startPos)
    size_t startPos;             // Start position in code
    size_t endPos;               // End position in code
    double confidence;           // Match confidence (0.0-1.0)
    ::std::string signature;     // Matched signature pattern
    int orbit_type;              // Orbit type for emission (1=function, 2=declaration, etc.)

    // Orbit-based matching data
    ::std::array<uint64_t, 6> orbitHashes;  // Hierarchical hashes for each orbit type
    ::std::array<size_t, 6> orbitCounts;    // Current orbit element counts

    OrbitMatch() = default;
    OrbitMatch(const ::std::string& name, GrammarType type, size_t start, size_t end,
               double conf, const ::std::string& sig, int otype = 0)
        : patternName(name), grammarType(type), position(start), length(end - start),
          startPos(start), endPos(end), confidence(conf), signature(sig), orbit_type(otype),
          orbitHashes{0}, orbitCounts{0} {}

        // Legacy constructor for backward compatibility
    OrbitMatch(size_t pos, OrbitType t, double conf, const ::std::string& snip)
        : patternName("legacy"), grammarType(GrammarType::UNKNOWN), position(pos), length(snip.length()),
          startPos(pos), endPos(pos + snip.length()), confidence(conf), signature(snip),
          orbit_type(static_cast<int>(t)), orbitHashes{0}, orbitCounts{0} {}
};

/**
 * Chapter 19: Orbit Context System
 *
 * Tracks structural balance and context for orbit detection.
 * Maintains depth counters for different delimiter types.
 * Based on hierarchical cascading typevidence pattern.
 */
class OrbitContext {
private:
    DenseOrbitContext dense_ctx;  // Densified context for locality
    ::std::vector<OrbitMatch> _matches;

public:
    /**
     * Constructor with optional max depth limit.
     */
    OrbitContext(size_t maxDepth = 100) {
        dense_ctx.max_depth = static_cast<uint16_t>(maxDepth);
        dense_ctx.reset();
    }

    /**
     * Process an orbit match and update context depth.
     * Returns true if match is valid in current context.
     */
    bool processMatch(const OrbitMatch& match);

    /**
     * Check if current context is structurally balanced.
     */
    bool isBalanced() const { return dense_ctx.isBalanced(); }

    /**
     * Get current depth for specific orbit type.
     */
    int depth(OrbitType type) const;

    /**
     * Get current total depth (sum of all depths).
     */
    int getDepth() const { return dense_ctx.current_depth; }

    /**
     * Get maximum allowed depth.
     */
    size_t getMaxDepth() const { return dense_ctx.max_depth; }

    /**
     * Calculate confidence score based on structural balance.
     * Returns value between 0.0 (unbalanced) and 1.0 (perfectly balanced).
     */
    double calculateConfidence() const;

    /**
     * Update context with a single character.
     * Tracks orbit element depths for structural analysis.
     */
    void update(char ch);

    /**
     * Get current orbit element counts as array.
     * Returns [brace_count, bracket_count, angle_count, paren_count, quote_count, number_count]
     */
    ::std::array<size_t, 6> getCounts() const;

    /**
     * Get current confix context as a bitmask.
     */
    uint8_t confixMask() const { return dense_ctx.confixMask(); }

    /**
     * Get all processed matches.
     */
    const ::std::vector<OrbitMatch>& matches() const { return _matches; }

    /**
     * Reset context to initial state.
     */
    void reset() {
        dense_ctx.reset();
        _matches.clear();
    }

    /**
     * Check if a match would be valid in current context.
     * Used for speculative matching.
     */
    bool wouldBeValid(const OrbitMatch& match) const;

    /**
     * Get access to densified context for advanced operations
     */
    const DenseOrbitContext& getDenseContext() const { return dense_ctx; }
    DenseOrbitContext& getDenseContext() { return dense_ctx; }
};

/**
 * OrbitPattern - unified pattern representation for clean room compliance
 * Alias to UnifiedOrbitPattern for backward compatibility
 */
struct OrbitPattern {
    ::std::string name;
    GrammarType grammarType;
    ::std::vector<::std::string> signatures;
    double weight = 1.0;
    uint8_t grammar_modes = 0x07;  // Default: C | CPP | CPP2

    OrbitPattern() = default;
    OrbitPattern(const ::std::string& n, GrammarType gt, const ::std::vector<::std::string>& sigs, double w = 1.0)
        : name(n), grammarType(gt), signatures(sigs), weight(w) {}
};

} // namespace ir
} // namespace cppfort