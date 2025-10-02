#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <array>

namespace cppfort::ir {

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
 * Chapter 19: Orbit Match Structure
 *
 * Represents a detected orbit delimiter with position and confidence.
 * Based on orbit_scanner_test.cpp reference implementation.
 */
struct OrbitMatch {
    ::std::string patternName;   // Name of matched pattern
    GrammarType grammarType;     // Grammar type (C, CPP, CPP2)
    size_t startPos;             // Start position in code
    size_t endPos;               // End position in code
    double confidence;           // Match confidence (0.0-1.0)
    ::std::string signature;     // Matched signature pattern

    // Orbit-based matching data
    ::std::array<uint64_t, 6> orbitHashes;  // Hierarchical hashes for each orbit type
    ::std::array<size_t, 6> orbitCounts;    // Current orbit element counts

    OrbitMatch() = default;
    OrbitMatch(const ::std::string& name, GrammarType type, size_t start, size_t end,
               double conf, const ::std::string& sig)
        : patternName(name), grammarType(type), startPos(start), endPos(end),
          confidence(conf), signature(sig), orbitHashes{0}, orbitCounts{0} {}

    // Legacy constructor for backward compatibility
    OrbitMatch(size_t pos, OrbitType t, double conf, const ::std::string& snip)
        : patternName("legacy"), grammarType(GrammarType::UNKNOWN), startPos(pos), endPos(pos + snip.length()),
          confidence(conf), signature(snip), orbitHashes{0}, orbitCounts{0} {}
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
    int _braceDepth = 0;       // { } balance
    int _bracketDepth = 0;     // [ ] balance
    int _angleDepth = 0;       // < > balance
    int _parenDepth = 0;       // ( ) balance
    int _quoteDepth = 0;       // " balance
    int _numberDepth = 0;      // Numeric literal balance
    size_t _maxDepth = 100;    // Maximum allowed depth

  ::std::vector<OrbitMatch> _matches;

public:
    /**
     * Constructor with optional max depth limit.
     */
  OrbitContext(size_t maxDepth = 100) : _maxDepth(maxDepth) {}

    /**
     * Process an orbit match and update context depth.
     * Returns true if match is valid in current context.
     */
    bool processMatch(const OrbitMatch& match);

    /**
     * Check if current context is structurally balanced.
     */
    bool isBalanced() const;

    /**
     * Get current depth for specific orbit type.
     */
    int depth(OrbitType type) const;

    /**
     * Get current total depth (sum of all depths).
     */
    int getDepth() const;

    /**
     * Get maximum allowed depth.
     */
    size_t getMaxDepth() const;

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
     * Get all processed matches.
     */
  const ::std::vector<OrbitMatch>& matches() const;

    /**
     * Reset context to initial state.
     */
    void reset();

    /**
     * Check if a match would be valid in current context.
     * Used for speculative matching.
     */
    bool wouldBeValid(const OrbitMatch& match) const;
};

} // namespace cppfort::ir