#pragma once

#include <array>
#include <cstdint>
#include <span>
#include "lattice_classes.h"

namespace cppfort::stage0 {

/**
 * Heuristic Tile for 256-byte analysis
 * 
 * Provides byte-level lattice classification for orbit pre-filtering
 * and feeds TypeEvidence counters for pattern matching.
 */
class HeuristicTile {
public:
    size_t tile_id;
    std::span<const char> tile_span;
    uint16_t lattice_mask;                    // Bitmask of detected classes in this tile
    std::array<uint16_t, 16> class_counters;  // 16-bit counters per class (fits 256 bytes)
    double orbit_confidence;                   // Prediction confidence for token orbits
    std::array<int, 3> lookahead_tiles;       // Predicted patterns for next 3 tiles

    HeuristicTile()
        : tile_id(0)
        , tile_span()
        , lattice_mask(0)
        , class_counters{}
        , orbit_confidence(0.0)
        , lookahead_tiles{0, 0, 0}
    {}

    HeuristicTile(size_t id, std::span<const char> span)
        : tile_id(id)
        , tile_span(span)
        , lattice_mask(0)
        , class_counters{}
        , orbit_confidence(0.0)
        , lookahead_tiles{0, 0, 0}
    {}

    /**
     * Analyze tile: accumulate lattice classes with SIMD-friendly loop
     * Updates class counters that feed into TypeEvidence for orbit carving
     */
    void analyze_tile() {
        lattice_mask = 0;
        std::fill(class_counters.begin(), class_counters.end(), 0);

        for (size_t i = 0; i < tile_span.size(); ++i) {
            char byte = tile_span[i];
            uint16_t byte_mask = classify_byte(byte);

            // Accumulate bitmask (OR all unique classes)
            lattice_mask |= byte_mask;

            // Count occurrences per class (16-bit counters)
            // Maps: bit 0-15 to class counters (aligned with CharClass enum)
            for (uint16_t bit = 0; bit < 16; ++bit) {
                if (byte_mask & (1 << bit)) {
                    if (class_counters[bit] < 65535) {
                        ++class_counters[bit];
                    }
                }
            }
        }

        // Normalize counters to confidence (max 256 bytes -> 1.0)
        for (size_t i = 0; i < 16; ++i) {
            class_counters[i] = static_cast<uint16_t>(
                std::min(65535u, static_cast<uint32_t>(class_counters[i]) * 256u)
            );
        }

        // Initial orbit confidence based on diversity
        int unique_classes = 0;
        for (int bit = 0; bit < 16; ++bit) {
            if (lattice_mask & (1 << bit)) {
                ++unique_classes;
            }
        }
        orbit_confidence = std::min(1.0, static_cast<double>(unique_classes) / 8.0);
    }

    /**
     * Apply AND mask to filter specific patterns for TypeEvidence
     * Example: filter to just digits + operators for numeric literal detection
     */
    uint16_t apply_and_mask(uint16_t mask) const {
        return lattice_mask & mask;
    }

    /**
     * Apply OR mask to combine signals (e.g., structural boundaries for orbit tracking)
     */
    uint16_t apply_or_mask(uint16_t mask) const {
        return lattice_mask | mask;
    }

    /**
     * Predict lookahead orbits based on current tile patterns
     * Generates predictions that guide confix depth tracking and region carving
     */
    void predict_lookahead_simple() {
        // Count structural openings vs closures (bit 5 = Structural)
        int open_structures = class_counters[5] - (class_counters[5] / 3); // Simplified

        if (open_structures > 0) {
            // Predict closure in next tiles (for wobbling window algorithm)
            lookahead_tiles[0] = 5;  // Next tile likely structural (CharClass::Structural)
            lookahead_tiles[1] = 3;  // Followed by punctuation
            lookahead_tiles[2] = 0;  // Then whitespace
            orbit_confidence *= 1.2; // Boost for predictable orbits
        } 
        else if ((lattice_mask & static_cast<uint16_t>(CharClass::Digit)) &&
                 (lattice_mask & static_cast<uint16_t>(CharClass::Operator))) {
            // Numeric orbit: predict continuation or decimal patterns
            // Guides TypeEvidence.numeric_literal() detection
            lookahead_tiles[0] = 1;  // Digit
            lookahead_tiles[1] = 4;  // Operator
            lookahead_tiles[2] = 3;  // Punctuation (decimal point or comma)
        }
        else if (lattice_mask & static_cast<uint16_t>(CharClass::Alpha)) {
            // Identifier orbit - guides identifier_start detection
            lookahead_tiles[0] = 2;  // Alpha (identifier continuation)
            lookahead_tiles[1] = 4;  // Operator (assignment, comparison)
            lookahead_tiles[2] = 3;  // Punctuation (end of statement)
        }
    }
};

} // namespace cppfort::stage0
