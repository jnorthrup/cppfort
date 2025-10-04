#pragma once

#include <array>
#include <cstdint>
#include <span>
#include "lattice_classes.h"

namespace cppfort::stage0 {

/**
 * Heuristic Tile for 256-byte analysis
 *
 * Extracted from build/x.txt heuristic_search_tiles.cpp2
 * Provides byte-level lattice classification for orbit pre-filtering
 */
class HeuristicTile {
public:
    size_t tile_id;
    std::span<const char> tile_span;
    uint16_t lattice_mask;                    // Bitmask of detected classes in this tile
    std::array<uint16_t, 16> class_counters;  // 16-bit counters per class (fits 256 bytes)
    double orbit_confidence;                   // Prediction confidence for token orbits
    std::array<int, 3> lookahead_tiles;       // Predicted patterns for next 3 tiles (0-15 class index)

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
     * Apply AND mask to filter specific patterns (e.g., numeric vs alphanumeric)
     */
    uint16_t apply_and_mask(uint16_t mask) const {
        return lattice_mask & mask;
    }

    /**
     * Apply OR mask to combine signals (e.g., structural boundaries)
     */
    uint16_t apply_or_mask(uint16_t mask) const {
        return lattice_mask | mask;
    }

    /**
     * Predict lookahead orbits based on current tile patterns
     * Note: Requires HeuristicGrid context for validation (forward declaration issue)
     */
    void predict_lookahead_simple() {
        // Lookahead logic: predict next 2-3 tiles based on open structures
        int open_structures = class_counters[4] - class_counters[5]; // Simplified open/close diff (bit 4 = STRUCTURAL)

        if (open_structures > 0) {
            // Predict closure in next tiles
            lookahead_tiles[0] = 4;  // Next tile likely structural (LatticeClasses::STRUCTURAL)
            lookahead_tiles[1] = 2;  // Followed by punctuation
            lookahead_tiles[2] = 3;  // Then whitespace/indent
            orbit_confidence *= 1.2; // Boost for predictable orbits
        } else if ((lattice_mask & LatticeClasses::DIGIT) &&
                   (lattice_mask & LatticeClasses::NUMERIC_OP)) {
            // Numeric orbit: predict continuation or decimal
            lookahead_tiles[0] = 0;  // DIGIT
            lookahead_tiles[1] = 5;  // NUMERIC_OP
            lookahead_tiles[2] = 2;  // End with period/comma (PUNCTUATION)
        } else if (lattice_mask & LatticeClasses::ALPHA) {
            // Identifier/string orbit
            lookahead_tiles[0] = 9;  // IDENTIFIER
            lookahead_tiles[1] = 8;  // OPERATOR
            lookahead_tiles[2] = 15; // SEMICOLON
        }
    }
};

} // namespace cppfort::stage0
