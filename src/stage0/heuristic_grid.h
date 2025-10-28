#pragma once

#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "heuristic_tile.h"
#include "lattice_classes.h"

namespace cppfort::stage0 {

/**
 * Grid of Heuristic Tiles (16 tiles per 4KB chunk)
 *
 * Extracted from build/x.txt heuristic_search_tiles.cpp2
 * Provides chunk-level orbit prediction and evidence accumulation
 */
class HeuristicGrid {
public:
    size_t chunk_id;
    std::unordered_map<size_t, HeuristicTile> tiles;  // tile_id -> tile
    uint16_t grid_mask;                                // Aggregate lattice for entire grid
    double prediction_accuracy;                        // Historical lookahead accuracy

    explicit HeuristicGrid(size_t id)
        : chunk_id(id)
        , tiles{}
        , grid_mask(0)
        , prediction_accuracy(0.5)  // Default
    {}

    /**
     * Build grid from 4KB span (16 x 256-byte tiles)
     */
    void build_grid(std::span<const char> chunk_span) {
        constexpr size_t TILE_SIZE = 256;
        size_t num_tiles = chunk_span.size() / TILE_SIZE;

        for (size_t i = 0; i < num_tiles; ++i) {
            size_t tile_start = i * TILE_SIZE;
            size_t tile_end = std::min(tile_start + TILE_SIZE, chunk_span.size());
            std::span<const char> tile_span{chunk_span.data() + tile_start, tile_end - tile_start};

            size_t tile_id = chunk_id * num_tiles + i;
            HeuristicTile tile(tile_id, tile_span);
            tile.analyze_tile();
            tiles[tile_id] = tile;

            // Aggregate grid mask
            grid_mask |= tile.lattice_mask;
        }

        // Perform lookahead predictions across tiles
        for (auto& [id, tile] : tiles) {
            tile.predict_lookahead_simple();
        }

        // Compute overall prediction accuracy (simplified: average tile confidences)
        double total_conf = 0.0;
        for (const auto& [id, tile] : tiles) {
            total_conf += tile.orbit_confidence;
        }
        if (!tiles.empty()) {
            prediction_accuracy = total_conf / tiles.size();
        }
    }

    /**
     * AND/OR mask applications at grid level
     */
    uint16_t apply_grid_and_mask(uint16_t mask) const {
        return grid_mask & mask;
    }

    uint16_t apply_grid_or_mask(uint16_t mask) const {
        return grid_mask | mask;
    }

    /**
     * Estimate confix token orbits: predict boundaries and types
     * Returns list of (orbit_name, confidence) pairs
     */
    std::vector<std::pair<std::string, double>> estimate_confix_orbits() const {
        std::vector<std::pair<std::string, double>> orbits;

        // Numeric orbit: DIGIT & NUMERIC_OP & ~ALPHA
        uint16_t numeric_mask = LatticeClasses::DIGIT | LatticeClasses::NUMERIC_OP;
        uint16_t filtered = apply_grid_and_mask(numeric_mask);
        if (filtered & LatticeClasses::DIGIT) {
            orbits.emplace_back("numeric_orbit", 0.9 * prediction_accuracy);
        }

        // String orbit: QUOTE & (ALPHA | DIGIT)
        uint16_t string_mask = LatticeClasses::QUOTE | LatticeClasses::ALPHA | LatticeClasses::DIGIT;
        filtered = apply_grid_and_mask(string_mask);
        if (filtered & LatticeClasses::QUOTE) {
            orbits.emplace_back("string_orbit", 0.85 * prediction_accuracy);
        }

        // Structural boundary: STRUCTURAL | PUNCTUATION | SEMICOLON
        uint16_t boundary_mask = LatticeClasses::STRUCTURAL | LatticeClasses::PUNCTUATION | LatticeClasses::SEMICOLON;
        filtered = apply_grid_or_mask(boundary_mask);
        if (filtered & LatticeClasses::STRUCTURAL) {
            orbits.emplace_back("structural_boundary", 0.95 * prediction_accuracy);
        }

        // Identifier orbit: IDENTIFIER & ~PUNCTUATION
        uint16_t ident_mask = LatticeClasses::IDENTIFIER;
        ident_mask &= ~LatticeClasses::PUNCTUATION;
        filtered = apply_grid_and_mask(ident_mask);
        if (filtered & LatticeClasses::IDENTIFIER) {
            orbits.emplace_back("identifier_orbit", 0.8 * prediction_accuracy);
        }

        return orbits;
    }

    /**
     * Check if a specific lattice pattern is present in this grid
     * Used for pattern pre-filtering
     */
    bool has_lattice_pattern(uint16_t pattern) const {
        return (grid_mask & pattern) != 0;
    }
};

} // namespace cppfort::stage0
