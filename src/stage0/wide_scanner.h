#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "orbit_mask.h"  // For OrbitContext
#include "orbit_ring.h"
#include "packrat_cache.h"

namespace cppfort {
namespace stage0 {
    struct AnchorTuple;  // Forward declaration
}
}

namespace cppfort::ir {

// Wide scanner for alternating anchor detection with SIMD acceleration
// Integrated with XAI 4.2 orbit system (5-anchor tuples)
class WideScanner {
public:
    // Anchor point structure
    struct AnchorPoint {
        size_t position;      // Byte position in source
        size_t spacing;       // Distance to next anchor (64 or 32)
        bool is_utf8_boundary; // True if at UTF-8 character boundary
    };

    // Boundary detection result with XAI 4.2 orbit data
    struct FanoutStats {
        size_t processed_bytes = 0;
        size_t vector_iterations = 0;
        size_t scalar_iterations = 0;
    };

    struct Boundary {
        size_t position;
        char delimiter;  // Character at boundary (if delimiter)
        bool is_delimiter; // True if this is a delimiter boundary

        // XAI 4.2 orbit metadata (populated by scanAnchorsWithOrbits)
        uint16_t lattice_mask;  // Byte-level class (from HeuristicGrid)
        double orbit_confidence; // Composite confidence from 5 anchors
    };

    // Generate alternating anchor points at UTF-8 boundaries
    // Initial spacing of 64 bytes, alternates to 32, then back to 64
    static ::std::vector<AnchorPoint> generateAlternatingAnchors(
        const ::std::string& source,
        size_t initial_spacing = 64
    );

    // SIMD-accelerated scanning between anchor points
    // Detects UTF-8 boundaries and common delimiters: ; , { } ( ) [ ]
    static ::std::vector<Boundary> scanAnchorsSIMD(
        const ::std::string& source,
        const ::std::vector<AnchorPoint>& anchors
    );

    // XAI 4.2 orbit-aware scanning with 5-anchor tuple detection
    // Returns boundaries with orbit metadata (lattice mask + confidence)
    ::std::vector<Boundary> scanAnchorsWithOrbits(
        const ::std::string& source,
        const ::std::vector<AnchorPoint>& anchors
    );

    const ::std::vector<::cppfort::stage0::OrbitFragment>& fragments() const { return fragments_; }
    const FanoutStats& stats() const { return stats_; }

    // Generate XAI 4.2 AnchorTuple for a given span
    // All 5 anchor types fire concurrently
    static ::std::vector<stage0::AnchorTuple> generateOrbitTuples(
        const ::std::string& source,
        size_t chunk_size = 4096  // Default: 4KB chunks (from HeuristicGrid)
    );

    // Find next UTF-8 boundary using SIMD (processes 16 bytes at a time)
    // Returns offset from current position, or npos if not found
    static size_t findBoundarySIMD(
        const uint8_t* data,
        size_t position,
        size_t remaining
    );

    // Check if position is at UTF-8 boundary (not a continuation byte)
    static bool isUTF8Boundary(const uint8_t* data, size_t position);

    // Constructor to initialize OrbitContext
    WideScanner(size_t max_orbit_depth = 100) : orbit_context_(max_orbit_depth) {}

private:
    // Orbit context for tracking structural balance during scanning
    OrbitContext orbit_context_;
    ::cppfort::stage0::PackratCache packrat_cache_;
    FanoutStats stats_;
    ::std::vector<::cppfort::stage0::OrbitFragment> fragments_;

    // SIMD delimiter detection helpers
    static bool hasDelimiter(const uint8_t* data, size_t len, char delim);
    static int findDelimiterMask(const uint8_t* data, size_t len);
    void reset_stats();
};

} // namespace cppfort::ir
