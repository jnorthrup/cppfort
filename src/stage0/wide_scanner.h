#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cppfort::ir {

// Wide scanner for alternating anchor detection with SIMD acceleration
class WideScanner {
public:
    // Anchor point structure
    struct AnchorPoint {
        size_t position;      // Byte position in source
        size_t spacing;       // Distance to next anchor (64 or 32)
        bool is_utf8_boundary; // True if at UTF-8 character boundary
    };

    // Boundary detection result
    struct Boundary {
        size_t position;
        char delimiter;  // Character at boundary (if delimiter)
        bool is_delimiter; // True if this is a delimiter boundary
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

    // Find next UTF-8 boundary using SIMD (processes 16 bytes at a time)
    // Returns offset from current position, or npos if not found
    static size_t findBoundarySIMD(
        const uint8_t* data,
        size_t position,
        size_t remaining
    );

    // Check if position is at UTF-8 boundary (not a continuation byte)
    static bool isUTF8Boundary(const uint8_t* data, size_t position);

private:
    // SIMD delimiter detection helpers
    static bool hasDelimiter(const uint8_t* data, size_t len, char delim);
    static int findDelimiterMask(const uint8_t* data, size_t len);
};

} // namespace cppfort::ir
