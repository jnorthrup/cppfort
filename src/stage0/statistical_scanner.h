#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <array>

namespace cppfort::ir {

/**
 * Statistical non-branching SIMD scanner
 * Pure dataflow: input bytes → SIMD masks → position vectors
 * No conditionals in hot path, gather-scatter only
 */
class StatisticalScanner {
public:
    // Delimiter histogram (8 delimiters: ; , { } ( ) [ ])
    struct DelimiterHistogram {
        std::array<uint32_t, 8> counts;  // Counts per delimiter type
        uint32_t total_delimiters {0};
        uint32_t total_bytes {0};
        float entropy {0.0f};  // Shannon entropy of delimiter distribution
    };

    // SIMD scan result: position masks for all 16-byte chunks
    struct ScanMask {
        size_t chunk_index;        // Which 16-byte chunk (offset / 16)
        uint16_t delimiter_mask;   // Bit i = 1 if byte i is delimiter
        uint16_t utf8_mask;        // Bit i = 1 if byte i is UTF-8 boundary
        uint8_t delimiter_types[16]; // Delimiter type index (0-7), or 0xFF if not delimiter
    };

    /**
     * Non-branching SIMD scan: generate position masks for entire input
     * Returns vector of masks, one per 16-byte chunk
     * Pure dataflow: no early exits, no conditionals
     */
    static std::vector<ScanMask> scanPositionMasks(
        const uint8_t* data,
        size_t length
    );

    /**
     * Gather delimiter positions from masks (branch-free scatter)
     * Input: position masks from scanPositionMasks
     * Output: sorted array of delimiter positions
     */
    static std::vector<size_t> gatherPositions(
        const std::vector<ScanMask>& masks
    );

    /**
     * Compute delimiter histogram (pure counting, no branches)
     * Statistical properties for adaptive parsing
     */
    static DelimiterHistogram computeHistogram(
        const std::vector<ScanMask>& masks
    );

    /**
     * Compute Shannon entropy of delimiter distribution
     * H = -Σ(p_i * log2(p_i))
     * Returns bits of information per delimiter
     */
    static float computeEntropy(const DelimiterHistogram& hist);

private:
    // SIMD helpers (platform-specific, but interface is branch-free)
    static ScanMask scanChunk16(const uint8_t* data);

    // Delimiter type encoding: 0=; 1=, 2={ 3=} 4=( 5=) 6=[ 7=]
    static constexpr uint8_t encodeDelimiter(char ch) {
        // Branch-free encoding using arithmetic
        return (ch == ';') ? 0 :
               (ch == ',') ? 1 :
               (ch == '{') ? 2 :
               (ch == '}') ? 3 :
               (ch == '(') ? 4 :
               (ch == ')') ? 5 :
               (ch == '[') ? 6 :
               (ch == ']') ? 7 : 0xFF;
    }
};

} // namespace cppfort::ir
