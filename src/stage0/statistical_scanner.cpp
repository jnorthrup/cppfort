#include "statistical_scanner.h"

#include <algorithm>
#include <cmath>
#include <cstring>

// Platform-specific SIMD intrinsics
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#include <immintrin.h>
#define CPPFORT_X86_SIMD 1
#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define CPPFORT_ARM_NEON 1
#endif

namespace cppfort::ir {

/**
 * Branch-free SIMD chunk scanning (16 bytes at a time)
 * Returns position masks for delimiters and UTF-8 boundaries
 */
StatisticalScanner::ScanMask StatisticalScanner::scanChunk16(const uint8_t* data) {
    ScanMask result;
    result.delimiter_mask = 0;
    result.utf8_mask = 0;
    std::memset(result.delimiter_types, 0xFF, 16);

#if defined(CPPFORT_ARM_NEON)
    // ARM NEON: parallel comparison for all delimiters
    uint8x16_t chunk = vld1q_u8(data);

    // UTF-8 boundary detection: (byte & 0xC0) != 0x80
    uint8x16_t high_bits = vandq_u8(chunk, vdupq_n_u8(0xC0));
    uint8x16_t is_continuation = vceqq_u8(high_bits, vdupq_n_u8(0x80));
    uint8x16_t is_boundary = vmvnq_u8(is_continuation);  // NOT continuation = boundary

    // Delimiter detection (8 parallel comparisons)
    uint8x16_t is_semicolon  = vceqq_u8(chunk, vdupq_n_u8(';'));  // 0
    uint8x16_t is_comma      = vceqq_u8(chunk, vdupq_n_u8(','));  // 1
    uint8x16_t is_lbrace     = vceqq_u8(chunk, vdupq_n_u8('{'));  // 2
    uint8x16_t is_rbrace     = vceqq_u8(chunk, vdupq_n_u8('}'));  // 3
    uint8x16_t is_lparen     = vceqq_u8(chunk, vdupq_n_u8('('));  // 4
    uint8x16_t is_rparen     = vceqq_u8(chunk, vdupq_n_u8(')'));  // 5
    uint8x16_t is_lbracket   = vceqq_u8(chunk, vdupq_n_u8('['));  // 6
    uint8x16_t is_rbracket   = vceqq_u8(chunk, vdupq_n_u8(']'));  // 7

    // Combine all delimiter masks
    uint8x16_t any_delimiter = vorrq_u8(
        vorrq_u8(vorrq_u8(is_semicolon, is_comma), vorrq_u8(is_lbrace, is_rbrace)),
        vorrq_u8(vorrq_u8(is_lparen, is_rparen), vorrq_u8(is_lbracket, is_rbracket))
    );

    // Extract bit masks (branch-free)
    uint8_t boundary_bytes[16];
    uint8_t delimiter_bytes[16];
    vst1q_u8(boundary_bytes, is_boundary);
    vst1q_u8(delimiter_bytes, any_delimiter);

    // Build position masks (branch-free using arithmetic)
    for (int i = 0; i < 16; ++i) {
        result.utf8_mask |= ((boundary_bytes[i] != 0) << i);
        result.delimiter_mask |= ((delimiter_bytes[i] != 0) << i);

        // Encode delimiter type (branch-free arithmetic cascade)
        uint8_t delim_type = 0xFF;
        uint8_t d0[16], d1[16], d2[16], d3[16], d4[16], d5[16], d6[16], d7[16];
        vst1q_u8(d0, is_semicolon);
        vst1q_u8(d1, is_comma);
        vst1q_u8(d2, is_lbrace);
        vst1q_u8(d3, is_rbrace);
        vst1q_u8(d4, is_lparen);
        vst1q_u8(d5, is_rparen);
        vst1q_u8(d6, is_lbracket);
        vst1q_u8(d7, is_rbracket);

        // Branch-free selection (only one will be 0xFF, rest are 0x00)
        delim_type = (d0[i] != 0) ? 0 : delim_type;
        delim_type = (d1[i] != 0) ? 1 : delim_type;
        delim_type = (d2[i] != 0) ? 2 : delim_type;
        delim_type = (d3[i] != 0) ? 3 : delim_type;
        delim_type = (d4[i] != 0) ? 4 : delim_type;
        delim_type = (d5[i] != 0) ? 5 : delim_type;
        delim_type = (d6[i] != 0) ? 6 : delim_type;
        delim_type = (d7[i] != 0) ? 7 : delim_type;

        result.delimiter_types[i] = delim_type;
    }

#elif defined(CPPFORT_X86_SIMD)
    // x86 SSE: parallel comparison for all delimiters
    __m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data));

    // UTF-8 boundary detection
    __m128i high_bits = _mm_and_si128(chunk, _mm_set1_epi8(0xC0));
    __m128i is_continuation = _mm_cmpeq_epi8(high_bits, _mm_set1_epi8(0x80));
    __m128i is_boundary = _mm_andnot_si128(is_continuation, _mm_set1_epi8(0xFF));

    // Delimiter detection (8 parallel comparisons)
    __m128i is_semicolon  = _mm_cmpeq_epi8(chunk, _mm_set1_epi8(';'));
    __m128i is_comma      = _mm_cmpeq_epi8(chunk, _mm_set1_epi8(','));
    __m128i is_lbrace     = _mm_cmpeq_epi8(chunk, _mm_set1_epi8('{'));
    __m128i is_rbrace     = _mm_cmpeq_epi8(chunk, _mm_set1_epi8('}'));
    __m128i is_lparen     = _mm_cmpeq_epi8(chunk, _mm_set1_epi8('('));
    __m128i is_rparen     = _mm_cmpeq_epi8(chunk, _mm_set1_epi8(')'));
    __m128i is_lbracket   = _mm_cmpeq_epi8(chunk, _mm_set1_epi8('['));
    __m128i is_rbracket   = _mm_cmpeq_epi8(chunk, _mm_set1_epi8(']'));

    // Combine all delimiter masks
    __m128i any_delimiter = _mm_or_si128(
        _mm_or_si128(
            _mm_or_si128(is_semicolon, is_comma),
            _mm_or_si128(is_lbrace, is_rbrace)
        ),
        _mm_or_si128(
            _mm_or_si128(is_lparen, is_rparen),
            _mm_or_si128(is_lbracket, is_rbracket)
        )
    );

    // Extract bit masks (branch-free)
    result.utf8_mask = static_cast<uint16_t>(_mm_movemask_epi8(is_boundary));
    result.delimiter_mask = static_cast<uint16_t>(_mm_movemask_epi8(any_delimiter));

    // Encode delimiter types (branch-free using movemask)
    uint16_t m0 = _mm_movemask_epi8(is_semicolon);
    uint16_t m1 = _mm_movemask_epi8(is_comma);
    uint16_t m2 = _mm_movemask_epi8(is_lbrace);
    uint16_t m3 = _mm_movemask_epi8(is_rbrace);
    uint16_t m4 = _mm_movemask_epi8(is_lparen);
    uint16_t m5 = _mm_movemask_epi8(is_rparen);
    uint16_t m6 = _mm_movemask_epi8(is_lbracket);
    uint16_t m7 = _mm_movemask_epi8(is_rbracket);

    // Branch-free type encoding
    for (int i = 0; i < 16; ++i) {
        uint8_t delim_type = 0xFF;
        delim_type = ((m0 >> i) & 1) ? 0 : delim_type;
        delim_type = ((m1 >> i) & 1) ? 1 : delim_type;
        delim_type = ((m2 >> i) & 1) ? 2 : delim_type;
        delim_type = ((m3 >> i) & 1) ? 3 : delim_type;
        delim_type = ((m4 >> i) & 1) ? 4 : delim_type;
        delim_type = ((m5 >> i) & 1) ? 5 : delim_type;
        delim_type = ((m6 >> i) & 1) ? 6 : delim_type;
        delim_type = ((m7 >> i) & 1) ? 7 : delim_type;
        result.delimiter_types[i] = delim_type;
    }

#else
    // Scalar fallback (still branch-free per byte)
    for (int i = 0; i < 16; ++i) {
        uint8_t byte = data[i];

        // UTF-8 boundary: (byte & 0xC0) != 0x80
        uint8_t is_boundary = ((byte & 0xC0) != 0x80);
        result.utf8_mask |= (is_boundary << i);

        // Delimiter check (branch-free using comparisons)
        uint8_t is_delim = (byte == ';') | (byte == ',') |
                          (byte == '{') | (byte == '}') |
                          (byte == '(') | (byte == ')') |
                          (byte == '[') | (byte == ']');
        result.delimiter_mask |= (is_delim << i);

        // Encode delimiter type (branch-free)
        result.delimiter_types[i] = encodeDelimiter(static_cast<char>(byte));
    }
#endif

    return result;
}

/**
 * Non-branching full scan: generate masks for entire input
 * Pure streaming dataflow, no early exits
 */
std::vector<StatisticalScanner::ScanMask> StatisticalScanner::scanPositionMasks(
    const uint8_t* data,
    size_t length
) {
    std::vector<ScanMask> masks;
    size_t num_chunks = (length + 15) / 16;  // Round up
    masks.reserve(num_chunks);

    // Process all full 16-byte chunks (no branching on completion)
    for (size_t i = 0; i < num_chunks; ++i) {
        ScanMask mask = scanChunk16(data + (i * 16));
        mask.chunk_index = i;
        masks.push_back(mask);
    }

    return masks;
}

/**
 * Gather positions from masks (branch-free scatter using popcount)
 */
std::vector<size_t> StatisticalScanner::gatherPositions(
    const std::vector<ScanMask>& masks
) {
    std::vector<size_t> positions;

    // Pre-allocate conservative upper bound (max 16 delimiters per chunk)
    positions.reserve(masks.size() * 16);

    for (const auto& mask : masks) {
        size_t base_offset = mask.chunk_index * 16;
        uint16_t delim_mask = mask.delimiter_mask;

        // Extract positions using bit manipulation (branch-free iteration)
        while (delim_mask != 0) {
            // Find position of lowest set bit (CTZ = count trailing zeros)
            int bit_pos = __builtin_ctz(delim_mask);
            positions.push_back(base_offset + bit_pos);

            // Clear lowest set bit (branch-free)
            delim_mask &= (delim_mask - 1);
        }
    }

    return positions;
}

/**
 * Compute delimiter histogram (pure counting, no branches)
 */
StatisticalScanner::DelimiterHistogram StatisticalScanner::computeHistogram(
    const std::vector<ScanMask>& masks
) {
    DelimiterHistogram hist;
    hist.counts.fill(0);

    for (const auto& mask : masks) {
        hist.total_bytes += 16;  // Each mask covers 16 bytes

        // Count each delimiter type (vectorized)
        for (int i = 0; i < 16; ++i) {
            uint8_t dtype = mask.delimiter_types[i];

            // Branch-free counting (only valid types 0-7)
            bool valid = (dtype < 8);
            hist.counts[dtype & 7] += valid;  // Mask to prevent overflow
            hist.total_delimiters += valid;
        }
    }

    // Compute entropy
    hist.entropy = computeEntropy(hist);

    return hist;
}

/**
 * Shannon entropy: H = -Σ(p_i * log2(p_i))
 * Branch-free using conditional moves
 */
float StatisticalScanner::computeEntropy(const DelimiterHistogram& hist) {
    if (hist.total_delimiters == 0) return 0.0f;

    float entropy = 0.0f;
    float total = static_cast<float>(hist.total_delimiters);

    for (uint32_t count : hist.counts) {
        if (count > 0) {
            float p = static_cast<float>(count) / total;
            entropy -= p * std::log2(p);
        }
    }

    return entropy;
}

} // namespace cppfort::ir
