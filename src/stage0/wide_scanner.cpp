#include "wide_scanner.h"

#include <algorithm>
#include <cstring>

// Platform-specific SIMD intrinsics
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#include <immintrin.h> // x86 SSE/AVX
#define CPPFORT_X86_SIMD 1
#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h> // ARM NEON
#define CPPFORT_ARM_NEON 1
#endif

#include "heuristic_grid.h"
#include "lattice_classes.h"

namespace cppfort::ir {

bool WideScanner::isUTF8Boundary(const uint8_t* data, size_t position) {
    // UTF-8 continuation bytes have pattern 10xxxxxx
    // Lead bytes and ASCII have pattern 0xxxxxxx or 11xxxxxx
    uint8_t byte = data[position];
    return (byte & 0xC0) != 0x80;
}

size_t WideScanner::findBoundarySIMD(
    const uint8_t* data,
    size_t position,
    size_t remaining
) {
    size_t pos = position;

#if defined(CPPFORT_ARM_NEON)
    // ARM NEON: process 16 bytes at a time
    while (remaining >= 16) {
        uint8x16_t chunk = vld1q_u8(data + pos);

        // Mask for UTF-8 continuation bytes (10xxxxxx)
        // High bits: check if (byte & 0xC0) != 0x80
        uint8x16_t mask_c0 = vdupq_n_u8(0xC0);
        uint8x16_t high_bits = vandq_u8(chunk, mask_c0);

        // Boundaries are where high_bits != 0x80
        // Check for ASCII (0x00) or lead bytes (0xC0)
        uint8x16_t is_ascii = vceqq_u8(high_bits, vdupq_n_u8(0x00));
        uint8x16_t is_lead = vceqq_u8(high_bits, vdupq_n_u8(0xC0));
        uint8x16_t boundary = vorrq_u8(is_ascii, is_lead);

        // Extract mask from comparison result
        uint64_t mask_low = vgetq_lane_u64(vreinterpretq_u64_u8(boundary), 0);
        uint64_t mask_high = vgetq_lane_u64(vreinterpretq_u64_u8(boundary), 1);

        if (mask_low != 0) {
            // Found boundary in lower 8 bytes
            for (int i = 0; i < 8; ++i) {
                if ((mask_low >> (i * 8)) & 0xFF) {
                    return (pos - position) + i;
                }
            }
        }
        if (mask_high != 0) {
            // Found boundary in upper 8 bytes
            for (int i = 0; i < 8; ++i) {
                if ((mask_high >> (i * 8)) & 0xFF) {
                    return (pos - position) + 8 + i;
                }
            }
        }

        pos += 16;
        remaining -= 16;
    }
#elif defined(CPPFORT_X86_SIMD)
    // x86 SSE: process 16 bytes at a time
    while (remaining >= 16) {
        __m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + pos));

        // Mask for UTF-8 continuation bytes (10xxxxxx)
        __m128i high_bits = _mm_and_si128(chunk, _mm_set1_epi8(0xC0));

        // Boundaries: ASCII (0x00) or lead bytes (0xC0)
        __m128i is_boundary = _mm_cmpeq_epi8(high_bits, _mm_set1_epi8(0x00));
        __m128i is_lead = _mm_cmpeq_epi8(high_bits, _mm_set1_epi8(0xC0));
        __m128i boundary = _mm_or_si128(is_boundary, is_lead);

        int mask = _mm_movemask_epi8(boundary);
        if (mask != 0) {
            int bit_pos = __builtin_ctz(mask);
            return (pos - position) + bit_pos;
        }

        pos += 16;
        remaining -= 16;
    }
#endif

    // Scalar fallback for remaining bytes or no SIMD
    for (size_t i = 0; i < remaining; ++i) {
        if (isUTF8Boundary(data, pos + i)) {
            return (pos - position) + i;
        }
    }

    return ::std::string::npos;
}

::std::vector<WideScanner::AnchorPoint> WideScanner::generateAlternatingAnchors(
    const ::std::string& source,
    size_t initial_spacing
) {
    ::std::vector<AnchorPoint> anchors;

    const uint8_t* data = reinterpret_cast<const uint8_t*>(source.data());
    size_t buf_size = source.size();
    size_t current_pos = 0;
    size_t anchor_spacing = initial_spacing ? initial_spacing : 64;

    while (current_pos < buf_size) {
        AnchorPoint anchor;
        anchor.position = current_pos;
        anchor.spacing = anchor_spacing;
        anchor.is_utf8_boundary = isUTF8Boundary(data, current_pos);
        anchors.push_back(anchor);

        size_t next_target = current_pos + anchor_spacing;
        if (next_target >= buf_size) {
            break;
        }

        size_t remaining = buf_size - next_target;
        size_t boundary_offset = findBoundarySIMD(data, next_target, remaining);
        if (boundary_offset != ::std::string::npos) {
            current_pos = next_target + boundary_offset;
        } else {
            current_pos = next_target;
        }

        anchor_spacing = (anchor_spacing == 64) ? 32 : 64;
    }

    return anchors;
}

::std::vector<WideScanner::Boundary> WideScanner::scanAnchorsSIMD(
    const ::std::string& source,
    const ::std::vector<AnchorPoint>& anchors
) {
    ::std::vector<Boundary> boundaries;

    const uint8_t* data = reinterpret_cast<const uint8_t*>(source.data());

    // Scan between consecutive anchor pairs
    for (size_t i = 0; i < anchors.size() - 1; ++i) {
        size_t start_pos = anchors[i].position;
        size_t end_pos = anchors[i + 1].position;

        if (end_pos <= start_pos) continue;

        size_t pos = start_pos;
        size_t remaining = end_pos - start_pos;

        // SIMD scan within anchor range
#if defined(CPPFORT_ARM_NEON)
        while (remaining >= 16 && pos < end_pos) {
            uint8x16_t chunk = vld1q_u8(data + pos);

            // Check for UTF-8 boundaries
            uint8x16_t high_bits = vandq_u8(chunk, vdupq_n_u8(0xC0));
            uint8x16_t utf8_boundary = vceqq_u8(high_bits, vdupq_n_u8(0x00));

            // Check for delimiters
            uint8x16_t semicolon = vceqq_u8(chunk, vdupq_n_u8(';'));
            uint8x16_t comma = vceqq_u8(chunk, vdupq_n_u8(','));
            uint8x16_t colon = vceqq_u8(chunk, vdupq_n_u8(':'));
            uint8x16_t lbrace = vceqq_u8(chunk, vdupq_n_u8('{'));
            uint8x16_t rbrace = vceqq_u8(chunk, vdupq_n_u8('}'));
            uint8x16_t lparen = vceqq_u8(chunk, vdupq_n_u8('('));
            uint8x16_t rparen = vceqq_u8(chunk, vdupq_n_u8(')'));
            uint8x16_t lbracket = vceqq_u8(chunk, vdupq_n_u8('['));
            uint8x16_t rbracket = vceqq_u8(chunk, vdupq_n_u8(']'));

            // Combine all checks
            uint8x16_t delimiters = vorrq_u8(
                vorrq_u8(vorrq_u8(semicolon, comma), vorrq_u8(colon, lbrace)),
                vorrq_u8(vorrq_u8(rbrace, lparen), vorrq_u8(vorrq_u8(rparen, lbracket), rbracket))
            );
            uint8x16_t combined = vorrq_u8(utf8_boundary, delimiters);

            // Check if any matches
            uint64_t mask_low = vgetq_lane_u64(vreinterpretq_u64_u8(combined), 0);
            uint64_t mask_high = vgetq_lane_u64(vreinterpretq_u64_u8(combined), 1);

            if (mask_low != 0 || mask_high != 0) {
                // Scan bytes to find exact position
                for (size_t i = 0; i < 16 && (pos + i) < end_pos; ++i) {
                    char ch = static_cast<char>(data[pos + i]);
                    bool is_delim = (ch == ';' || ch == ',' || ch == ':' || ch == '{' || ch == '}' ||
                                     ch == '(' || ch == ')' || ch == '[' || ch == ']');
                    if (is_delim || isUTF8Boundary(data, pos + i)) {
                        Boundary boundary;
                        boundary.position = pos + i;
                        boundary.delimiter = ch;
                        boundary.is_delimiter = is_delim;
                        boundaries.push_back(boundary);
                        break;
                    }
                }
            }

            size_t advance = ::std::min(size_t(16), remaining);
            pos += advance;
            remaining -= advance;
        }
#elif defined(CPPFORT_X86_SIMD)
        while (remaining >= 16 && pos < end_pos) {
            __m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + pos));

            // Check for UTF-8 boundaries
            __m128i high_bits = _mm_and_si128(chunk, _mm_set1_epi8(0xC0));
            __m128i utf8_boundary = _mm_cmpeq_epi8(high_bits, _mm_set1_epi8(0x00));

            // Check for common delimiters
            __m128i semicolon = _mm_cmpeq_epi8(chunk, _mm_set1_epi8(';'));
            __m128i comma = _mm_cmpeq_epi8(chunk, _mm_set1_epi8(','));
            __m128i colon = _mm_cmpeq_epi8(chunk, _mm_set1_epi8(':'));
            __m128i lbrace = _mm_cmpeq_epi8(chunk, _mm_set1_epi8('{'));
            __m128i rbrace = _mm_cmpeq_epi8(chunk, _mm_set1_epi8('}'));
            __m128i lparen = _mm_cmpeq_epi8(chunk, _mm_set1_epi8('('));
            __m128i rparen = _mm_cmpeq_epi8(chunk, _mm_set1_epi8(')'));
            __m128i lbracket = _mm_cmpeq_epi8(chunk, _mm_set1_epi8('['));
            __m128i rbracket = _mm_cmpeq_epi8(chunk, _mm_set1_epi8(']'));

            // Combine all delimiter checks
            __m128i delimiters = _mm_or_si128(
                _mm_or_si128(
                    _mm_or_si128(semicolon, comma),
                    _mm_or_si128(colon, lbrace)
                ),
                _mm_or_si128(
                    _mm_or_si128(rbrace, lparen),
                    _mm_or_si128(
                        _mm_or_si128(rparen, lbracket),
                        rbracket
                    )
                )
            );

            __m128i combined = _mm_or_si128(utf8_boundary, delimiters);
            int mask = _mm_movemask_epi8(combined);

            if (mask != 0) {
                // Find first match
                int bit_pos = __builtin_ctz(mask);
                size_t boundary_pos = pos + bit_pos;

                if (boundary_pos < end_pos) {
                    Boundary boundary;
                    boundary.position = boundary_pos;
                    boundary.delimiter = static_cast<char>(data[boundary_pos]);
                    boundary.is_delimiter = (
                        boundary.delimiter == ';' ||
                        boundary.delimiter == ',' ||
                        boundary.delimiter == ':' ||
                        boundary.delimiter == '{' ||
                        boundary.delimiter == '}' ||
                        boundary.delimiter == '(' ||
                        boundary.delimiter == ')' ||
                        boundary.delimiter == '[' ||
                        boundary.delimiter == ']'
                    );
                    boundaries.push_back(boundary);
                }
            }

            size_t advance = ::std::min(size_t(16), remaining);
            pos += advance;
            remaining -= advance;
        }
#endif

        // Scalar fallback for remaining bytes in this range
        while (pos < end_pos) {
            char ch = static_cast<char>(data[pos]);
            bool is_delim = (
                ch == ';' || ch == ',' || ch == ':' ||
                ch == '{' || ch == '}' ||
                ch == '(' || ch == ')' ||
                ch == '[' || ch == ']'
            );

            if (is_delim || isUTF8Boundary(data, pos)) {
                Boundary boundary;
                boundary.position = pos;
                boundary.delimiter = ch;
                boundary.is_delimiter = is_delim;
                boundaries.push_back(boundary);
            }

            pos++;
        }
    }

    return boundaries;
}

::std::vector<WideScanner::Boundary> WideScanner::scanAnchorsWithOrbits(
    const ::std::string& source,
    const ::std::vector<AnchorPoint>& anchors
) {
    ::std::vector<Boundary> boundaries;

    const uint8_t* data = reinterpret_cast<const uint8_t*>(source.data());
    orbit_context_.reset();  // Reset context between scans

    // Handle small files with single anchor: scan entire buffer
    if (anchors.size() <= 1) {
        for (size_t pos = 0; pos < source.size(); ++pos) {
            char ch = static_cast<char>(data[pos]);
            orbit_context_.update(ch);

            bool is_delim = (
                ch == ';' || ch == ',' || ch == ':' ||
                ch == '{' || ch == '}' ||
                ch == '(' || ch == ')' ||
                ch == '[' || ch == ']'
            );

            if (is_delim) {
                Boundary boundary;
                boundary.position = pos;
                boundary.delimiter = ch;
                boundary.is_delimiter = true;
                boundary.lattice_mask = stage0::classify_byte(ch);
                boundary.orbit_confidence = orbit_context_.calculateConfidence();
                boundaries.push_back(boundary);
            }
        }
        return boundaries;
    }

    // Scan between consecutive anchor pairs
    for (size_t i = 0; i < anchors.size() - 1; ++i) {
        size_t start_pos = anchors[i].position;
        size_t end_pos = anchors[i + 1].position;

        if (end_pos <= start_pos) continue;

        size_t pos = start_pos;
        size_t remaining = end_pos - start_pos;

        // Reset orbit context for each anchor span
        orbit_context_.reset();

        // SIMD scan within anchor range with orbit tracking
#if defined(CPPFORT_ARM_NEON)
        while (remaining >= 16 && pos < end_pos) {
            uint8x16_t chunk = vld1q_u8(data + pos);

            // Check for UTF-8 boundaries
            uint8x16_t high_bits = vandq_u8(chunk, vdupq_n_u8(0xC0));
            uint8x16_t utf8_boundary = vceqq_u8(high_bits, vdupq_n_u8(0x00));

            // Check for delimiters
            uint8x16_t semicolon = vceqq_u8(chunk, vdupq_n_u8(';'));
            uint8x16_t comma = vceqq_u8(chunk, vdupq_n_u8(','));
            uint8x16_t colon = vceqq_u8(chunk, vdupq_n_u8(':'));
            uint8x16_t lbrace = vceqq_u8(chunk, vdupq_n_u8('{'));
            uint8x16_t rbrace = vceqq_u8(chunk, vdupq_n_u8('}'));
            uint8x16_t lparen = vceqq_u8(chunk, vdupq_n_u8('('));
            uint8x16_t rparen = vceqq_u8(chunk, vdupq_n_u8(')'));
            uint8x16_t lbracket = vceqq_u8(chunk, vdupq_n_u8('['));
            uint8x16_t rbracket = vceqq_u8(chunk, vdupq_n_u8(']'));

            // Combine all checks
            uint8x16_t delimiters = vorrq_u8(
                vorrq_u8(vorrq_u8(semicolon, comma), vorrq_u8(colon, lbrace)),
                vorrq_u8(vorrq_u8(rbrace, lparen), vorrq_u8(vorrq_u8(rparen, lbracket), rbracket))
            );
            uint8x16_t combined = vorrq_u8(utf8_boundary, delimiters);

            // Check if any matches
            uint64_t mask_low = vgetq_lane_u64(vreinterpretq_u64_u8(combined), 0);
            uint64_t mask_high = vgetq_lane_u64(vreinterpretq_u64_u8(combined), 1);

            if (mask_low != 0 || mask_high != 0) {
                // Scan bytes to find exact position
                for (size_t i = 0; i < 16 && (pos + i) < end_pos; ++i) {
                    char ch = static_cast<char>(data[pos + i]);
                    bool is_delim = (ch == ';' || ch == ',' || ch == ':' || ch == '{' || ch == '}' ||
                                     ch == '(' || ch == ')' || ch == '[' || ch == ']');
                    if (is_delim) {
                        // Update orbit context for each byte scanned
                        for (size_t j = 0; j <= i; ++j) {
                            orbit_context_.update(static_cast<char>(data[pos + j]));
                        }

                        Boundary boundary;
                        boundary.position = pos + i;
                        boundary.delimiter = ch;
                        boundary.is_delimiter = true;

                        // Populate lattice_mask using HeuristicGrid classification
                        boundary.lattice_mask = stage0::classify_byte(ch);

                        // Populate orbit_confidence using OrbitContext
                        boundary.orbit_confidence = orbit_context_.calculateConfidence();

                        boundaries.push_back(boundary);
                        break;
                    }
                }
            }

            size_t advance = ::std::min(size_t(16), remaining);
            pos += advance;
            remaining -= advance;
        }
#elif defined(CPPFORT_X86_SIMD)
        while (remaining >= 16 && pos < end_pos) {
            __m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + pos));

            // Check for UTF-8 boundaries
            __m128i high_bits = _mm_and_si128(chunk, _mm_set1_epi8(0xC0));
            __m128i utf8_boundary = _mm_cmpeq_epi8(high_bits, _mm_set1_epi8(0x00));

            // Check for common delimiters
            __m128i semicolon = _mm_cmpeq_epi8(chunk, _mm_set1_epi8(';'));
            __m128i comma = _mm_cmpeq_epi8(chunk, _mm_set1_epi8(','));
            __m128i colon = _mm_cmpeq_epi8(chunk, _mm_set1_epi8(':'));
            __m128i lbrace = _mm_cmpeq_epi8(chunk, _mm_set1_epi8('{'));
            __m128i rbrace = _mm_cmpeq_epi8(chunk, _mm_set1_epi8('}'));
            __m128i lparen = _mm_cmpeq_epi8(chunk, _mm_set1_epi8('('));
            __m128i rparen = _mm_cmpeq_epi8(chunk, _mm_set1_epi8(')'));
            __m128i lbracket = _mm_cmpeq_epi8(chunk, _mm_set1_epi8('['));
            __m128i rbracket = _mm_cmpeq_epi8(chunk, _mm_set1_epi8(']'));

            // Combine all delimiter checks
            __m128i delimiters = _mm_or_si128(
                _mm_or_si128(
                    _mm_or_si128(semicolon, comma),
                    _mm_or_si128(colon, lbrace)
                ),
                _mm_or_si128(
                    _mm_or_si128(rbrace, lparen),
                    _mm_or_si128(
                        _mm_or_si128(rparen, lbracket),
                        rbracket
                    )
                )
            );

            __m128i combined = _mm_or_si128(utf8_boundary, delimiters);
            int mask = _mm_movemask_epi8(combined);

            if (mask != 0) {
                // Find first match
                int bit_pos = __builtin_ctz(mask);
                size_t boundary_pos = pos + bit_pos;

                if (boundary_pos < end_pos) {
                    char ch = static_cast<char>(data[boundary_pos]);
                    bool is_delim = (
                        ch == ';' || ch == ',' || ch == ':' || ch == '{' || ch == '}' ||
                        ch == '(' || ch == ')' || ch == '[' || ch == ']'
                    );

                    if (is_delim) {
                        // Update orbit context for each byte up to this boundary
                        for (size_t j = 0; j <= bit_pos; ++j) {
                            orbit_context_.update(static_cast<char>(data[pos + j]));
                        }

                        Boundary boundary;
                        boundary.position = boundary_pos;
                        boundary.delimiter = ch;
                        boundary.is_delimiter = true;

                        // Populate lattice_mask using HeuristicGrid classification
                        boundary.lattice_mask = stage0::classify_byte(ch);

                        // Populate orbit_confidence using OrbitContext
                        boundary.orbit_confidence = orbit_context_.calculateConfidence();

                        boundaries.push_back(boundary);
                    }
                }
            }

            size_t advance = ::std::min(size_t(16), remaining);
            pos += advance;
            remaining -= advance;
        }
#endif

        // Scalar fallback for remaining bytes in this range
        while (pos < end_pos) {
            char ch = static_cast<char>(data[pos]);

            // Always update orbit context
            orbit_context_.update(ch);

            bool is_delim = (
                ch == ';' || ch == ',' || ch == ':' ||
                ch == '{' || ch == '}' ||
                ch == '(' || ch == ')' ||
                ch == '[' || ch == ']'
            );

            if (is_delim) {
                Boundary boundary;
                boundary.position = pos;
                boundary.delimiter = ch;
                boundary.is_delimiter = true;

                // Populate lattice_mask using HeuristicGrid classification
                boundary.lattice_mask = stage0::classify_byte(ch);

                // Populate orbit_confidence using OrbitContext
                boundary.orbit_confidence = orbit_context_.calculateConfidence();

                boundaries.push_back(boundary);
            }

            pos++;
        }
    }

    return boundaries;
}
} // namespace cppfort::ir
