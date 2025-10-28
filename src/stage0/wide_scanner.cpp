#include "wide_scanner.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <string_view>

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
#include "evidence.h"
#include "confix_orbit.h"
#include "pattern_loader.h"

namespace cppfort::ir {

void WideScanner::reset_stats() {
    stats_ = FanoutStats{};
}

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
    fragments_.clear();
    packrat_cache_.clear();
    reset_stats();

    const uint8_t* data = reinterpret_cast<const uint8_t*>(source.data());
    orbit_context_.reset();  // Reset context between scans

    using ::cppfort::stage0::EvidenceGrammarKind;
    using ::cppfort::stage0::OrbitFragment;
    using ::cppfort::stage0::TypeEvidence;

    auto emit_fragment = [&](size_t span_start, size_t span_end, uint16_t mask) {
        if (span_start >= span_end || span_end > source.size()) {
            return;
        }
        std::string_view view(source.data() + span_start, span_end - span_start);
        TypeEvidence evidence;
        evidence.ingest(view);
        EvidenceGrammarKind grammar = evidence.deduce();

        OrbitFragment fragment;
        fragment.start_pos = span_start;
        fragment.end_pos = span_end;
        fragment.lattice_mask = mask;
        fragment.confidence = (grammar == EvidenceGrammarKind::Unknown) ? 0.25 : 1.0;

        switch (grammar) {
            case EvidenceGrammarKind::C:
                fragment.classified_grammar = ::cppfort::ir::GrammarType::C;
                break;
            case EvidenceGrammarKind::CPP:
                fragment.classified_grammar = ::cppfort::ir::GrammarType::CPP;
                break;
            case EvidenceGrammarKind::CPP2:
                fragment.classified_grammar = ::cppfort::ir::GrammarType::CPP2;
                break;
            case EvidenceGrammarKind::Unknown:
            default:
                fragment.classified_grammar = ::cppfort::ir::GrammarType::UNKNOWN;
                break;
        }

        fragments_.push_back(std::move(fragment));
    };

    // Helper: detect if position starts CPP2 function pattern (name followed by colon+paren)
    auto is_cpp2_function_start = [&](size_t pos) -> bool {
        if (pos >= source.size()) return false;
        // Scan forward to find colon
        size_t colon_pos = pos;
        while (colon_pos < source.size() && source[colon_pos] != ':' && source[colon_pos] != '\n') {
            colon_pos++;
        }
        if (colon_pos >= source.size() || source[colon_pos] != ':') {
            return false;
        }
        // Check if colon is followed by whitespace then '('
        size_t paren_pos = colon_pos + 1;
        while (paren_pos < source.size() && (source[paren_pos] == ' ' || source[paren_pos] == '\t')) {
            paren_pos++;
        }
        return (paren_pos < source.size() && source[paren_pos] == '(');
    };

    // Helper: find matching closing delimiter accounting for nesting
    auto find_matching_close = [&](size_t start_pos, char open_char, char close_char) -> size_t {
        if (start_pos >= source.size()) return std::string::npos;
        int depth = 1;
        size_t pos = start_pos + 1;
        while (pos < source.size() && depth > 0) {
            if (source[pos] == open_char) {
                depth++;
            } else if (source[pos] == close_char) {
                depth--;
            }
            if (depth == 0) return pos;
            pos++;
        }
        return std::string::npos;
    };

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
        // DON'T emit a single fragment for the whole file - fall through to use CPP2 function detection below
        std::cerr << "DEBUG scanAnchorsWithOrbits: Small file with " << boundaries.size() << " boundaries, falling through to function detection\n";
        // DO NOT return here - let the code below handle fragmentation
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
                // Scan ALL bytes in this chunk to find ALL delimiters
                for (size_t i = 0; i < 16 && (pos + i) < end_pos; ++i) {
                    char ch = static_cast<char>(data[pos + i]);

                    // Update orbit context for each byte
                    orbit_context_.update(ch);

                    bool is_delim = (ch == ';' || ch == ',' || ch == ':' || ch == '{' || ch == '}' ||
                                     ch == '(' || ch == ')' || ch == '[' || ch == ']');
                    if (is_delim) {
                        Boundary boundary;
                        boundary.position = pos + i;
                        boundary.delimiter = ch;
                        boundary.is_delimiter = true;

                        // Populate lattice_mask using HeuristicGrid classification
                        boundary.lattice_mask = stage0::classify_byte(ch);

                        // Populate orbit_confidence using OrbitContext
                        boundary.orbit_confidence = orbit_context_.calculateConfidence();

                        packrat_cache_.store_cache(boundary.position, ::cppfort::stage0::OrbitType::Confix, boundary.orbit_confidence);
                        boundaries.push_back(boundary);
                    }
                }
            } else {
                // No delimiters found, but still update orbit context for all bytes
                for (size_t i = 0; i < 16 && (pos + i) < end_pos; ++i) {
                    orbit_context_.update(static_cast<char>(data[pos + i]));
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
                // Scan ALL bytes in this chunk to find ALL delimiters
                for (size_t i = 0; i < 16 && (pos + i) < end_pos; ++i) {
                    char ch = static_cast<char>(data[pos + i]);

                    // Update orbit context for each byte
                    orbit_context_.update(ch);

                    bool is_delim = (
                        ch == ';' || ch == ',' || ch == ':' || ch == '{' || ch == '}' ||
                        ch == '(' || ch == ')' || ch == '[' || ch == ']'
                    );

                    if (is_delim) {
                        Boundary boundary;
                        boundary.position = pos + i;
                        boundary.delimiter = ch;
                        boundary.is_delimiter = true;

                        // Populate lattice_mask using HeuristicGrid classification
                        boundary.lattice_mask = stage0::classify_byte(ch);

                        // Populate orbit_confidence using OrbitContext
                        boundary.orbit_confidence = orbit_context_.calculateConfidence();

                        packrat_cache_.store_cache(boundary.position, ::cppfort::stage0::OrbitType::Confix, boundary.orbit_confidence);
                        boundaries.push_back(boundary);
                    }
                }
            } else {
                // No delimiters found, but still update orbit context for all bytes
                for (size_t i = 0; i < 16 && (pos + i) < end_pos; ++i) {
                    orbit_context_.update(static_cast<char>(data[pos + i]));
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

    std::cerr << "DEBUG scanAnchorsWithOrbits: Starting fragment emission with " << boundaries.size() << " boundaries\n";
    size_t fragment_start = 0;
    size_t i = 0;
    size_t function_count = 0;
    while (i < boundaries.size()) {
        const auto& boundary = boundaries[i];

        // Check if this boundary is a colon that starts a CPP2 function
        if (boundary.delimiter == ':' && boundary.is_delimiter) {
            // Look back to start of line to get function name
            size_t line_start = boundary.position;
            while (line_start > 0 && source[line_start - 1] != '\n') {
                line_start--;
            }
            // Skip leading whitespace on line
            while (line_start < source.size() && (source[line_start] == ' ' || source[line_start] == '\t')) {
                line_start++;
            }

            // Check if this line contains a CPP2 function pattern
            if (line_start < boundary.position && is_cpp2_function_start(line_start)) {
                // Find the function body end (matching closing brace)
                size_t brace_pos = boundary.position;
                while (brace_pos < source.size() && source[brace_pos] != '{' && source[brace_pos] != ';') {
                    brace_pos++;
                }

                if (brace_pos < source.size()) {
                    size_t func_end;
                    if (source[brace_pos] == '{') {
                        // Function with body - find matching close brace
                        size_t close_brace = find_matching_close(brace_pos, '{', '}');
                        func_end = (close_brace != std::string::npos) ? close_brace + 1 : brace_pos + 1;
                    } else {
                        // Forward declaration or inline - end at semicolon
                        func_end = brace_pos + 1;
                    }

                    // Emit entire function as single fragment
                    if (func_end > line_start) {
                        function_count++;
                        std::cerr << "DEBUG scanAnchorsWithOrbits: Emitting function #" << function_count
                                  << " fragment [" << line_start << ", " << func_end << ")\n";
                        std::cerr << "DEBUG scanAnchorsWithOrbits: Fragment text: '"
                                  << source.substr(line_start, std::min(size_t(50), func_end - line_start)) << "...'\n";
                        emit_fragment(line_start, func_end, boundary.lattice_mask);
                        fragment_start = func_end;

                        // Skip boundaries within this function
                        size_t skipped = 0;
                        while (i < boundaries.size() && boundaries[i].position < func_end) {
                            skipped++;
                            i++;
                        }
                        std::cerr << "DEBUG scanAnchorsWithOrbits: Skipped " << skipped << " boundaries, now at i=" << i << "\n";
                        continue;
                    }
                }
            }
        }

        // Standard fragment boundary handling
        if (boundary.position > fragment_start) {
            emit_fragment(fragment_start, boundary.position, boundary.lattice_mask);
        }
        if (boundary.is_delimiter && boundary.position < source.size()) {
            fragment_start = boundary.position + 1;
        } else {
            fragment_start = boundary.position;
        }
        if (fragment_start > source.size()) {
            fragment_start = source.size();
        }
        i++;
    }
    std::cerr << "DEBUG scanAnchorsWithOrbits: Total functions emitted: " << function_count << "\n";
    std::cerr << "DEBUG scanAnchorsWithOrbits: Total fragments: " << fragments_.size() << "\n";
    if (fragment_start < source.size()) {
        emit_fragment(fragment_start, source.size(), 0);
    }

    return boundaries;
}
} // namespace cppfort::ir
