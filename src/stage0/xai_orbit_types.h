#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <variant>
#include <string>
#include <algorithm>

namespace cppfort::stage0 {

/**
 * XAI 4.2 Orbit System - 5 Anchor Types for Evidence-Based Parsing
 * Extracted from build/x.txt
 */

enum class AnchorType : uint8_t {
    COUNT_DELIMITERS = 0,   // Array bounds, loop constructs, repetition patterns
    INDENTATION = 1,        // Scope-based evidence from whitespace patterns
    NUMBER_DUCK_TYPE = 2,   // Numeric literal classification and type inference
    LEGAL_CLASSES = 3,      // Valid C++2/CPP2 type constructs and metaclass patterns
    CASCADING_RANGES = 4    // Hierarchical evidence propagation through spans
};

/**
 * XAI variant type for anchor values
 * Can hold integer counts, string spans, or confidence scores
 */
using xai_variant = std::variant<
    int,                          // Count delimiters
    std::span<const char>,        // Text spans
    double                        // Numeric confidence
>;

/**
 * Evidence Anchor with xAI 4.2 Variant Support
 */
struct EvidenceAnchor {
    AnchorType anchor_type;
    xai_variant value;
    double confidence;
    std::span<const char> span_range;
    size_t position;

    EvidenceAnchor()
        : anchor_type(AnchorType::COUNT_DELIMITERS)
        , value(0)
        , confidence(0.0)
        , span_range()
        , position(0)
    {}

    EvidenceAnchor(AnchorType type, xai_variant val, double conf,
                   std::span<const char> span, size_t pos)
        : anchor_type(type)
        , value(val)
        , confidence(conf)
        , span_range(span)
        , position(pos)
    {}

    // Evidence composition operators
    EvidenceAnchor operator+(const EvidenceAnchor& other) const {
        // Merge spans and average confidence
        double merged_conf = (confidence + other.confidence) / 2.0;
        return EvidenceAnchor(anchor_type, value, merged_conf, span_range, position);
    }

    EvidenceAnchor operator|(const EvidenceAnchor& other) const {
        // Alternative: take higher confidence anchor
        return (confidence > other.confidence) ? *this : other;
    }

    EvidenceAnchor operator&(const EvidenceAnchor& other) const {
        // Intersection: require both anchors to be valid (min confidence)
        double merged_conf = std::min(confidence, other.confidence);
        return EvidenceAnchor(anchor_type, value, merged_conf, span_range, position);
    }
};

/**
 * Tuple-based Anchor Interleaving for Evidence Ranges
 * All 5 anchor types fire concurrently on same span
 */
struct AnchorTuple {
    std::array<EvidenceAnchor, 5> anchors;  // One per anchor type
    std::span<const char> evidence_range;
    double composite_confidence;

    AnchorTuple()
        : anchors{}
        , evidence_range()
        , composite_confidence(0.0)
    {}

    explicit AnchorTuple(std::span<const char> range)
        : evidence_range(range)
        , composite_confidence(0.0)
    {
        // Initialize with empty anchors
        for (size_t i = 0; i < 5; ++i) {
            anchors[i] = EvidenceAnchor(
                static_cast<AnchorType>(i),
                xai_variant{},
                0.0,
                range,
                0
            );
        }
    }

    /**
     * Interleave evidence from all 5 anchor types
     * Computes composite confidence from anchor contributions
     */
    void interleave_evidence() {
        // Count delimiters contribution
        int delim_count = count_delimiters(evidence_range);
        anchors[0] = EvidenceAnchor(
            AnchorType::COUNT_DELIMITERS,
            delim_count,
            delim_count > 0 ? 0.7 : 0.0,
            evidence_range,
            0
        );

        // Indentation contribution
        int indent_depth = detect_indentation(evidence_range);
        anchors[1] = EvidenceAnchor(
            AnchorType::INDENTATION,
            indent_depth,
            indent_depth > 0 ? 0.6 : 0.0,
            evidence_range,
            0
        );

        // Number duck typing
        bool has_numbers = detect_numeric_patterns(evidence_range);
        anchors[2] = EvidenceAnchor(
            AnchorType::NUMBER_DUCK_TYPE,
            has_numbers ? 1 : 0,
            has_numbers ? 0.8 : 0.0,
            evidence_range,
            0
        );

        // Legal classes (CPP2/C++ syntax)
        bool has_legal_syntax = detect_legal_classes(evidence_range);
        anchors[3] = EvidenceAnchor(
            AnchorType::LEGAL_CLASSES,
            has_legal_syntax ? 1 : 0,
            has_legal_syntax ? 0.9 : 0.0,
            evidence_range,
            0
        );

        // Cascading ranges (hierarchical structure)
        bool has_cascade = detect_cascading_ranges(evidence_range);
        anchors[4] = EvidenceAnchor(
            AnchorType::CASCADING_RANGES,
            has_cascade ? 1 : 0,
            has_cascade ? 0.75 : 0.0,
            evidence_range,
            0
        );

        // Compute composite confidence (weighted average)
        composite_confidence = 0.0;
        for (const auto& anchor : anchors) {
            composite_confidence += anchor.confidence;
        }
        composite_confidence /= 5.0;
    }

private:
    // Helper methods for anchor detection
    static int count_delimiters(std::span<const char> span) {
        int count = 0;
        for (char c : span) {
            if (c == '(' || c == ')' || c == '{' || c == '}' ||
                c == '[' || c == ']' || c == ';' || c == ',') {
                ++count;
            }
        }
        return count;
    }

    static int detect_indentation(std::span<const char> span) {
        int depth = 0;
        bool at_line_start = true;
        for (char c : span) {
            if (at_line_start && (c == ' ' || c == '\t')) {
                ++depth;
            } else if (c == '\n') {
                at_line_start = true;
            } else if (!std::isspace(static_cast<unsigned char>(c))) {
                at_line_start = false;
            }
        }
        return depth;
    }

    static bool detect_numeric_patterns(std::span<const char> span) {
        for (char c : span) {
            if (std::isdigit(static_cast<unsigned char>(c))) {
                return true;
            }
        }
        return false;
    }

    static bool detect_legal_classes(std::span<const char> span) {
        // Simple detection: look for ':' (CPP2 type annotations)
        std::string_view sv(span.data(), span.size());
        return sv.find(':') != std::string_view::npos;
    }

    static bool detect_cascading_ranges(std::span<const char> span) {
        // Simple detection: nested brackets
        int brace_depth = 0;
        for (char c : span) {
            if (c == '{') ++brace_depth;
            else if (c == '}') --brace_depth;
        }
        return brace_depth != 0;  // Has unclosed braces = cascading
    }
};

} // namespace cppfort::stage0
