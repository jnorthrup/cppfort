#pragma once

#include "evidence.h"
#include "confix_orbit.h"
#include <string_view>
#include <vector>

namespace cppfort::stage0 {

// OrbitCombinator: Handles language-specific patterns and positional reasoning
// Operates on orbits and evidence, NOT in evidence.h (separation of concerns)
struct OrbitCombinator {
    // Check if orbit has balanced delimiters
    static bool check_balance(const TypeEvidence& evidence) {
        return evidence.brace_open == evidence.brace_close &&
               evidence.paren_open == evidence.paren_close &&
               evidence.bracket_open == evidence.bracket_close &&
               evidence.angle_open == evidence.angle_close;
    }

    // Check if this looks like a function signature (CPP2)
    static bool is_cpp2_signature(const TypeEvidence& evidence) {
        return evidence.cpp2_signature_hits > 0 ||
               (evidence.paren_open > 0 && evidence.arrow > 0);
    }

    // Check if this is a CPP2 keyword
    static bool is_cpp2_keyword(const TypeEvidence& evidence) {
        return evidence.cpp2_keyword_hits > 0;
    }

    // Get nesting depth from evidence
    static size_t nesting_depth(const TypeEvidence& evidence) {
        return evidence.brace_open + evidence.paren_open +
               evidence.bracket_open + evidence.angle_open;
    }

    // Check if evidence suggests a function declaration
    static bool is_function(const TypeEvidence& evidence) {
        return evidence.function_hits > 0 ||
               evidence.paren_open > 0;
    }

    // Calculate confidence for CPP2 grammar
    static double cpp2_confidence(const TypeEvidence& evidence) {
        if (evidence.total_tokens == 0) return 0.0;

        double score = 0.0;
        score += evidence.cpp2_keyword_hits * 4.0;
        score += evidence.cpp2_signature_hits * 6.0;
        score += evidence.inspect_hits * 5.0;
        score += evidence.contract_hits * 5.0;
        score += (evidence.is_keyword_hits + evidence.as_keyword_hits) * 3.0;
        score += evidence.flow_keyword_hits * 2.0;
        score += evidence.arrow * 1.0;

        return std::min(1.0, score / static_cast<double>(evidence.total_tokens));
    }

    // Calculate confidence for CPP grammar
    static double cpp_confidence(const TypeEvidence& evidence) {
        if (evidence.total_tokens == 0) return 0.0;

        double score = 0.0;
        score += evidence.cpp_keyword_hits * 3.0;
        score += evidence.template_hits * 5.0;
        score += evidence.namespace_hits * 4.0;
        score += evidence.double_colon * 5.0;
        score += evidence.lambda_captures * 3.0;
        score += evidence.concept_hits * 5.0;
        score += evidence.requires_hits * 4.0;
        score += evidence.arrow * 1.0;
        score += evidence.angle_open * 1.0;

        return std::min(1.0, score / static_cast<double>(evidence.total_tokens));
    }

    // Calculate confidence for C grammar
    static double c_confidence(const TypeEvidence& evidence) {
        if (evidence.total_tokens == 0) return 0.0;

        double score = 0.0;
        score += evidence.c_keyword_hits * 4.0;
        score += evidence.typedef_hits * 5.0;
        score += evidence.struct_hits * 4.0;
        score += evidence.pointer_indicators * 2.0;

        return std::min(1.0, score / static_cast<double>(evidence.total_tokens));
    }

    // Deduce grammar family from evidence
    enum class GrammarFamily { UNKNOWN, C, CPP, CPP2 };

    static GrammarFamily deduce_grammar(const TypeEvidence& evidence) {
        double c = c_confidence(evidence);
        double cpp = cpp_confidence(evidence);
        double cpp2 = cpp2_confidence(evidence);

        if (cpp2 > cpp && cpp2 > c) return GrammarFamily::CPP2;
        if (cpp > c) return GrammarFamily::CPP;
        if (c > 0.0) return GrammarFamily::C;
        return GrammarFamily::UNKNOWN;
    }

    // Extended delimiter tracking (needs to be in combinator, not evidence)
    struct DelimiterEvidence {
        uint16_t brace_open = 0;
        uint16_t brace_close = 0;
        uint16_t paren_open = 0;
        uint16_t paren_close = 0;
        uint16_t bracket_open = 0;
        uint16_t bracket_close = 0;
        uint16_t angle_open = 0;
        uint16_t angle_close = 0;
        uint16_t colon = 0;
        uint16_t double_colon = 0;
        uint16_t arrow = 0;
        uint16_t semicolon = 0;
        uint16_t comma = 0;
        uint16_t lambda_captures = 0;
        uint16_t pointer_indicators = 0;
        uint16_t reference_indicators = 0;

        // Language keyword counters
        uint16_t c_keyword_hits = 0;
        uint16_t cpp_keyword_hits = 0;
        uint16_t cpp2_keyword_hits = 0;
        uint16_t typedef_hits = 0;
        uint16_t struct_hits = 0;
        uint16_t enum_hits = 0;
        uint16_t union_hits = 0;
        uint16_t template_hits = 0;
        uint16_t namespace_hits = 0;
        uint16_t concept_hits = 0;
        uint16_t requires_hits = 0;
        uint16_t inspect_hits = 0;
        uint16_t contract_hits = 0;
        uint16_t is_keyword_hits = 0;
        uint16_t as_keyword_hits = 0;
        uint16_t flow_keyword_hits = 0;
        uint16_t cpp2_signature_hits = 0;
        uint16_t function_hits = 0;

        // Token metrics
        uint16_t total_tokens = 0;
        uint16_t identifier_hits = 0;
        uint16_t number_literals = 0;
        uint16_t string_literals = 0;
        uint16_t char_literals = 0;

        // Merge from TypeEvidence
        void merge_from(const TypeEvidence& te) {
            // Copy extended counters from TypeEvidence
            // This allows TypeEvidence to stay pure while combinator handles language features
        }
    };
};

} // namespace cppfort::stage0
