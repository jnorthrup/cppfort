#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace cppfort::stage0 {

// EvidenceSpan captures a concrete snippet of source text together with
// positional bounds, confidence score, and correctness metrics.
// These spans are the currency passed between orbits when vying for a winning interpretation.
struct EvidenceSpan {
    size_t start_pos = 0;
    size_t end_pos = 0;
    std::string content;
    double confidence = 0.0;
    
    // Correctness metrics for rule-based winnowing
    size_t match_length = 0;           // Length of the matched pattern
    size_t evidence_count = 0;         // Number of evidence segments
    double specificity_score = 0.0;    // Score based on pattern specificity
    size_t nesting_depth = 0;          // Nesting depth at match position
    bool is_speculative = false;       // Whether this is a speculative match
    bool is_valid = true;              // Whether this span passed validation
    
    // Pattern information for conflict resolution
    std::string pattern_name;
    std::string pattern_category;
    
    // Spatial relationship data for contra-mask filtering
    std::vector<size_t> contra_mask_positions;  // Positions that must NOT match certain patterns
    std::vector<std::string> required_contexts; // Context strings that must be present
    size_t min_locality_distance = 0;           // Minimum distance to related spans
    size_t max_locality_distance = SIZE_MAX;   // Maximum distance to related spans

    EvidenceSpan() = default;

    EvidenceSpan(size_t start, size_t end, std::string text, double conf = 0.0)
        : start_pos(start), end_pos(end), content(std::move(text)), confidence(conf),
          match_length(end - start) {}

    // Merge another span into this span by widening the covered range and
    // preserving the highest confidence score. Content is extended only with
    // the non-overlapping suffix to avoid quadratic growth during aggregation.
    void merge(const EvidenceSpan& other) {
        if (other.start_pos < start_pos) {
            start_pos = other.start_pos;
        }
        if (other.end_pos > end_pos) {
            const bool overlaps = other.start_pos <= end_pos;
            if (!overlaps || other.end_pos > end_pos) {
                const size_t overlap = overlaps ? (end_pos > other.start_pos ? end_pos - other.start_pos : 0) : 0;
                if (overlap < other.content.size()) {
                    content.append(other.content.substr(overlap));
                }
            }
            end_pos = other.end_pos;
        }
        confidence = std::max(confidence, other.confidence);
        match_length = end_pos - start_pos;
        evidence_count = std::max(evidence_count, other.evidence_count);
        specificity_score = std::max(specificity_score, other.specificity_score);
        nesting_depth = std::max(nesting_depth, other.nesting_depth);
        is_valid = is_valid && other.is_valid; // Both must be valid
    }
    
    // Calculate correctness score for winnowing decisions (favoring correctness over performance)
    double correctness_score() const {
        // If not valid, score is 0
        if (!is_valid) return 0.0;
        
        // Weight factors prioritizing correctness
        constexpr double VALIDITY_WEIGHT = 1.0;      // Must be valid
        constexpr double CONFIDENCE_WEIGHT = 0.8;     // High confidence
        constexpr double SPECIFICITY_WEIGHT = 0.7;    // More specific patterns
        constexpr double EVIDENCE_WEIGHT = 0.6;       // More evidence segments
        constexpr double LENGTH_WEIGHT = 0.5;         // Longer matches (but not dominant)
        constexpr double LOCALITY_WEIGHT = 0.4;       // Better spatial locality
        
        // Normalize values to 0-1 range
        double norm_confidence = std::clamp(confidence, 0.0, 1.0);
        double norm_length = (match_length > 0) ? std::min(1.0, static_cast<double>(match_length) / 1000.0) : 0.0;
        double norm_specificity = std::clamp(specificity_score, 0.0, 1.0);
        double norm_evidence = (evidence_count > 0) ? std::min(1.0, static_cast<double>(evidence_count) / 10.0) : 0.0;
        
        // Locality score (prefers moderate distances)
        double norm_locality = 1.0;
        if (min_locality_distance > 0 || max_locality_distance < SIZE_MAX) {
            // Prefer spans that are neither too close nor too far
            size_t avg_distance = (min_locality_distance + max_locality_distance) / 2;
            if (avg_distance < 10) {
                norm_locality = 0.5; // Too close
            } else if (avg_distance > 1000) {
                norm_locality = 0.3; // Too far
            } else {
                norm_locality = 1.0; // Just right
            }
        }
        
        return (VALIDITY_WEIGHT +
                norm_confidence * CONFIDENCE_WEIGHT +
                norm_specificity * SPECIFICITY_WEIGHT +
                norm_evidence * EVIDENCE_WEIGHT +
                norm_length * LENGTH_WEIGHT +
                norm_locality * LOCALITY_WEIGHT);
    }
    
    // Check if this span overlaps with another
    bool overlaps_with(const EvidenceSpan& other) const {
        return !(end_pos <= other.start_pos || start_pos >= other.end_pos);
    }
    
    // Check if this span completely contains another
    bool contains(const EvidenceSpan& other) const {
        return start_pos <= other.start_pos && end_pos >= other.end_pos;
    }
    
    // Check if this span is adjacent to another (within locality constraints)
    bool is_adjacent_to(const EvidenceSpan& other, size_t max_distance = 100) const {
        if (end_pos <= start_pos) return false;
        if (other.end_pos <= other.start_pos) return false;
        
        // Check if spans are within max_distance of each other
        size_t distance = 0;
        if (end_pos <= other.start_pos) {
            distance = other.start_pos - end_pos;
        } else if (other.end_pos <= start_pos) {
            distance = start_pos - other.end_pos;
        } else {
            return true; // Overlapping spans are considered adjacent
        }
        
        return distance <= max_distance;
    }
    
    // Validate against contra-masks (returns false if any contra-mask matches)
    bool validate_contra_masks(std::string_view source_text) const {
        // For now, assume validation passes
        // In a full implementation, this would check against negative patterns
        return true;
    }
    
    // Validate required contexts are present
    bool validate_required_contexts(std::string_view source_text) const {
        for (const auto& context : required_contexts) {
            // Simple substring search for required context
            if (source_text.find(context) == std::string_view::npos) {
                return false;
            }
        }
        return true;
    }
};

// Dominant grammar family inferred from TypeEvidence heuristics.
enum class EvidenceGrammarKind {
    Unknown,
    C,
    CPP,
    CPP2
};

struct TypeEvidence {
    // Structural punctuation counters
    std::uint16_t brace_open = 0;
    std::uint16_t brace_close = 0;
    std::uint16_t paren_open = 0;
    std::uint16_t paren_close = 0;
    std::uint16_t bracket_open = 0;
    std::uint16_t bracket_close = 0;
    std::uint16_t angle_open = 0;
    std::uint16_t angle_close = 0;
    std::uint16_t colon = 0;
    std::uint16_t double_colon = 0;
    std::uint16_t arrow = 0;
    std::uint16_t lambda_captures = 0;
    std::uint16_t pointer_indicators = 0;

    // Token-derived indicators
    std::uint16_t c_keyword_hits = 0;
    std::uint16_t cpp_keyword_hits = 0;
    std::uint16_t cpp2_keyword_hits = 0;
    std::uint16_t typedef_hits = 0;
    std::uint16_t struct_hits = 0;
    std::uint16_t template_hits = 0;
    std::uint16_t namespace_hits = 0;
    std::uint16_t concept_hits = 0;
    std::uint16_t requires_hits = 0;
    std::uint16_t inspect_hits = 0;
    std::uint16_t contract_hits = 0;
    std::uint16_t is_keyword_hits = 0;
    std::uint16_t as_keyword_hits = 0;
    std::uint16_t flow_keyword_hits = 0;
    std::uint16_t cpp2_signature_hits = 0;

    std::uint16_t total_tokens = 0;

    void observe(char ch) {
        switch (ch) {
            case '{': ++brace_open; break;
            case '}': ++brace_close; break;
            case '(': ++paren_open; break;
            case ')': ++paren_close; break;
            case '[': ++bracket_open; break;
            case ']': ++bracket_close; break;
            case '<': ++angle_open; break;
            case '>': ++angle_close; break;
            case ':': ++colon; break;
            case '*':
                if (last_char_ == ' ' || last_char_ == '\t' || last_char_ == '\n' || last_char_ == '\r' ||
                    last_char_ == '(' || last_char_ == ',' || last_char_ == '*') {
                    ++pointer_indicators;
                }
                break;
            default:
                break;
        }

        if (last_char_ == ':' && ch == ':') {
            ++double_colon;
        }
        if (last_char_ == '-' && ch == '>') {
            ++arrow;
        }
        if (last_char_ == '[' && ch == ']') {
            ++lambda_captures;
        }

        last_char_ = ch;
    }

    void ingest(std::string_view text) {
        if (text.empty()) {
            return;
        }

        for (char ch : text) {
            observe(ch);
        }

        size_t index = 0;
        while (index < text.size()) {
            if (!is_identifier_start(text[index])) {
                ++index;
                continue;
            }

            size_t begin = index;
            ++index;
            while (index < text.size() && is_identifier_continue(text[index])) {
                ++index;
            }

            process_token(text, begin, index);
        }
    }

    TypeEvidence& operator+=(char ch) {
        observe(ch);
        return *this;
    }

    void merge_max(const TypeEvidence& rhs) {
        brace_open = std::max(brace_open, rhs.brace_open);
        brace_close = std::max(brace_close, rhs.brace_close);
        paren_open = std::max(paren_open, rhs.paren_open);
        paren_close = std::max(paren_close, rhs.paren_close);
        bracket_open = std::max(bracket_open, rhs.bracket_open);
        bracket_close = std::max(bracket_close, rhs.bracket_close);
        angle_open = std::max(angle_open, rhs.angle_open);
        angle_close = std::max(angle_close, rhs.angle_close);
        colon = std::max(colon, rhs.colon);
        double_colon = std::max(double_colon, rhs.double_colon);
        arrow = std::max(arrow, rhs.arrow);
        lambda_captures = std::max(lambda_captures, rhs.lambda_captures);
        pointer_indicators = std::max(pointer_indicators, rhs.pointer_indicators);

        c_keyword_hits = std::max(c_keyword_hits, rhs.c_keyword_hits);
        cpp_keyword_hits = std::max(cpp_keyword_hits, rhs.cpp_keyword_hits);
        cpp2_keyword_hits = std::max(cpp2_keyword_hits, rhs.cpp2_keyword_hits);
        typedef_hits = std::max(typedef_hits, rhs.typedef_hits);
        struct_hits = std::max(struct_hits, rhs.struct_hits);
        template_hits = std::max(template_hits, rhs.template_hits);
        namespace_hits = std::max(namespace_hits, rhs.namespace_hits);
        concept_hits = std::max(concept_hits, rhs.concept_hits);
        requires_hits = std::max(requires_hits, rhs.requires_hits);
        inspect_hits = std::max(inspect_hits, rhs.inspect_hits);
        contract_hits = std::max(contract_hits, rhs.contract_hits);
        is_keyword_hits = std::max(is_keyword_hits, rhs.is_keyword_hits);
        as_keyword_hits = std::max(as_keyword_hits, rhs.as_keyword_hits);
        flow_keyword_hits = std::max(flow_keyword_hits, rhs.flow_keyword_hits);
        cpp2_signature_hits = std::max(cpp2_signature_hits, rhs.cpp2_signature_hits);
        total_tokens = std::max(total_tokens, rhs.total_tokens);
    }

    EvidenceGrammarKind deduce() const {
        const std::uint32_t c_score =
            static_cast<std::uint32_t>(c_keyword_hits) * 4 +
            static_cast<std::uint32_t>(typedef_hits) * 5 +
            static_cast<std::uint32_t>(struct_hits) * 4 +
            static_cast<std::uint32_t>(pointer_indicators) * 2;

        const std::uint32_t cpp_score =
            static_cast<std::uint32_t>(cpp_keyword_hits) * 3 +
            static_cast<std::uint32_t>(template_hits) * 5 +
            static_cast<std::uint32_t>(namespace_hits) * 4 +
            static_cast<std::uint32_t>(double_colon) * 5 +
            static_cast<std::uint32_t>(lambda_captures) * 3 +
            static_cast<std::uint32_t>(concept_hits) * 5 +
            static_cast<std::uint32_t>(requires_hits) * 4 +
            static_cast<std::uint32_t>(arrow) +
            static_cast<std::uint32_t>(angle_open);

        const std::uint32_t cpp2_score =
            static_cast<std::uint32_t>(cpp2_keyword_hits) * 4 +
            static_cast<std::uint32_t>(cpp2_signature_hits) * 6 +
            static_cast<std::uint32_t>(inspect_hits) * 5 +
            static_cast<std::uint32_t>(contract_hits) * 5 +
            static_cast<std::uint32_t>(is_keyword_hits + as_keyword_hits) * 3 +
            static_cast<std::uint32_t>(flow_keyword_hits) * 2 +
            static_cast<std::uint32_t>(arrow);

        EvidenceGrammarKind result = EvidenceGrammarKind::Unknown;
        std::uint32_t best = 0;
        int best_precedence = 3;

        const auto consider = [&](EvidenceGrammarKind kind, std::uint32_t score, int precedence) {
            if (score > best || (score == best && precedence < best_precedence)) {
                best = score;
                result = kind;
                best_precedence = precedence;
            }
        };

        consider(EvidenceGrammarKind::CPP2, cpp2_score, 0);
        consider(EvidenceGrammarKind::CPP, cpp_score, 1);
        consider(EvidenceGrammarKind::C, c_score, 2);

        return best == 0 ? EvidenceGrammarKind::Unknown : result;
    }



private:
    template <std::size_t N>
    static bool equals_any(std::string_view token, const std::array<std::string_view, N>& set) {
        for (auto candidate : set) {
            if (token == candidate) {
                return true;
            }
        }
        return false;
    }

    static bool is_identifier_start(char ch) {
        return std::isalpha(static_cast<unsigned char>(ch)) || ch == '_';
    }

    static bool is_identifier_continue(char ch) {
        return std::isalnum(static_cast<unsigned char>(ch)) || ch == '_';
    }

    static std::string to_lower_copy(std::string_view view) {
        std::string lowered(view);
        for (char& ch : lowered) {
            ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        }
        return lowered;
    }

    void process_token(std::string_view text, size_t begin, size_t end) {
        std::string lowered = to_lower_copy(text.substr(begin, end - begin));
        std::string_view token(lowered);
        ++total_tokens;

        static constexpr std::array<std::string_view, 6> c_tokens {
            "typedef", "struct", "enum", "union", "extern", "sizeof"
        };
        static constexpr std::array<std::string_view, 14> cpp_tokens {
            "class", "template", "typename", "namespace", "constexpr", "concept",
            "requires", "decltype", "noexcept", "operator", "using", "friend",
            "virtual", "override"
        };
        static constexpr std::array<std::string_view, 11> cpp2_tokens {
            "inspect", "let", "mut", "in", "out", "inout", "move", "forward",
            "contract", "pre", "post"
        };
        static constexpr std::array<std::string_view, 5> cpp2_colon_exclusions {
            "case", "default", "public", "protected", "private"
        };

        const bool is_c_token = equals_any(token, c_tokens);
        const bool is_cpp_token = equals_any(token, cpp_tokens);
        const bool is_cpp2_token = equals_any(token, cpp2_tokens);

        if (is_c_token) {
            ++c_keyword_hits;
            if (token == "typedef") ++typedef_hits;
            if (token == "struct") ++struct_hits;
        }
        if (is_cpp_token) {
            ++cpp_keyword_hits;
            if (token == "template") ++template_hits;
            if (token == "namespace") ++namespace_hits;
            if (token == "concept") ++concept_hits;
            if (token == "requires") ++requires_hits;
        }
        if (is_cpp2_token) {
            ++cpp2_keyword_hits;
            if (token == "inspect") ++inspect_hits;
            if (token == "contract" || token == "pre" || token == "post") {
                ++contract_hits;
            }
            if (token == "in" || token == "out" || token == "inout" || token == "move" || token == "forward") {
                ++flow_keyword_hits;
            }
        }

        if (token == "is") {
            ++is_keyword_hits;
        }
        if (token == "as") {
            ++as_keyword_hits;
        }

        if (!equals_any(token, cpp2_colon_exclusions)) {
            note_cpp2_signature(text, end);
        }
    }

    void note_cpp2_signature(std::string_view text, size_t identifier_end) {
        size_t pos = identifier_end;
        while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
            ++pos;
        }
        if (pos >= text.size() || text[pos] != ':') {
            return;
        }
        ++pos; // skip ':'
        while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
            ++pos;
        }
        if (pos < text.size() && text[pos] == '(') {
            ++cpp2_signature_hits;
        }
    }

    char last_char_ = '\0';
};

// Semantic trace for recording validation steps during orbit processing
struct SemanticTrace {
    std::string pattern_name;
    size_t anchor_index = 0;
    size_t evidence_start = 0;
    size_t evidence_end = 0;
    std::string evidence_content;
    TypeEvidence traits;
    std::string expected_type;
    bool verdict = false;  // true = valid, false = contradiction/failure
    std::string failure_reason;  // description of why validation failed
    
    // Serialize to JSON-like format for regression capture
    std::string to_json() const {
        std::string json = "{";
        json += "\"pattern\":\"" + pattern_name + "\",";
        json += "\"anchor\":" + std::to_string(anchor_index) + ",";
        json += "\"evidence_start\":" + std::to_string(evidence_start) + ",";
        json += "\"evidence_end\":" + std::to_string(evidence_end) + ",";
        json += "\"evidence\":\"" + escape_json_string(evidence_content) + "\",";
        json += "\"traits\":{" + traits_to_json() + "},";
        json += "\"expected_type\":\"" + expected_type + "\",";
        json += "\"verdict\":" + std::string(verdict ? "true" : "false");
        if (!failure_reason.empty()) {
            json += ",\"failure_reason\":\"" + escape_json_string(failure_reason) + "\"";
        }
        json += "}";
        return json;
    }
    
private:
    std::string traits_to_json() const {
        std::string json;
        json += "\"brace_open\":" + std::to_string(traits.brace_open) + ",";
        json += "\"brace_close\":" + std::to_string(traits.brace_close) + ",";
        json += "\"paren_open\":" + std::to_string(traits.paren_open) + ",";
        json += "\"paren_close\":" + std::to_string(traits.paren_close) + ",";
        json += "\"bracket_open\":" + std::to_string(traits.bracket_open) + ",";
        json += "\"bracket_close\":" + std::to_string(traits.bracket_close) + ",";
        json += "\"angle_open\":" + std::to_string(traits.angle_open) + ",";
        json += "\"angle_close\":" + std::to_string(traits.angle_close) + ",";
        json += "\"colon\":" + std::to_string(traits.colon) + ",";
        json += "\"double_colon\":" + std::to_string(traits.double_colon) + ",";
        json += "\"arrow\":" + std::to_string(traits.arrow) + ",";
        json += "\"lambda_captures\":" + std::to_string(traits.lambda_captures) + ",";
        json += "\"pointer_indicators\":" + std::to_string(traits.pointer_indicators) + ",";
        json += "\"c_keyword_hits\":" + std::to_string(traits.c_keyword_hits) + ",";
        json += "\"cpp_keyword_hits\":" + std::to_string(traits.cpp_keyword_hits) + ",";
        json += "\"cpp2_keyword_hits\":" + std::to_string(traits.cpp2_keyword_hits) + ",";
        json += "\"typedef_hits\":" + std::to_string(traits.typedef_hits) + ",";
        json += "\"struct_hits\":" + std::to_string(traits.struct_hits) + ",";
        json += "\"template_hits\":" + std::to_string(traits.template_hits) + ",";
        json += "\"namespace_hits\":" + std::to_string(traits.namespace_hits) + ",";
        json += "\"concept_hits\":" + std::to_string(traits.concept_hits) + ",";
        json += "\"requires_hits\":" + std::to_string(traits.requires_hits) + ",";
        json += "\"inspect_hits\":" + std::to_string(traits.inspect_hits) + ",";
        json += "\"contract_hits\":" + std::to_string(traits.contract_hits) + ",";
        json += "\"is_keyword_hits\":" + std::to_string(traits.is_keyword_hits) + ",";
        json += "\"as_keyword_hits\":" + std::to_string(traits.as_keyword_hits) + ",";
        json += "\"flow_keyword_hits\":" + std::to_string(traits.flow_keyword_hits) + ",";
        json += "\"cpp2_signature_hits\":" + std::to_string(traits.cpp2_signature_hits) + ",";
        json += "\"total_tokens\":" + std::to_string(traits.total_tokens);
        return json;
    }
    
    static std::string escape_json_string(std::string_view str) {
        std::string escaped;
        for (char c : str) {
            switch (c) {
                case '"': escaped += "\\\""; break;
                case '\\': escaped += "\\\\"; break;
                case '\n': escaped += "\\n"; break;
                case '\r': escaped += "\\r"; break;
                case '\t': escaped += "\\t"; break;
                default: escaped += c; break;
            }
        }
        return escaped;
    }
};

inline void update_evidence_columns(std::vector<TypeEvidence>& aggregate,
                                    const std::vector<TypeEvidence>& line) {
    if (aggregate.size() < line.size()) {
        aggregate.resize(line.size());
    }
    for (std::size_t idx = 0; idx < line.size(); ++idx) {
        aggregate[idx].merge_max(line[idx]);
    }
}

} // namespace cppfort::stage0

