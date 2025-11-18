#pragma once

#include <cstdint>
#include <array>
#include <algorithm>
#include <string>
#include <string_view>
#include <vector>
#include <functional>

namespace cppfort::stage0 {
// Dominant grammar family inferred from TypeEvidence heuristics.
enum class EvidenceGrammarKind {
    Unknown,
    C,
    CPP,
    CPP2
};


// ConfixType: Different types of delimiter contexts for tracking balance
// Based on MLIR's block/region structure and delimiter types
enum class ConfixType : uint8_t {
    INVALID = 0,

    // Code structure delimiters
    PAREN = 1,           // () - function calls, expressions
    BRACE = 2,           // {} - code blocks, scopes
    BRACKET = 3,         // [] - arrays, attributes
    ANGLE = 4,           // <> - templates, generics

    // String/char literals
    STRING_DOUBLE = 5,   // "..." - double-quoted strings
    STRING_SINGLE = 6,   // '...' - single-quoted strings/chars
    STRING_RAW = 7,      // R"(...)", raw strings

    // Comments
    COMMENT_LINE = 8,    // // ... line comments
    COMMENT_BLOCK = 9,   // /* ... */ block comments
    COMMENT_DOC = 10,    // ///, /** ... */ documentation

    // Preprocessor/annotations
    PREPROCESSOR = 11,   // #... preprocessor directives

    MAX_TYPE = 11
};

// TypeEvidence: Hierarchical evidence for character and context classification
// Tracks counters across multiple dimensions for fast pattern matching
struct TypeEvidence {
    // --- Layer 1: Character classification (trikeshed-style) ---
    // Pure character-based counters, no semantics
    uint16_t digits = 0;           // '0'-'9'
    uint16_t periods = 0;          // '.'
    uint16_t exponent = 0;         // 'e', 'E'
    uint16_t signs = 0;            // '+', '-'
    uint16_t special = 0;          // Other special chars
    uint16_t alpha = 0;            // Letters (a-z, A-Z)
    uint16_t truefalse = 0;        // Letters in "true" and "false"
    uint16_t empty = 0;            // Empty column indicator
    uint16_t quotes = 0;           // Single quotes
    uint16_t dquotes = 0;          // Double quotes
    uint16_t whitespaces = 0;      // ' ', '\t'
    uint16_t backslashes = 0;      // '\\'
    uint16_t linefeed = 0;         // '\n' counter
    uint16_t columnLength = 0;     // Total column length

    // --- Layer 2: Confix open/close tracking ---
    // Track each delimiter type separately for balance detection
    // Index by static_cast<uint8_t>(ConfixType)
    std::array<uint16_t, 12> confix_open = {0};   // Opening delimiters per type
    std::array<uint16_t, 12> confix_close = {0};  // Closing delimiters per type

    // Track nested confix sequences for pattern detection
    // e.g., "(()" has depth 2 for PAREN at position
    std::array<uint16_t, 12> max_confix_depth = {0};  // Max nesting depth per type
    std::array<uint16_t, 12> min_confix_depth = {0};  // Min nesting depth per type

    // --- Layer 3: Identifier classification ---
    uint16_t c_identifiers = 0;     // C-style identifiers
    uint16_t cpp_identifiers = 0;   // C++-style identifiers with ::
    uint16_t template_ids = 0;      // Template instantiations with <>

    // --- Layer 4: Language keywords ---
    // Separate counters for each language family for grammar deduction
    uint16_t c_keywords = 0;        // C keywords (typedef, struct, enum, union)
    uint16_t cpp_keywords = 0;      // C++ keywords (class, template, namespace)
    uint16_t cpp2_keywords = 0;     // Cpp2 keywords (inspect, in, out, move)
    uint16_t preprocessor = 0;      // Preprocessor directives (#include, #define)

    // --- Layer 5: Expression structure ---
    uint16_t comma = 0;             // Comma separators
    uint16_t semicolon = 0;         // Statement terminators
    uint16_t arrow = 0;             // -> operator
    uint16_t double_colon = 0;      // :: scope resolution
    uint16_t colon = 0;             // Single colon (labels, inheritance)

    // --- Layer 6: Literal tracking ---
    uint16_t number_literals = 0;   // Numeric literals
    uint16_t string_literals = 0;   // String literal markers
    uint16_t char_literals = 0;     // Character literals
    uint16_t raw_strings = 0;       // Raw string literals (R"(...)")

    // --- Layer 7: Flow control ---
    uint16_t flow_keywords = 0;     // Control flow keywords
    uint16_t contract_keywords = 0; // Contract keywords (pre, post)

    // --- Derived metrics (computed on-demand) ---
    uint16_t total_tokens = 0;      // Total tokens observed
    uint16_t nesting_complexity = 0; // Computed from confix depth variance
    
    // --- C-specific counters for correlator compatibility ---
    uint16_t typedef_hits = 0;      // typedef keyword count
    uint16_t struct_hits = 0;       // struct keyword count
    uint16_t pointer_indicators = 0; // Pointer/reference indicators
    
    // --- Additional counters for correlator compatibility ---
    uint16_t namespace_hits = 0;    // namespace keyword count
    uint16_t lambda_captures = 0;   // Lambda capture indicators
    uint16_t concept_hits = 0;      // concept keyword count
    uint16_t requires_hits = 0;     // requires keyword count
    uint16_t angle_open = 0;        // Opening angle brackets <
    uint16_t cpp2_signature_hits = 0; // Cpp2 signature patterns (: ())
    uint16_t inspect_hits = 0;      // inspect keyword count
    uint16_t is_keyword_hits = 0;   // is keyword count
    uint16_t as_keyword_hits = 0;   // as keyword count

    // Reset all counters
    void reset() {
        *this = TypeEvidence();
    }
    
    // Deduce dominant grammar family from counters
    EvidenceGrammarKind deduce() const;

    // --- Member function declarations ---
    void observe_char(char ch);
    void observe_confix_open(ConfixType type);
    void observe_confix_close(ConfixType type);
    void observe_token(std::string_view token);
    void ingest(std::string_view text);
    void merge_max(const TypeEvidence& other);

    // --- Balance checking ---
    bool all_confix_balanced() const;
    bool confix_balanced(ConfixType type) const;
    int16_t confix_delta(ConfixType type) const;
    uint16_t get_max_depth(ConfixType type) const;
    bool encloses(ConfixType outer, ConfixType inner) const;
    bool has_nested_pattern(ConfixType type) const;
    std::vector<ConfixType> get_mixed_patterns() const;
    uint32_t compute_nesting_complexity() const;

    // --- Confidence scores for language families ---
    double cpp2_confidence() const;
    double cpp_confidence() const;
    double c_confidence() const;

    // Deduce dominant language family
    enum class LanguageFamily { UNKNOWN, C, CPP, CPP2 };
    LanguageFamily deduce_language() const;

// NOTE: Helper definitions follow - these are out-of-class definitions for the
// private helper functions declared above in the struct.
    // Token classification helpers (static - no access to instance)
private:
    static bool is_c_identifier(std::string_view token);
    static bool is_c_keyword(std::string_view token);
    static inline bool is_cpp_keyword(std::string_view token);
    static inline bool is_cpp2_keyword(std::string_view token);
};

inline EvidenceGrammarKind TypeEvidence::deduce() const {
    using uint = std::uint32_t;
    uint c_score = static_cast<uint>(c_keywords) * 4u + static_cast<uint>(typedef_hits) * 5u + static_cast<uint>(struct_hits) * 4u + static_cast<uint>(pointer_indicators) * 2u;
    uint cpp_score = static_cast<uint>(cpp_keywords) * 3u + static_cast<uint>(template_ids) * 5u + static_cast<uint>(namespace_hits) * 4u + static_cast<uint>(double_colon) * 5u + static_cast<uint>(lambda_captures) * 3u + static_cast<uint>(concept_hits) * 5u + static_cast<uint>(requires_hits) * 4u + static_cast<uint>(arrow) + static_cast<uint>(angle_open);
    uint cpp2_score = static_cast<uint>(cpp2_keywords) * 4u + static_cast<uint>(cpp2_signature_hits) * 6u + static_cast<uint>(inspect_hits) * 5u + static_cast<uint>(contract_keywords) * 5u + static_cast<uint>(is_keyword_hits + as_keyword_hits) * 3u + static_cast<uint>(flow_keywords) * 2u + static_cast<uint>(arrow);

    if (c_score >= cpp_score && c_score >= cpp2_score) return EvidenceGrammarKind::C;
    if (cpp_score >= c_score && cpp_score >= cpp2_score) return EvidenceGrammarKind::CPP;
    if (cpp2_score >= c_score && cpp2_score >= cpp_score) return EvidenceGrammarKind::CPP2;
    return EvidenceGrammarKind::Unknown;
}

    // Observe a character and update character classification
    // Must be called for every character in source
    inline void TypeEvidence::observe_char(char ch) {
        // Layer 1: Character classification
        if (ch >= '0' && ch <= '9') {
            ++digits; ++alpha;
        } else if (ch == '.') {
            ++periods; ++special;
        } else if (ch == 'e' || ch == 'E') {
            ++exponent; ++alpha;
        } else if (ch == '+' || ch == '-') {
            ++signs; ++special;
        } else if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z')) {
            ++alpha;
            if (ch == 't' || ch == 'T' || ch == 'r' || ch == 'R' ||
                ch == 'u' || ch == 'U' || ch == 'f' || ch == 'F' ||
                ch == 'a' || ch == 'A' || ch == 'l' || ch == 'L' ||
                ch == 's' || ch == 'S') {
                ++truefalse;
            }
        } else if (ch == '"') {
            ++dquotes; ++special;
        } else if (ch == '\'') {
            ++quotes; ++special;
        } else if (ch == '\\') {
            ++backslashes; ++special;
        } else if (ch == ' ' || ch == '\t') {
            ++whitespaces;
        } else if (ch == '\n') {
            ++linefeed;
        } else if (ch == ',') {
            ++comma; ++special;
        } else if (ch == ';') {
            ++semicolon; ++special;
        } else if (ch == ':') {
            ++colon; ++special;
        } else if (ch == '&') {
            ++special;
        } else if (ch == '*') {
            ++special;
        } else {
            ++special;
        }

        ++columnLength;
    }

    // Observe opening delimiter of specified type
    // Called when encountering (, [, {, <, ", ', /*, //, etc.
    inline void TypeEvidence::observe_confix_open(ConfixType type) {
        uint8_t idx = static_cast<uint8_t>(type);
        if (idx < 12) {
            ++confix_open[idx];
            // Update max depth
            uint16_t depth = confix_open[idx] - confix_close[idx];
            if (depth > max_confix_depth[idx]) {
                max_confix_depth[idx] = depth;
            }
        }
    }

    // Observe closing delimiter of specified type
    // Called when encountering ), ], }, >, ", ', */, etc.
    inline void TypeEvidence::observe_confix_close(ConfixType type) {
        uint8_t idx = static_cast<uint8_t>(type);
        if (idx < 12) {
            ++confix_close[idx];
            // Update current depth
            uint16_t depth = confix_open[idx] - confix_close[idx];
            if (depth < min_confix_depth[idx]) {
                min_confix_depth[idx] = depth;
            }
        }
    }

    // Observe a complete token (identifier, keyword, etc.)
    inline void TypeEvidence::observe_token(std::string_view token) {
        ++total_tokens;

        // Check C identifiers
        if (is_c_identifier(token)) {
            ++c_identifiers;
        }

        // Check C++ identifiers with ::
        if (token.find("::") != std::string::npos) {
            ++cpp_identifiers;
            ++double_colon;
        }

        // Check for template instantiations
        if (token.find('<') != std::string::npos && token.find('>') != std::string::npos) {
            ++template_ids;
        }

        // Check keywords
        if (is_c_keyword(token)) ++c_keywords;
        if (is_cpp_keyword(token)) ++cpp_keywords;
        if (is_cpp2_keyword(token)) ++cpp2_keywords;

        // Track some individual cpp2-relevant keywords for scoring
        if (token == "inspect") ++inspect_hits;
        if (token == "is") ++is_keyword_hits;
        if (token == "as") ++as_keyword_hits;

        // Count special operators
        for (size_t i = 0; i < token.length() - 1; ++i) {
            if (token[i] == '-' && token[i+1] == '>') ++arrow;
        }
    }

    // Ingest complete text and update all counters
    // Used for correlator compatibility
    inline void TypeEvidence::ingest(std::string_view text) {
        reset();

        std::string token;
        const char* s = text.data();
        size_t n = text.size();
        for (size_t i = 0; i < n; ++i) {
            char ch = s[i];
            observe_char(ch);

            // Simple token extraction
            if (std::isspace(static_cast<unsigned char>(ch)) || ch == '(' || ch == ')' ||
                ch == '{' || ch == '}' || ch == '[' || ch == ']' || ch == ';' || ch == ',' ||
                ch == ':' || ch == '<' || ch == '>') {
                if (!token.empty()) {
                    observe_token(token);
                    token.clear();
                }
            } else {
                  token += ch;
            }

            // Heuristic: detect Cpp2 signature patterns like ": (" or ":(" at char level
            if (ch == ':') {
                // skip spaces and check for '('
                size_t j = i + 1;
                while (j < n && std::isspace(static_cast<unsigned char>(s[j]))) ++j;
                if (j < n && s[j] == '(') {
                    ++cpp2_signature_hits;
                }
            }
        }
        if (!token.empty()) {
            observe_token(token);
        }
    }

    // Merge with another TypeEvidence (take max of each field)
    // Used for sliding window accumulation
    inline void TypeEvidence::merge_max(const TypeEvidence& other) {
        // Layer 1
        digits = std::max(digits, other.digits);
        periods = std::max(periods, other.periods);
        exponent = std::max(exponent, other.exponent);
        signs = std::max(signs, other.signs);
        special = std::max(special, other.special);
        alpha = std::max(alpha, other.alpha);
        truefalse = std::max(truefalse, other.truefalse);
        empty = std::max(empty, other.empty);
        quotes = std::max(quotes, other.quotes);
        dquotes = std::max(dquotes, other.dquotes);
        whitespaces = std::max(whitespaces, other.whitespaces);
        backslashes = std::max(backslashes, other.backslashes);
        linefeed = std::max(linefeed, other.linefeed);
        columnLength = std::max(columnLength, other.columnLength);

        // Layer 2: Confix tracking
        for (size_t i = 0; i < 12; ++i) {
            confix_open[i] = std::max(confix_open[i], other.confix_open[i]);
            confix_close[i] = std::max(confix_close[i], other.confix_close[i]);
            max_confix_depth[i] = std::max(max_confix_depth[i], other.max_confix_depth[i]);
            min_confix_depth[i] = std::min(min_confix_depth[i], other.min_confix_depth[i]);
        }

        // Layer 3-7
        c_identifiers = std::max(c_identifiers, other.c_identifiers);
        cpp_identifiers = std::max(cpp_identifiers, other.cpp_identifiers);
        template_ids = std::max(template_ids, other.template_ids);

        c_keywords = std::max(c_keywords, other.c_keywords);
        cpp_keywords = std::max(cpp_keywords, other.cpp_keywords);
        cpp2_keywords = std::max(cpp2_keywords, other.cpp2_keywords);
        preprocessor = std::max(preprocessor, other.preprocessor);

        comma = std::max(comma, other.comma);
        semicolon = std::max(semicolon, other.semicolon);
        arrow = std::max(arrow, other.arrow);
        double_colon = std::max(double_colon, other.double_colon);
        colon = std::max(colon, other.colon);

        number_literals = std::max(number_literals, other.number_literals);
        string_literals = std::max(string_literals, other.string_literals);
        char_literals = std::max(char_literals, other.char_literals);
        raw_strings = std::max(raw_strings, other.raw_strings);

        flow_keywords = std::max(flow_keywords, other.flow_keywords);
        contract_keywords = std::max(contract_keywords, other.contract_keywords);

        total_tokens = std::max(total_tokens, other.total_tokens);
        nesting_complexity = std::max(nesting_complexity, other.nesting_complexity);
        
        // C-specific counters
        typedef_hits = std::max(typedef_hits, other.typedef_hits);
        struct_hits = std::max(struct_hits, other.struct_hits);
        pointer_indicators = std::max(pointer_indicators, other.pointer_indicators);
        
        // Additional counters
        namespace_hits = std::max(namespace_hits, other.namespace_hits);
        lambda_captures = std::max(lambda_captures, other.lambda_captures);
        concept_hits = std::max(concept_hits, other.concept_hits);
        requires_hits = std::max(requires_hits, other.requires_hits);
        angle_open = std::max(angle_open, other.angle_open);
        cpp2_signature_hits = std::max(cpp2_signature_hits, other.cpp2_signature_hits);
        inspect_hits = std::max(inspect_hits, other.inspect_hits);
        is_keyword_hits = std::max(is_keyword_hits, other.is_keyword_hits);
        as_keyword_hits = std::max(as_keyword_hits, other.as_keyword_hits);
    }

    // --- Balance checking ---

    // Check if all confix types are balanced (open == close)
    // Returns true if all types balanced
    inline bool TypeEvidence::all_confix_balanced() const {
        for (size_t i = 0; i < 12; ++i) {
            if (confix_open[i] != confix_close[i]) {
                return false;
            }
        }
        return true;
    }

    // Check if specific confix type is balanced
    inline bool TypeEvidence::confix_balanced(ConfixType type) const {
        uint8_t idx = static_cast<uint8_t>(type);
        if (idx >= 12) return false;
        return confix_open[idx] == confix_close[idx];
    }

    // Get confix balance delta (open - close) for a type
    // Positive = more opens, Negative = more closes, Zero = balanced
    inline int16_t TypeEvidence::confix_delta(ConfixType type) const {
        uint8_t idx = static_cast<uint8_t>(type);
        if (idx >= 12) return 0;
        return static_cast<int16_t>(confix_open[idx]) - static_cast<int16_t>(confix_close[idx]);
    }

    // Get maximum nesting depth for a confix type
    inline uint16_t TypeEvidence::get_max_depth(ConfixType type) const {
        uint8_t idx = static_cast<uint8_t>(type);
        if (idx >= 12) return 0;
        return max_confix_depth[idx];
    }

    // Check for specific nesting patterns
    // Returns true if confix type A encloses type B
    inline bool TypeEvidence::encloses(ConfixType outer, ConfixType inner) const {
        // If outer has greater max depth than inner, it likely encloses
        return get_max_depth(outer) > get_max_depth(inner);
    }

    // Detect if this looks like a nested expression pattern
    // e.g., ((())) or {{{}}} or [][][]
    inline bool TypeEvidence::has_nested_pattern(ConfixType type) const {
        uint8_t idx = static_cast<uint8_t>(type);
        if (idx >= 12) return false;
        return max_confix_depth[idx] > 1;
    }

    // Detect mixed delimiter patterns: (){}[] or {[()]}
    inline std::vector<ConfixType> TypeEvidence::get_mixed_patterns() const {
        std::vector<ConfixType> result;
        for (size_t i = 1; i <= 4; ++i) {  // Check PAREN, BRACE, BRACKET, ANGLE
            if (confix_open[i] > 0 || confix_close[i] > 0) {
                result.push_back(static_cast<ConfixType>(i));
            }
        }
        return result;
    }

    // Compute nesting complexity score (higher = more complex)
    inline uint32_t TypeEvidence::compute_nesting_complexity() const {
        uint32_t score = 0;

        // Sum of max depths across all types
        for (size_t i = 0; i < 12; ++i) {
            score += max_confix_depth[i] * max_confix_depth[i];  // Square for complexity
        }

        // Penalty for imbalance
        for (size_t i = 0; i < 12; ++i) {
            int16_t delta = confix_delta(static_cast<ConfixType>(i));
            score += static_cast<uint32_t>(delta * delta * 10);  // Heavy penalty
        }

        // Bonus for mixed patterns (multiple types)
        auto mixed = get_mixed_patterns();
        if (mixed.size() > 1) {
            score += mixed.size() * 5;  // Modest bonus for complexity
        }

        return score;
    }

    // --- Confidence scores for language families ---

    inline double TypeEvidence::cpp2_confidence() const {
        if (total_tokens == 0) return 0.0;

        double score = 0.0;
        score += cpp2_keywords * 4.0;
        score += flow_keywords * 2.0;
        score += contract_keywords * 3.0;

        return std::min(1.0, score / static_cast<double>(std::max(uint16_t(1), total_tokens)));
    }

    inline double TypeEvidence::cpp_confidence() const {
        if (total_tokens == 0) return 0.0;

        double score = 0.0;
        score += cpp_keywords * 3.0;
        score += template_ids * 5.0;
        score += cpp_identifiers * 2.0;
        score += double_colon * 3.0;
        score += confix_open[static_cast<uint8_t>(ConfixType::ANGLE)] * 2.0;

        return std::min(1.0, score / static_cast<double>(std::max(uint16_t(1), total_tokens)));
    }

    inline double TypeEvidence::c_confidence() const {
        if (total_tokens == 0) return 0.0;

        double score = 0.0;
        score += c_keywords * 4.0;
        score += c_identifiers * 2.0;

        return std::min(1.0, score / static_cast<double>(std::max(uint16_t(1), total_tokens)));
    }

    // Deduce dominant language family
    inline TypeEvidence::LanguageFamily TypeEvidence::deduce_language() const {
        double c = c_confidence();
        double cpp = cpp_confidence();
        double cpp2 = cpp2_confidence();

        if (cpp2 > cpp && cpp2 > c) return LanguageFamily::CPP2;
        if (cpp > c) return LanguageFamily::CPP;
        if (c > 0.1) return LanguageFamily::C;
        return LanguageFamily::UNKNOWN;
    }
    // Token classification helpers
    inline bool TypeEvidence::is_c_identifier(std::string_view token) {
        // C identifiers: [a-zA-Z_][a-zA-Z0-9_]*
        if (token.empty()) return false;
        if (!std::isalpha(static_cast<unsigned char>(token[0])) && token[0] != '_') {
            return false;
        }
        for (size_t i = 1; i < token.size(); ++i) {
            if (!std::isalnum(static_cast<unsigned char>(token[i])) && token[i] != '_') {
                return false;
            }
        }
        return true;
    }

    inline bool TypeEvidence::is_c_keyword(std::string_view token) {
        static const std::array<std::string_view, 6> keywords{
            "typedef", "struct", "enum", "union", "extern", "sizeof"
        };
        for (auto kw : keywords) {
            if (token == kw) return true;
        }
        return false;
    }

    inline bool TypeEvidence::is_cpp_keyword(std::string_view token) {
        static const std::array<std::string_view, 14> keywords{
            "class", "template", "typename", "namespace", "constexpr", "concept",
            "requires", "decltype", "noexcept", "operator", "using", "friend",
            "virtual", "override"
        };
        for (auto kw : keywords) {
            if (token == kw) return true;
        }
        return false;
    }

    inline bool TypeEvidence::is_cpp2_keyword(std::string_view token) {
        static const std::array<std::string_view, 11> keywords{
            "inspect", "let", "mut", "in", "out", "inout", "move", "forward",
            "contract", "pre", "post"
        };
        for (auto kw : keywords) {
            if (token == kw) return true;
        }
        return false;
    }

} // namespace cppfort::stage0
