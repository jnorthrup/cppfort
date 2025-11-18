#pragma once

#include <cstdint>
#include <cctype>
#include "type_evidence.h"

namespace cppfort::stage0 {

// ─────────────────────────────────────────────────────────────────────────────
// Character Class Lattice - Byte-level classification for 2D evidence system
// 
// Each byte feeds into the multidimensional TypeEvidence counters:
// - Layer 1: Character classification (digits, alpha, punctuation)
// - Layer 2: Confix type tracking (PAREN, BRACE, BRACKET, ANGLE)
// - Layer 3+: Identifier, keyword, literal tracking
// 
// This bridges raw bytes to EvidenceSpan locality bubbles.
// ─────────────────────────────────────────────────────────────────────────────

enum class CharClass : uint16_t {
    None      = 0,
    Whitespace = 1 << 0,
    Digit      = 1 << 1,   // 0-9
    Alpha      = 1 << 2,   // a-z A-Z _
    Punct      = 1 << 3,   // . , ; : ! ? etc
    Operator   = 1 << 4,   // + - * / = < > & | etc
    Structural = 1 << 5,   // { } [ ] ( ) < > - confix boundaries
    Newline    = 1 << 6,   // \n \r
    StringDelim = 1 << 7,  // " '
    
    // Evidence-tracking categories for TypeEvidence integration
    IdentifierStart = 1 << 8,  // First char of identifier (alpha or _)
    NumericLiteral  = 1 << 9,  // Digits and . for numbers
    Keyword         = 1 << 10, // Potential keyword letters
    Literal         = 1 << 11, // String/char literal content
    
    // Rapid pattern matching masks for EvidenceSpan locality
    AnchorCandidate = 1 << 12  // Likely pattern boundary char
};

// Check if a byte has a specific character class
inline bool has_class(uint16_t mask, CharClass cls) {
    return mask & static_cast<uint16_t>(cls);
}

// Classify a single byte - returns bitmask of all applicable classes
inline uint16_t classify_byte(char c) {
    uint16_t mask = static_cast<uint16_t>(CharClass::None);
    unsigned char uc = static_cast<unsigned char>(c);
    
    // Basic character classes (Layer 1 of TypeEvidence)
    if (std::isspace(uc)) {
        mask |= static_cast<uint16_t>(CharClass::Whitespace);
        if (c == '\n' || c == '\r') {
            mask |= static_cast<uint16_t>(CharClass::Newline);
        }
    }
    else if (std::isdigit(uc)) {
        mask |= static_cast<uint16_t>(CharClass::Digit);
        mask |= static_cast<uint16_t>(CharClass::NumericLiteral);
    }
    else if (std::isalpha(uc) || c == '_') {
        mask |= static_cast<uint16_t>(CharClass::Alpha);
        mask |= static_cast<uint16_t>(CharClass::IdentifierStart);
        mask |= static_cast<uint16_t>(CharClass::Keyword);  // May be keyword
    }
    
    // Structural characters (Layer 2 of TypeEvidence - confix tracking)
    switch (c) {
        case '{': case '}':
        case '[': case ']':
        case '(': case ')':
        case '<': case '>':
            mask |= static_cast<uint16_t>(CharClass::Structural);
            mask |= static_cast<uint16_t>(CharClass::AnchorCandidate);
            break;
            
        case '.': case ',': case ';': case ':': 
            mask |= static_cast<uint16_t>(CharClass::Punct);
            mask |= static_cast<uint16_t>(CharClass::AnchorCandidate);
            break;
            
        case '+': case '-': case '*': case '/': case '%':
        case '=': case '!':
        case '&': case '|': case '^':
            mask |= static_cast<uint16_t>(CharClass::Operator);
            mask |= static_cast<uint16_t>(CharClass::AnchorCandidate);
            break;
            
        case '"': case '\'':
            mask |= static_cast<uint16_t>(CharClass::StringDelim);
            mask |= static_cast<uint16_t>(CharClass::Literal);
            mask |= static_cast<uint16_t>(CharClass::AnchorCandidate);
            break;
            
        case '#':  // Preprocessor
            mask |= static_cast<uint16_t>(CharClass::AnchorCandidate);
            break;
    }
    
    return mask;
}

// ─────────────────────────────────────────────────────────────────────────────
// TypeEvidence integration - direct mapping to 2D evidence system
// These feed into the multidimensional counters for EvidenceSpan locality
// ─────────────────────────────────────────────────────────────────────────────

// Check if this byte starts an identifier (feeds into TypeEvidence identifier tracking)
inline bool is_identifier_start(char c) {
    return has_class(classify_byte(c), CharClass::IdentifierStart);
}

// Check if this byte could be part of a numeric literal (feeds into TypeEvidence)
inline bool is_numeric_literal_char(char c) {
    uint16_t mask = classify_byte(c);
    return has_class(mask, CharClass::Digit) || 
           has_class(mask, CharClass::Punct);  // For '.' in floats
}

// Check if this byte is structural (feeds into TypeEvidence confix tracking)
inline bool is_structural(char c) {
    return has_class(classify_byte(c), CharClass::Structural);
}

// Check if this byte is a likely pattern anchor point (for EvidenceSpan boundaries)
inline bool is_anchor_candidate(char c) {
    return has_class(classify_byte(c), CharClass::AnchorCandidate);
}

// ─────────────────────────────────────────────────────────────────────────────
// 2D Evidence System Integration
// Maps lattice classes to TypeEvidence's multidimensional structure
// This operates within EvidenceSpan locality bubbles
// ─────────────────────────────────────────────────────────────────────────────

// Update TypeEvidence from lattice classification - works within EvidenceSpan locality
inline void update_type_evidence_from_lattice(char c, TypeEvidence& evidence) {
    uint16_t mask = classify_byte(c);
    
    // Layer 1: Character classification (feeds TypeEvidence Layer 1)
    if (has_class(mask, CharClass::Digit)) {
        evidence.digits++;
        evidence.number_literals++;
    }
    if (has_class(mask, CharClass::Alpha)) {
        evidence.alpha++;
    }
    if (has_class(mask, CharClass::Whitespace)) {
        evidence.whitespaces++;
        if (has_class(mask, CharClass::Newline)) {
            evidence.linefeed++;
        }
    }
    if (has_class(mask, CharClass::StringDelim)) {
        if (c == '"') evidence.dquotes++;
        if (c == '\'') evidence.quotes++;
    }
    
    // Layer 2: Confix type tracking (feeds TypeEvidence Layer 2)
    // This is the 2D evidence system: [confix_type][open/close/depth]
    if (has_class(mask, CharClass::Structural)) {
        ConfixType confix_type = ConfixType::INVALID;
        bool is_open = false;
        
        switch (c) {
            case '{': confix_type = ConfixType::BRACE; is_open = true; break;
            case '}': confix_type = ConfixType::BRACE; is_open = false; break;
            case '(': confix_type = ConfixType::PAREN; is_open = true; break;
            case ')': confix_type = ConfixType::PAREN; is_open = false; break;
            case '[': confix_type = ConfixType::BRACKET; is_open = true; break;
            case ']': confix_type = ConfixType::BRACKET; is_open = false; break;
            case '<': confix_type = ConfixType::ANGLE; is_open = true; break;
            case '>': confix_type = ConfixType::ANGLE; is_open = false; break;
        }
        
        if (confix_type != ConfixType::INVALID) {
            uint8_t type_idx = static_cast<uint8_t>(confix_type);
            if (is_open) {
                evidence.confix_open[type_idx]++;
                // Compute current depth and update max depth if needed
                int current_depth = static_cast<int>(evidence.confix_open[type_idx]) - static_cast<int>(evidence.confix_close[type_idx]);
                if (current_depth > static_cast<int>(evidence.max_confix_depth[type_idx])) {
                    evidence.max_confix_depth[type_idx] = static_cast<uint16_t>(current_depth);
                }
            } else {
                evidence.confix_close[type_idx]++;
                int current_depth = static_cast<int>(evidence.confix_open[type_idx]) - static_cast<int>(evidence.confix_close[type_idx]);
                if (current_depth < static_cast<int>(evidence.min_confix_depth[type_idx])) {
                    evidence.min_confix_depth[type_idx] = static_cast<uint16_t>(current_depth);
                }
            }
        }
    }
    
    // Layer 3+: Expression structure (feeds TypeEvidence Layer 5)
    if (has_class(mask, CharClass::Operator)) {
        evidence.special++;  // Count operators
        if (c == ':') evidence.colon++;
        if (c == ',') evidence.comma++;
        if (c == ';') evidence.semicolon++;
    }
    if (has_class(mask, CharClass::Punct)) {
        evidence.special++;  // Count punctuation
    }
}

// Get the dominant confix type for a structural character (for 2D evidence)
inline ConfixType get_confix_type(char c) {
    switch (c) {
        case '{': case '}': return ConfixType::BRACE;
        case '(': case ')': return ConfixType::PAREN;
        case '[': case ']': return ConfixType::BRACKET;
        case '<': case '>': return ConfixType::ANGLE;
        default: return ConfixType::INVALID;
    }
}

// Check if a span has balanced confixes within EvidenceSpan locality
inline bool has_balanced_confixes(const TypeEvidence& evidence) {
    for (uint8_t i = 1; i < static_cast<uint8_t>(ConfixType::MAX_TYPE); ++i) {
        if (evidence.confix_open[i] != evidence.confix_close[i]) {
            return false;
        }
    }
    return true;
}

// Get confix depth vector for EvidenceSpan validation
inline std::array<int, 12> get_confix_depths(const TypeEvidence& evidence) {
    std::array<int, 12> depths = {0};
    for (uint8_t i = 0; i < 12; ++i) {
        depths[i] = evidence.max_confix_depth[i] - evidence.min_confix_depth[i];
    }
    return depths;
}

} // namespace cppfort::stage0

// Temporary LatticeClasses for heuristic_grid.h
enum LatticeClasses : uint16_t {
    DIGIT = 1 << 1,
    ALPHA = 1 << 2,
    PUNCTUATION = 1 << 3,
    STRUCTURAL = 1 << 5,
    SEMICOLON = 1 << 3, // same as PUNCTUATION
    IDENTIFIER = 1 << 2, // same as ALPHA
    QUOTE = 1 << 7,
    NUMERIC_OP = 1 << 4,
};
