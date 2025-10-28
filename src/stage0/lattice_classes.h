#pragma once

#include <cstdint>

namespace cppfort::stage0 {

/**
 * Character Class Lattice Bit Positions (16-bit for efficiency)
 *
 * Extracted from build/x.txt heuristic_search_tiles.cpp2
 * Used for byte-level classification in orbit detection
 */
namespace LatticeClasses {
    constexpr uint16_t DIGIT       = 1 << 0;   // 0-9
    constexpr uint16_t ALPHA       = 1 << 1;   // a-z, A-Z
    constexpr uint16_t PUNCTUATION = 1 << 2;   // .,;:!?
    constexpr uint16_t WHITESPACE  = 1 << 3;   // space, tab, newline
    constexpr uint16_t STRUCTURAL  = 1 << 4;   // {}[]()
    constexpr uint16_t NUMERIC_OP  = 1 << 5;   // +-.eE
    constexpr uint16_t QUOTE       = 1 << 6;   // "'"`
    constexpr uint16_t BOOLEAN     = 1 << 7;   // t,r,u,e,f,a,l,s (for true/false)
    constexpr uint16_t OPERATOR    = 1 << 8;   // =,+,-,*,/,<,>,==,etc.
    constexpr uint16_t IDENTIFIER  = 1 << 9;   // Mix of alpha + underscore/digit after
    constexpr uint16_t COMMENT     = 1 << 10;  // // or /*
    constexpr uint16_t PREPROCESS  = 1 << 11;  // #
    constexpr uint16_t KEYWORD     = 1 << 12;  // if,for,while,type,namespace,etc.
    constexpr uint16_t LITERAL     = 1 << 13;  // string, char literals
    constexpr uint16_t BRACKET     = 1 << 14;  // < > for templates
    constexpr uint16_t SEMICOLON   = 1 << 15;  // ; for statements
}

/**
 * Classify single byte into lattice mask
 * Returns bitmask of all applicable character classes
 *
 * Note: Intentional collision - single byte may map to multiple classes
 * Example: 't' -> ALPHA | BOOLEAN | IDENTIFIER | KEYWORD
 */
inline uint16_t classify_byte(char byte) {
    uint16_t mask = 0;

    if (std::isdigit(static_cast<unsigned char>(byte))) {
        mask |= LatticeClasses::DIGIT;
    }

    if (std::isalpha(static_cast<unsigned char>(byte))) {
        mask |= LatticeClasses::ALPHA;

        // Check for boolean letters
        switch (std::tolower(static_cast<unsigned char>(byte))) {
            case 't': case 'r': case 'u': case 'e':
            case 'f': case 'a': case 'l': case 's':
                mask |= LatticeClasses::BOOLEAN;
                break;
        }
    }

    if (std::isspace(static_cast<unsigned char>(byte))) {
        mask |= LatticeClasses::WHITESPACE;
    }

    switch (byte) {
        case '.': case ',': case ';': case ':': case '!': case '?':
            mask |= LatticeClasses::PUNCTUATION;
            if (byte == ';') mask |= LatticeClasses::SEMICOLON;
            break;

        case '{': case '}': case '[': case ']': case '(': case ')':
            mask |= LatticeClasses::STRUCTURAL;
            break;

        case '+': case '-': case '*': case '/': case '%': case '=':
        case '<': case '>': case '&': case '|': case '^': case '~':
            mask |= LatticeClasses::NUMERIC_OP | LatticeClasses::OPERATOR;
            if (byte == '<' || byte == '>') mask |= LatticeClasses::BRACKET;
            break;

        case '"': case '\'': case '`':
            mask |= LatticeClasses::QUOTE;
            break;

        case '#':
            mask |= LatticeClasses::PREPROCESS;
            break;

        case '_':
            mask |= LatticeClasses::IDENTIFIER;
            break;
    }

    // Detect potential keywords/literals via context (simplified)
    if (std::isalpha(static_cast<unsigned char>(byte)) || byte == '_') {
        mask |= LatticeClasses::IDENTIFIER | LatticeClasses::KEYWORD;
    }

    if (byte == '"') {
        mask |= LatticeClasses::LITERAL;
    }

    return mask;
}

} // namespace cppfort::stage0
