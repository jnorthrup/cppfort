#pragma once

// Core types extracted to modular headers
#include "core/tokens.hpp"

#include <string>
#include <string_view>
#include <vector>
#include <cstddef>
#include <cstdint>

namespace cpp2_transpiler {

/**
 * Lexer for the cppfort language.
 *
 * The lexer can operate in two modes:
 *   * pure‑Cpp2 – default mode, detects Cpp2‑specific constructs.
 *   * mixed‑mode – when a source file contains both legacy C++1 code and
 *                 Cpp2 syntax. In this mode the lexer suppresses Cpp2‑only
 *                 tokens so the existing Cpp2 parser can still handle the file.
 */
class Lexer {
public:
    explicit Lexer(std::string_view source);
    explicit Lexer(const std::string& source);

    // Tokenise the whole source and return the token vector.
    std::vector<Token> tokenize();

    // Returns true if the lexer detected any Cpp2‑specific syntax during the
    // scan. Used for the original mixed‑mode passthrough behaviour.
    bool has_cpp2_syntax() const { return m_has_cpp2_syntax; }

    // Detect if a source string contains both Cpp2 and legacy C++1 syntax.
    // Simple heuristic: presence of a "#pragma mixed-mode" directive or a
    // comment marker "// MIXED_MODE" (case‑insensitive).
    static bool isMixedMode(const std::string& src);

    // Returns true when the current lexing session is in mixed‑mode.
    bool mixedMode() const { return m_is_mixed; }

private:
    std::string_view source;
    std::size_t current = 0;
    std::size_t line = 1;
    std::size_t column = 1;
    std::size_t start = 0;

    std::vector<Token> tokens;

    // Flag to track if any Cpp2‑specific syntax was found.
    bool m_has_cpp2_syntax = false;
    // Flag indicating mixed‑mode (legacy C++1 + Cpp2).
    bool m_is_mixed = false;

    void scan_token();

    char advance();
    char peek() const;
    char peek_next() const;
    bool match(char expected);
    bool is_at_end() const;

    void add_token(TokenType type);
    void add_token(TokenType type, std::string_view lexeme);

    void scan_identifier();
    void scan_number();
    void scan_string();
    void scan_string_with_prefix(); // u", U", u8", L" string prefixes
    void scan_raw_string(bool is_interpolated = false); // C++11 R"..." or Cpp2 $R"..."
    void scan_character();
    void scan_line_comment();
    void scan_block_comment();
    void scan_markdown_block();
    void scan_preprocessor();
    void scan_cpp26_attribute();

    bool is_digit(char c) const;
    bool is_hex_digit(char c) const;
    bool is_identifier_start(char c) const;
    bool is_identifier_char(char c) const;

    TokenType check_keyword(std::size_t start, std::size_t length,
                           const char* rest, TokenType type);
    TokenType identifier_type();

    void skip_whitespace();
};

} // namespace cpp2_transpiler
