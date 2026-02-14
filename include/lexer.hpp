#pragma once

// Core types extracted to modular headers
#include "core/tokens.hpp"

#include <string>
#include <string_view>
#include <vector>
#include <cstddef>
#include <cstdint>

namespace cpp2_transpiler {

// TokenType and Token are now defined in core/tokens.hpp
// This header re-exports them for backward compatibility

class Lexer {
public:
    explicit Lexer(std::string_view source);
    explicit Lexer(const std::string& source);

    std::vector<Token> tokenize();

    // Check if any Cpp2-specific syntax was found during tokenization
    // Used for mixed-mode passthrough (pure C++ files skip transpilation)
    bool has_cpp2_syntax() const { return m_has_cpp2_syntax; }

private:
    std::string_view source;
    std::size_t current;
    std::size_t line;
    std::size_t column;
    std::size_t start;

    std::vector<Token> tokens;

    // Flag to track if any Cpp2-specific syntax was found
    bool m_has_cpp2_syntax = false;

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
    void scan_string_with_prefix();  // u", U", u8", L" string prefixes
    void scan_raw_string(bool is_interpolated = false);  // C++11 R"..." or Cpp2 $R"..." raw string literals
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