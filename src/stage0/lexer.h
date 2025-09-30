#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "token.h"

namespace cppfort::stage0 {

struct LexError : std::runtime_error {
    explicit LexError(const std::string& message)
        : std::runtime_error(message) {}
};

class Lexer {
  public:
    Lexer(std::string source, std::string file);

    [[nodiscard]] std::vector<Token> tokenize();
    [[nodiscard]] const std::string& source() const noexcept { return m_source; }

  private:
    [[nodiscard]] bool is_at_end() const noexcept;
    [[nodiscard]] char peek() const noexcept;
    [[nodiscard]] char peek_next() const noexcept;
    char advance() noexcept;

    void add_token(TokenType type, std::size_t start_offset, std::size_t length,
        std::size_t line, std::size_t column);
    void add_token(TokenType type, std::size_t start_offset, std::size_t length,
        std::size_t line, std::size_t column, std::string lexeme_override);
    void skip_whitespace();
    void skip_single_line_comment();
    void skip_multi_line_comment();

    void lex_identifier();
    void lex_number();
    void lex_string();
    void lex_char();
    void lex_preprocessor();

    [[nodiscard]] bool match(char expected) noexcept;

    TokenType keyword_type(const std::string& identifier) const noexcept;

  private:
    std::string m_source;
    std::string m_file;
    std::vector<Token> m_tokens;
    std::size_t m_current {0};
    std::size_t m_line {1};
    std::size_t m_column {1};
};

} // namespace cppfort::stage0
