#include "lexer.h"

#include <cctype>
#include <sstream>
#include <stdexcept>

namespace cppfort::stage0 {

Lexer::Lexer(std::string source, std::string file)
    : m_source(std::move(source)), m_file(std::move(file)) {}

std::vector<Token> Lexer::tokenize() {
    m_tokens.clear();
    m_current = 0;
    m_line = 1;
    m_column = 1;

    while (!is_at_end()) {
        skip_whitespace();
        if (is_at_end()) {
            break;
        }

        auto token_start_offset = m_current;
        auto token_line = m_line;
        auto token_column = m_column;
        char c = advance();

        switch (c) {
            case ':':
                if (match(':')) {
                    add_token(TokenType::DoubleColon, token_start_offset, 2, token_line, token_column);
                } else {
                    add_token(TokenType::Colon, token_start_offset, 1, token_line, token_column);
                }
                break;
            case ';':
                add_token(TokenType::Semicolon, token_start_offset, 1, token_line, token_column);
                break;
            case ',':
                add_token(TokenType::Comma, token_start_offset, 1, token_line, token_column);
                break;
            case '(': 
                add_token(TokenType::LParen, token_start_offset, 1, token_line, token_column);
                break;
            case ')':
                add_token(TokenType::RParen, token_start_offset, 1, token_line, token_column);
                break;
            case '{':
                add_token(TokenType::LBrace, token_start_offset, 1, token_line, token_column);
                break;
            case '}':
                add_token(TokenType::RBrace, token_start_offset, 1, token_line, token_column);
                break;
            case '[':
                add_token(TokenType::LBracket, token_start_offset, 1, token_line, token_column);
                break;
            case ']':
                add_token(TokenType::RBracket, token_start_offset, 1, token_line, token_column);
                break;
            case '.':
                add_token(TokenType::Dot, token_start_offset, 1, token_line, token_column);
                break;
            case '<':
                add_token(TokenType::Less, token_start_offset, 1, token_line, token_column);
                break;
            case '>':
                add_token(TokenType::Greater, token_start_offset, 1, token_line, token_column);
                break;
            case '&':
                add_token(TokenType::Ampersand, token_start_offset, 1, token_line, token_column);
                break;
            case '*':
                add_token(TokenType::Star, token_start_offset, 1, token_line, token_column);
                break;
            case '+':
                add_token(TokenType::Plus, token_start_offset, 1, token_line, token_column);
                break;
            case '-':
                if (match('>')) {
                    add_token(TokenType::Arrow, token_start_offset, 2, token_line, token_column);
                } else {
                    add_token(TokenType::Minus, token_start_offset, 1, token_line, token_column);
                }
                break;
            case '/':
                add_token(TokenType::Slash, token_start_offset, 1, token_line, token_column);
                break;
            case '%':
                add_token(TokenType::Percent, token_start_offset, 1, token_line, token_column);
                break;
            case '|':
                add_token(TokenType::Pipe, token_start_offset, 1, token_line, token_column);
                break;
            case '^':
                add_token(TokenType::Caret, token_start_offset, 1, token_line, token_column);
                break;
            case '!':
                add_token(TokenType::Exclamation, token_start_offset, 1, token_line, token_column);
                break;
            case '?':
                add_token(TokenType::Question, token_start_offset, 1, token_line, token_column);
                break;
            case '=':
                add_token(TokenType::Equals, token_start_offset, 1, token_line, token_column);
                break;
            case '"':
                lex_string();
                break;
            default:
                if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
                    lex_identifier();
                } else if (std::isdigit(static_cast<unsigned char>(c))) {
                    lex_number();
                } else {
                    std::ostringstream oss;
                    oss << "Unexpected character '" << c << "' at " << token_line << ':' << token_column;
                    throw LexError(oss.str());
                }
                break;
        }
    }

    add_token(TokenType::EndOfFile, m_current, 0, m_line, m_column);
    return m_tokens;
}

bool Lexer::is_at_end() const noexcept {
    return m_current >= m_source.size();
}

char Lexer::peek() const noexcept {
    if (is_at_end()) {
        return '\0';
    }
    return m_source[m_current];
}

char Lexer::peek_next() const noexcept {
    if (m_current + 1 >= m_source.size()) {
        return '\0';
    }
    return m_source[m_current + 1];
}

char Lexer::advance() noexcept {
    char c = m_source[m_current++];
    if (c == '\n') {
        ++m_line;
        m_column = 1;
    } else {
        ++m_column;
    }
    return c;
}

void Lexer::add_token(TokenType type, std::size_t start_offset, std::size_t length,
    std::size_t line, std::size_t column) {
    add_token(type, start_offset, length, line, column, m_source.substr(start_offset, length));
}

void Lexer::add_token(TokenType type, std::size_t start_offset, std::size_t length,
    std::size_t line, std::size_t column, std::string lexeme_override) {
    SourceLocation loc {m_file, line, column};
    m_tokens.emplace_back(type, std::move(lexeme_override), std::move(loc), start_offset, length);
}

bool Lexer::match(char expected) noexcept {
    if (is_at_end()) {
        return false;
    }
    if (m_source[m_current] != expected) {
        return false;
    }
    advance();
    return true;
}

void Lexer::skip_whitespace() {
    while (!is_at_end()) {
        char c = peek();
        switch (c) {
            case ' ':
            case '\r':
            case '\t':
                advance();
                break;
            case '\n':
                advance();
                break;
            case '/':
                if (peek_next() == '/') {
                    skip_single_line_comment();
                } else if (peek_next() == '*') {
                    skip_multi_line_comment();
                } else {
                    return;
                }
                break;
            case '#':
                // Skip preprocessor directives (for bidirectional compatibility)
                while (!is_at_end() && peek() != '\n') {
                    advance();
                }
                break;
            default:
                return;
        }
    }
}

void Lexer::skip_single_line_comment() {
    while (!is_at_end() && peek() != '\n') {
        advance();
    }
}

void Lexer::skip_multi_line_comment() {
    advance(); // consume '/'
    advance(); // consume '*'
    while (!is_at_end()) {
        if (peek() == '*' && peek_next() == '/') {
            advance(); // '*'
            advance(); // '/'
            return;
        }
        advance();
    }
    throw LexError("Unterminated block comment");
}

void Lexer::lex_identifier() {
    auto start_offset = m_current - 1;
    auto start_line = m_line;
    auto start_column = m_column - 1;

    while (!is_at_end()) {
        char c = peek();
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '$') {
            advance();
        } else if (c == ':' && peek_next() == ':') {
            // allow identifiers to include :: for fully qualified names
            advance();
            advance();
        } else {
            break;
        }
    }

    auto length = m_current - start_offset;
    auto text = m_source.substr(start_offset, length);

    if (text == "::") {
        add_token(TokenType::DoubleColon, start_offset, length, start_line, start_column);
        return;
    }

    TokenType type = keyword_type(text);
    if (type != TokenType::Identifier) {
        add_token(type, start_offset, length, start_line, start_column);
    } else {
        add_token(TokenType::Identifier, start_offset, length, start_line, start_column);
    }
}

void Lexer::lex_number() {
    auto start_offset = m_current - 1;
    auto start_line = m_line;
    auto start_column = m_column - 1;

    while (!is_at_end() && std::isdigit(static_cast<unsigned char>(peek()))) {
        advance();
    }

    if (!is_at_end() && peek() == '.' && std::isdigit(static_cast<unsigned char>(peek_next()))) {
        advance();
        while (!is_at_end() && std::isdigit(static_cast<unsigned char>(peek()))) {
            advance();
        }
    }

    auto length = m_current - start_offset;
    add_token(TokenType::Number, start_offset, length, start_line, start_column);
}

void Lexer::lex_string() {
    auto start_offset = m_current - 1;
    auto start_line = m_line;
    auto start_column = m_column - 1;

    bool terminated = false;
    while (!is_at_end()) {
        char c = advance();
        if (c == '"') {
            terminated = true;
            break;
        }
        if (c == '\\' && !is_at_end()) {
            advance();
        }
    }

    if (!terminated) {
        throw LexError("Unterminated string literal");
    }

    auto length = m_current - start_offset;
    add_token(TokenType::String, start_offset, length, start_line, start_column);
}

TokenType Lexer::keyword_type(const std::string& identifier) const noexcept {
    // Basic Cpp2 keywords
    if (identifier == "return") {
        return TokenType::KeywordReturn;
    }
    if (identifier == "if") {
        return TokenType::KeywordIf;
    }
    if (identifier == "else") {
        return TokenType::KeywordElse;
    }
    if (identifier == "for") {
        return TokenType::KeywordFor;
    }
    if (identifier == "while") {
        return TokenType::KeywordWhile;
    }
    if (identifier == "do") {
        return TokenType::KeywordDo;
    }
    if (identifier == "assert") {
        return TokenType::KeywordAssert;
    }
    if (identifier == "pre") {
        return TokenType::KeywordPre;
    }
    if (identifier == "post") {
        return TokenType::KeywordPost;
    }
    if (identifier == "using") {
        return TokenType::KeywordUsing;
    }
    if (identifier == "namespace") {
        return TokenType::KeywordNamespace;
    }
    if (identifier == "type") {
        return TokenType::KeywordType;
    }
    if (identifier == "public") {
        return TokenType::KeywordPublic;
    }
    if (identifier == "protected") {
        return TokenType::KeywordProtected;
    }
    if (identifier == "private") {
        return TokenType::KeywordPrivate;
    }
    if (identifier == "virtual") {
        return TokenType::KeywordVirtual;
    }
    if (identifier == "override") {
        return TokenType::KeywordOverride;
    }
    if (identifier == "final") {
        return TokenType::KeywordFinal;
    }
    if (identifier == "in") {
        return TokenType::KeywordIn;
    }
    if (identifier == "inout") {
        return TokenType::KeywordInout;
    }
    if (identifier == "out") {
        return TokenType::KeywordOut;
    }
    if (identifier == "copy") {
        return TokenType::KeywordCopy;
    }
    if (identifier == "move") {
        return TokenType::KeywordMove;
    }
    if (identifier == "forward") {
        return TokenType::KeywordForward;
    }

    return TokenType::Identifier;
}

} // namespace cppfort::stage0
