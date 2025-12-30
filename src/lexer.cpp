#include "lexer.hpp"
#include "utils.hpp"
#include <unordered_map>
#include <cassert>
#include <cctype>

namespace cpp2_transpiler {

Lexer::Lexer(std::string_view source)
    : source(source), current(0), line(1), column(1), start(0) {}

Lexer::Lexer(const std::string& source)
    : Lexer(std::string_view(source)) {}

std::vector<Token> Lexer::tokenize() {
    while (!is_at_end()) {
        start = current;
        scan_token();
    }

    tokens.emplace_back(TokenType::EndOfFile, "", line, column, current);
    return tokens;
}

void Lexer::scan_token() {
    char c = advance();

    switch (c) {
        // Single character tokens
        case '(': add_token(TokenType::LeftParen); break;
        case ')': add_token(TokenType::RightParen); break;
        case '[': add_token(TokenType::LeftBracket); break;
        case ']': add_token(TokenType::RightBracket); break;
        case '{': add_token(TokenType::LeftBrace); break;
        case '}': add_token(TokenType::RightBrace); break;
        case ',': add_token(TokenType::Comma); break;
        case ';': add_token(TokenType::Semicolon); break;
        case '.': {
            if (match('.')) {
                if (match('.')) {
                    add_token(TokenType::TripleDot);
                } else {
                    add_token(TokenType::DoubleDot);
                    if (match('<')) {
                        tokens.back().type = TokenType::RangeExclusive;
                    } else if (match('=')) {
                        tokens.back().type = TokenType::RangeInclusive;
                    }
                }
            } else {
                add_token(TokenType::Dot);
            }
            break;
        }
        case '-': {
            if (match('>')) {
                add_token(TokenType::Arrow);
            } else if (match('-')) {
                add_token(TokenType::MinusMinus);
            } else if (match('=')) {
                add_token(TokenType::MinusEqual);
            } else {
                add_token(TokenType::Minus);
            }
            break;
        }
        case '+': {
            if (match('+')) {
                add_token(TokenType::PlusPlus);
            } else if (match('=')) {
                add_token(TokenType::PlusEqual);
            } else {
                add_token(TokenType::Plus);
            }
            break;
        }
        case '*': {
            if (match('*')) {
                add_token(TokenType::Exponentiation);
            } else if (match('=')) {
                add_token(TokenType::AsteriskEqual);
            } else {
                add_token(TokenType::Asterisk);
            }
            break;
        }
        case '%': {
            if (match('=')) {
                add_token(TokenType::PercentEqual);
            } else {
                add_token(TokenType::Percent);
            }
            break;
        }
        case '!': {
            if (match('=')) {
                add_token(TokenType::NotEqual);
            } else {
                add_token(TokenType::Exclamation);
            }
            break;
        }
        case '=': {
            if (match('=')) {
                add_token(TokenType::DoubleEqual);
            } else if (match('>')) {
                add_token(TokenType::FatArrow);
            } else if (match(':')) {
                add_token(TokenType::EqualColon);
            } else {
                add_token(TokenType::Equal);
            }
            break;
        }
        case '<': {
            if (match('=')) {
                add_token(TokenType::LessThanOrEqual);
            } else if (match('<')) {
                if (match('=')) {
                    add_token(TokenType::LeftShiftEqual);
                } else {
                    add_token(TokenType::LeftShift);
                }
            } else {
                add_token(TokenType::LessThan);
            }
            break;
        }
        case '>': {
            if (match('=')) {
                add_token(TokenType::GreaterThanOrEqual);
            } else if (match('>')) {
                if (match('=')) {
                    add_token(TokenType::RightShiftEqual);
                } else {
                    add_token(TokenType::RightShift);
                }
            } else {
                add_token(TokenType::GreaterThan);
            }
            break;
        }
        case '&': {
            if (match('&')) {
                add_token(TokenType::DoubleAmpersand);
            } else if (match('=')) {
                add_token(TokenType::AmpersandEqual);
            } else {
                add_token(TokenType::Ampersand);
            }
            break;
        }
        case '|': {
            if (match('|')) {
                add_token(TokenType::DoublePipe);
            } else if (match('=')) {
                add_token(TokenType::PipeEqual);
            } else {
                add_token(TokenType::Pipe);
            }
            break;
        }
        case '^': {
            if (match('=')) {
                add_token(TokenType::CaretEqual);
            } else {
                add_token(TokenType::Caret);
            }
            break;
        }
        case '~': add_token(TokenType::Tilde); break;
        case '?': {
            if (match(':')) {
                add_token(TokenType::Elvis);
            } else {
                add_token(TokenType::Question);
            }
            break;
        }
        case ':': {
            if (match(':')) {
                add_token(TokenType::DoubleColon);
            } else if (match('=')) {
                add_token(TokenType::ColonEqual);
            } else {
                add_token(TokenType::Colon);
            }
            break;
        }
        case '$': add_token(TokenType::Dollar); break;
        case '@': add_token(TokenType::At); break;
        case '_': add_token(TokenType::Underscore); break;
        case '#': scan_preprocessor(); break;

        // String and character literals
        case '"': scan_string(); break;
        case '\'': scan_character(); break;

        // Comments
        case '/': {
            if (match('/')) {
                scan_line_comment();
            } else if (match('*')) {
                scan_block_comment();
            } else if (match('=')) {
                add_token(TokenType::SlashEqual);
            } else {
                add_token(TokenType::Slash);
            }
            break;
        }

        // Markdown blocks (bare ``` syntax)
        case '`': {
            if (peek() == '`' && peek_next() == '`') {
                advance(); // second `
                advance(); // third `
                scan_markdown_block();
            } else {
                add_token(TokenType::Unknown);
            }
            break;
        }

        default:
            if (is_digit(c)) {
                scan_number();
            } else if (is_identifier_start(c)) {
                scan_identifier();
            } else if (!std::isspace(static_cast<unsigned char>(c))) {
                add_token(TokenType::Unknown);
            }
            break;
    }
}

char Lexer::advance() {
    char c = source[current++];
    if (c == '\n') {
        line++;
        column = 1;
    } else {
        column++;
    }
    return c;
}

char Lexer::peek() const {
    return is_at_end() ? '\0' : source[current];
}

char Lexer::peek_next() const {
    return current + 1 >= source.length() ? '\0' : source[current + 1];
}

bool Lexer::match(char expected) {
    if (is_at_end()) return false;
    if (source[current] != expected) return false;

    advance();
    return true;
}

bool Lexer::is_at_end() const {
    return current >= source.length();
}

void Lexer::add_token(TokenType type) {
    add_token(type, source.substr(start, current - start));
}

void Lexer::add_token(TokenType type, std::string_view lexeme) {
    tokens.emplace_back(type, lexeme, line, column - lexeme.length(), start);
}

void Lexer::scan_identifier() {
    while (is_identifier_char(peek())) {
        advance();
    }

    std::string_view text = source.substr(start, current - start);
    TokenType type = identifier_type();
    add_token(type, text);
}

void Lexer::scan_number() {
    while (is_digit(peek())) {
        advance();
    }

    // Check for fractional part
    if (peek() == '.' && is_digit(peek_next())) {
        // Consume the decimal point
        advance();

        while (is_digit(peek())) {
            advance();
        }

        // Check for exponent
        if (peek() == 'e' || peek() == 'E') {
            advance();
            if (peek() == '+' || peek() == '-') {
                advance();
            }
            if (!is_digit(peek())) {
                add_token(TokenType::Unknown);
                return;
            }
            while (is_digit(peek())) {
                advance();
            }
        }

        add_token(TokenType::FloatLiteral);
    } else {
        add_token(TokenType::IntegerLiteral);
    }
}

void Lexer::scan_string() {
    while (peek() != '"' && !is_at_end()) {
        if (peek() == '\n') {
            line++;
            column = 1;
        }
        if (peek() == '\\') {
            advance(); // Skip escape character
        }
        advance();
    }

    if (is_at_end()) {
        add_token(TokenType::Unknown);
        return;
    }

    advance(); // Closing quote
    add_token(TokenType::StringLiteral);
}

void Lexer::scan_character() {
    if (is_at_end()) {
        add_token(TokenType::Unknown);
        return;
    }

    if (peek() == '\\') {
        advance(); // Skip escape character
    }

    advance(); // Character

    if (peek() != '\'') {
        add_token(TokenType::Unknown);
        return;
    }

    advance(); // Closing quote
    add_token(TokenType::CharacterLiteral);
}

void Lexer::scan_line_comment() {
    while (peek() != '\n' && !is_at_end()) {
        advance();
    }
}

void Lexer::scan_preprocessor() {
    // Scan the entire preprocessor directive (from # to end of line)
    std::size_t start = current - 1; // Include the '#' character
    while (peek() != '\n' && !is_at_end()) {
        advance();
    }
    // Add the Hash token with the full directive text
    add_token(TokenType::Hash, source.substr(start, current - start));
}

void Lexer::scan_block_comment() {
    // Check for markdown block syntax: /*```
    if (peek() == '`' && peek_next() == '`' && current + 2 < source.length() && source[current + 2] == '`') {
        // This is a markdown block wrapped in comment
        advance(); // first `
        advance(); // second `
        advance(); // third `
        scan_markdown_block();
        // After scan_markdown_block, verify closing */
        if (peek() == '*' && peek_next() == '/') {
            advance(); // *
            advance(); // /
        }
        return;
    }

    // Regular block comment
    while (!is_at_end() && !(peek() == '*' && peek_next() == '/')) {
        if (peek() == '\n') {
            line++;
            column = 1;
        }
        advance();
    }

    if (is_at_end()) {
        return; // Unterminated comment
    }

    advance(); // *
    advance(); // /
}

void Lexer::scan_markdown_block() {
    // At this point we've already consumed: ```
    // Now we need to capture everything until closing ```

    std::size_t content_start = current;

    // Skip optional name/identifier immediately after opening ```
    while (!is_at_end() && peek() != '\n' && !std::isspace(static_cast<unsigned char>(peek()))) {
        advance();
    }

    // Skip whitespace/newline after name
    if (peek() == '\n' || std::isspace(static_cast<unsigned char>(peek()))) {
        if (peek() == '\n') {
            line++;
            column = 1;
        }
        advance();
    }

    // Now capture content until we find closing ```
    while (!is_at_end()) {
        // Check for closing delimiter: ```
        if (peek() == '`' && current + 2 < source.length()) {
            if (source[current] == '`' &&
                source[current + 1] == '`' &&
                source[current + 2] == '`') {
                // Found closing delimiter
                std::size_t content_end = current;

                // Extract the full markdown block content (including name if present)
                std::string_view content = source.substr(content_start, content_end - content_start);

                // Skip past ```
                advance(); // first `
                advance(); // second `
                advance(); // third `

                // Add token with content
                add_token(TokenType::MarkdownBlock, content);
                return;
            }
        }

        if (peek() == '\n') {
            line++;
            column = 1;
        }
        advance();
    }

    // Unterminated markdown block
    // Still add what we have as a token
    std::string_view content = source.substr(content_start, current - content_start);
    add_token(TokenType::MarkdownBlock, content);
}

bool Lexer::is_digit(char c) const {
    return std::isdigit(static_cast<unsigned char>(c));
}

bool Lexer::is_identifier_start(char c) const {
    return std::isalpha(static_cast<unsigned char>(c)) || c == '_';
}

bool Lexer::is_identifier_char(char c) const {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

TokenType Lexer::check_keyword(std::size_t start, std::size_t length,
                               const char* rest, TokenType type) {
    if (current - this->start == start + length &&
        std::equal(rest, rest + length, source.begin() + this->start + start)) {
        return type;
    }
    return TokenType::Identifier;
}

TokenType Lexer::identifier_type() {
    std::string_view text = source.substr(start, current - start);
    static const std::unordered_map<std::string, TokenType, cpp2_transpiler::SimpleStringHash> keywords = {
        {"as", TokenType::As}, {"base", TokenType::Base}, {"break", TokenType::Break}, {"case", TokenType::Case}, {"class", TokenType::Class},
        {"concept", TokenType::Concept}, {"const", TokenType::Const}, {"continue", TokenType::Continue}, {"do", TokenType::Do}, {"else", TokenType::Else},
        {"enum", TokenType::Enum}, {"explicit", TokenType::Explicit}, {"final", TokenType::Final}, {"for", TokenType::For},
        {"func", TokenType::Func}, {"if", TokenType::If}, {"import", TokenType::Import}, {"in", TokenType::In},
        {"inspect", TokenType::Inspect}, {"interface", TokenType::Interface}, {"is", TokenType::Is}, {"implicit", TokenType::Implicit}, {"let", TokenType::Let},
        {"module", TokenType::Module}, {"mut", TokenType::Mut}, {"namespace", TokenType::Namespace}, {"next", TokenType::Next}, {"operator", TokenType::Operator},
        {"private", TokenType::Private}, {"public", TokenType::Public}, {"post", TokenType::ContractPost}, {"return", TokenType::Return},
        {"requires", TokenType::Requires}, {"struct", TokenType::Struct}, {"super", TokenType::Super}, {"switch", TokenType::Switch},
        {"this", TokenType::This}, {"try", TokenType::Try}, {"type", TokenType::Type}, {"union", TokenType::Union},
        {"while", TokenType::While}, {"when", TokenType::When}, {"true", TokenType::True}, {"false", TokenType::False},
        {"pre", TokenType::ContractPre}, {"post", TokenType::ContractPost}, {"assert", TokenType::ContractAssert}, {"meta", TokenType::Meta},
        {"using", TokenType::Using}, {"template", TokenType::Template},
        {"virtual", TokenType::Virtual}, {"override", TokenType::Override},
        {"inout", TokenType::Inout}, {"out", TokenType::Out}, {"move", TokenType::Move}, {"forward", TokenType::Forward},
        // Concurrency keywords (Kotlin-style)
        {"suspend", TokenType::Suspend}, {"async", TokenType::Async}, {"await", TokenType::Await},
        {"launch", TokenType::Launch}, {"coroutineScope", TokenType::CoroutineScope},
        {"channel", TokenType::Channel}, {"select", TokenType::Select}, {"parallel_for", TokenType::ParallelFor}
    };

    auto it = keywords.find(std::string(text));
    if (it != keywords.end()) return it->second;
    return TokenType::Identifier;
}

} // namespace cpp2_transpiler
