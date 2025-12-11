#include "lexer.hpp"
#include <unordered_map>
#include <cassert>
#include <cctype>

namespace cpp2_transpiler {

Lexer::Lexer(std::string_view source)
    : source(source), current(0), line(1), column(1), start(0) {}

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
            } else {
                add_token(TokenType::Minus);
            }
            break;
        }
        case '+': {
            if (match('+')) {
                add_token(TokenType::PlusPlus);
            } else {
                add_token(TokenType::Plus);
            }
            break;
        }
        case '*': {
            if (match('*')) {
                add_token(TokenType::Exponentiation);
            } else {
                add_token(TokenType::Asterisk);
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
                    add_token(TokenType::LeftShiftAssign);
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
                    add_token(TokenType::RightShiftAssign);
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
            } else {
                add_token(TokenType::Ampersand);
            }
            break;
        }
        case '|': {
            if (match('|')) {
                add_token(TokenType::DoublePipe);
            } else {
                add_token(TokenType::Pipe);
            }
            break;
        }
        case '^': add_token(TokenType::Caret); break;
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
            } else {
                add_token(TokenType::Colon);
            }
            break;
        }
        case '$': add_token(TokenType::Dollar); break;
        case '@': add_token(TokenType::At); break;
        case '_': add_token(TokenType::Underscore); break;

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
                add_token(TokenType::SlashAssign);
            } else {
                add_token(TokenType::Slash);
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

void Lexer::scan_block_comment() {
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
    switch (source[start]) {
        case 'a':
            return check_keyword(1, 4, "s", TokenType::As);
        case 'b':
            return check_keyword(1, 4, "ase", TokenType::Base);
        case 'c':
            if (current - start > 1) {
                switch (source[start + 1]) {
                    case 'a':
                        return check_keyword(2, 3, "se", TokenType::Case);
                    case 'l':
                        return check_keyword(2, 3, "ass", TokenType::Class);
                    case 'o':
                        if (current - start > 2) {
                            switch (source[start + 2]) {
                                case 'n':
                                    return check_keyword(3, 4, "cept", TokenType::Concept);
                                case 'n':
                                    return check_keyword(3, 4, "st", TokenType::Const);
                            }
                        }
                        break;
                }
            }
            break;
        case 'd':
            return check_keyword(1, 3, "o", TokenType::Do);
        case 'e':
            if (current - start > 1) {
                switch (source[start + 1]) {
                    case 'l':
                        return check_keyword(2, 2, "se", TokenType::Else);
                    case 'n':
                        return check_keyword(2, 2, "um", TokenType::Enum);
                    case 'x':
                        return check_keyword(2, 6, "plicit", TokenType::Explicit);
                }
            }
            break;
        case 'f':
            if (current - start > 1) {
                switch (source[start + 1]) {
                    case 'a':
                        return check_keyword(2, 3, "lse", TokenType::BooleanLiteral);
                    case 'i':
                        return check_keyword(2, 3, "nal", TokenType::Final);
                    case 'o':
                        return check_keyword(2, 1, "r", TokenType::For);
                    case 'u':
                        return check_keyword(2, 2, "nc", TokenType::Func);
                }
            }
            break;
        case 'i':
            if (current - start > 1) {
                switch (source[start + 1]) {
                    case 'f':
                        return check_keyword(2, 0, "", TokenType::If);
                    case 'm':
                        return check_keyword(2, 4, "port", TokenType::Import);
                    case 'n':
                        return check_keyword(2, 0, "", TokenType::In);
                    case 'n':
                        return check_keyword(2, 7, "terface", TokenType::Interface);
                    case 's':
                        return check_keyword(2, 0, "", TokenType::Is);
                    case 'm':
                        return check_keyword(2, 6, "plicit", TokenType::Implicit);
                }
            }
            break;
        case 'l':
            return check_keyword(1, 2, "et", TokenType::Let);
        case 'm':
            if (current - start > 1) {
                switch (source[start + 1]) {
                    case 'o':
                        return check_keyword(2, 4, "dule", TokenType::Module);
                    case 'u':
                        return check_keyword(2, 1, "t", TokenType::Mut);
                }
            }
            break;
        case 'n':
            return check_keyword(1, 7, "amespace", TokenType::Namespace);
        case 'o':
            return check_keyword(1, 6, "perator", TokenType::Operator);
        case 'p':
            if (current - start > 1) {
                switch (source[start + 1]) {
                    case 'r':
                        return check_keyword(2, 4, "ivate", TokenType::Private);
                    case 'u':
                        return check_keyword(2, 5, "blic", TokenType::Public);
                    case 'o':
                        return check_keyword(2, 2, "st", TokenType::ContractPost);
                }
            }
            break;
        case 'r':
            if (current - start > 1) {
                switch (source[start + 1]) {
                    case 'e':
                        return check_keyword(2, 5, "quire", TokenType::Requires);
                    case 'e':
                        return check_keyword(2, 5, "turn", TokenType::Return);
                }
            }
            break;
        case 's':
            if (current - start > 1) {
                switch (source[start + 1]) {
                    case 't':
                        return check_keyword(2, 4, "ruct", TokenType::Struct);
                    case 'u':
                        return check_keyword(2, 3, "per", TokenType::Super);
                    case 'w':
                        return check_keyword(2, 4, "itch", TokenType::Switch);
                }
            }
            break;
        case 't':
            if (current - start > 1) {
                switch (source[start + 1]) {
                    case 'h':
                        return check_keyword(2, 2, "is", TokenType::This);
                    case 'r':
                        if (current - start > 2) {
                            switch (source[start + 2]) {
                                case 'u':
                                    return check_keyword(3, 1, "e", TokenType::True);
                                case 'y':
                                    return check_keyword(3, 0, "", TokenType::Try);
                            }
                        }
                        break;
                    case 'y':
                        return check_keyword(2, 2, "pe", TokenType::Type);
                }
            }
            break;
        case 'u':
            return check_keyword(1, 5, "nion", TokenType::Union);
        case 'w':
            if (current - start > 1) {
                switch (source[start + 1]) {
                    case 'h':
                        return check_keyword(2, 3, "ile", TokenType::While);
                    case 'h':
                        return check_keyword(2, 3, "en", TokenType::When);
                }
            }
            break;
    }
    return TokenType::Identifier;
}

} // namespace cpp2_transpiler