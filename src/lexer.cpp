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

    // Scan for Cpp2-specific patterns that span multiple tokens
    // This handles patterns like:
    // - identifier: ( ... )  -> Cpp2 function declaration
    // - identifier: type     -> Cpp2 variable declaration with type annotation
    for (size_t i = 0; i < tokens.size() - 2; ++i) {
        // Pattern: identifier : ( -> Cpp2 function declaration
        if (tokens[i].type == TokenType::Identifier &&
            tokens[i + 1].type == TokenType::Colon &&
            tokens[i + 2].type == TokenType::LeftParen) {
            m_has_cpp2_syntax = true;
            break;
        }
        // Pattern: identifier : identifier/type -> Cpp2 variable declaration
        // Skip if followed by : (which would be :: namespace access)
        if (i < tokens.size() - 3 &&
            tokens[i].type == TokenType::Identifier &&
            tokens[i + 1].type == TokenType::Colon &&
            (tokens[i + 2].type == TokenType::Identifier ||
             tokens[i + 2].type == TokenType::Struct ||
             tokens[i + 2].type == TokenType::Class ||
             tokens[i + 2].type == TokenType::Union ||
             tokens[i + 2].type == TokenType::Enum) &&
            tokens[i + 2].type != TokenType::Colon) {
            m_has_cpp2_syntax = true;
            break;
        }
        // Pattern: identifier : std:: -> Cpp2 variable with qualified type
        if (i < tokens.size() - 4 &&
            tokens[i].type == TokenType::Identifier &&
            tokens[i + 1].type == TokenType::Colon &&
            tokens[i + 2].type == TokenType::Identifier &&
            tokens[i + 3].type == TokenType::DoubleColon) {
            m_has_cpp2_syntax = true;
            break;
        }
    }

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
                if (match('>')) {
                    add_token(TokenType::Spaceship);  // <=>
                } else {
                    add_token(TokenType::LessThanOrEqual);
                }
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
            } else if (peek() == '=' && peek_next() == '=') {
                // :== is Colon followed by DoubleEqual
                // Don't consume the ==, just emit the colon
                add_token(TokenType::Colon);
                m_has_cpp2_syntax = true;  // :== is Cpp2-specific
            } else if (match('=')) {
                add_token(TokenType::ColonEqual);
                m_has_cpp2_syntax = true;  // := is Cpp2-specific
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
    // Track if any Cpp2-specific syntax is found
    // These tokens don't appear in standard C++ code
    switch (type) {
        case TokenType::ColonEqual:   // := type inference
        case TokenType::At:           // @ metafunctions
        case TokenType::Inout:        // inout parameter
        case TokenType::Out:          // out parameter
        case TokenType::Move:         // move parameter
        case TokenType::Forward:      // forward parameter
        case TokenType::Copy:         // copy parameter
        case TokenType::Inspect:      // inspect expression
        case TokenType::Underscore:   // _ wildcard/discard (when used as identifier)
            m_has_cpp2_syntax = true;
            break;
        default:
            break;
    }
    tokens.emplace_back(type, lexeme, line, column - lexeme.length(), start);
}

void Lexer::scan_identifier() {
    while (is_identifier_char(peek())) {
        advance();
    }

    std::string_view text = source.substr(start, current - start);

    // Check for raw string literals: R"...", LR"...", uR"...", UR"...", u8R"..."
    if (peek() == '"' && (text == "R" || text == "LR" || text == "uR" || text == "UR" || text == "u8R")) {
        scan_raw_string();
        return;
    }

    // Check for string literal prefixes: u", U", u8", L"
    if (peek() == '"' && (text == "u" || text == "U" || text == "u8" || text == "L")) {
        scan_string_with_prefix();
        return;
    }
    
    TokenType type = identifier_type();

    // Mark Cpp2-specific keywords
    switch (type) {
        case TokenType::As:
        case TokenType::Is:
        case TokenType::In:
        case TokenType::Inspect:
        case TokenType::When:
        case TokenType::Let:
        case TokenType::Mut:
        case TokenType::Func:
        case TokenType::Type:
        case TokenType::Next:
        case TokenType::ContractPre:
        case TokenType::ContractPost:
        case TokenType::Inout:
        case TokenType::Out:
        case TokenType::Move:
        case TokenType::Forward:
        case TokenType::Copy:
        case TokenType::Suspend:
        case TokenType::Async:
        case TokenType::Await:
        case TokenType::Launch:
        case TokenType::CoroutineScope:
        case TokenType::Channel:
        case TokenType::Select:
        case TokenType::ParallelFor:
            m_has_cpp2_syntax = true;
            break;
        default:
            break;
    }

    add_token(type, text);
}

void Lexer::scan_raw_string() {
    // Called after we've seen R", LR", uR", UR", or u8R"
    // Current position is at the opening quote
    advance(); // consume opening quote
    
    // Read delimiter (if any) until we hit '('
    std::string delimiter;
    while (peek() != '(' && !is_at_end()) {
        delimiter += advance();
    }
    
    if (is_at_end()) {
        add_token(TokenType::Unknown);
        return;
    }
    
    advance(); // consume opening paren
    
    // Now read until we find ')delimiter"'
    std::string closing = ")" + delimiter + "\"";
    
    while (!is_at_end()) {
        // Check if we've found the closing sequence
        bool found_closing = true;
        for (size_t i = 0; i < closing.length() && !is_at_end(); ++i) {
            if (current + i >= source.length() || source[current + i] != closing[i]) {
                found_closing = false;
                break;
            }
        }
        
        if (found_closing) {
            // Consume the closing sequence
            for (size_t i = 0; i < closing.length(); ++i) {
                advance();
            }
            break;
        }
        
        if (peek() == '\n') {
            line++;
            column = 1;
        }
        advance();
    }
    
    add_token(TokenType::StringLiteral);
}

void Lexer::scan_number() {
    // Check for hex, binary, or octal
    bool is_hex = false;
    bool is_binary = false;
    bool is_float = false;
    
    if (peek() == '0') {
        if (peek_next() == 'x' || peek_next() == 'X') {
            is_hex = true;
            advance(); // 0
            advance(); // x
            while (is_hex_digit(peek()) || peek() == '\'') {
                advance();
            }
        } else if (peek_next() == 'b' || peek_next() == 'B') {
            is_binary = true;
            advance(); // 0
            advance(); // b
            while (peek() == '0' || peek() == '1' || peek() == '\'') {
                advance();
            }
        }
    }
    
    if (!is_hex && !is_binary) {
        // Regular decimal number with optional digit separators
        while (is_digit(peek()) || peek() == '\'') {
            advance();
        }
    }

    // Check for fractional part (not for hex/binary)
    if (!is_hex && !is_binary && peek() == '.' && is_digit(peek_next())) {
        is_float = true;
        // Consume the decimal point
        advance();

        while (is_digit(peek()) || peek() == '\'') {
            advance();
        }
    }

    // Check for exponent (not for hex/binary) - can be on int or float
    if (!is_hex && !is_binary && (peek() == 'e' || peek() == 'E')) {
        is_float = true;
        advance();
        if (peek() == '+' || peek() == '-') {
            advance();
        }
        if (!is_digit(peek())) {
            add_token(TokenType::Unknown);
            return;
        }
        while (is_digit(peek()) || peek() == '\'') {
            advance();
        }
    }
    
    if (is_float) {
        // Float suffix: f, F, l, L
        if (peek() == 'f' || peek() == 'F' || peek() == 'l' || peek() == 'L') {
            advance();
        }

        add_token(TokenType::FloatLiteral);
    } else {
        // Integer suffixes: u, U, l, L, ll, LL, ul, UL, ull, ULL, etc.
        // Also z/Z for size_t (C++23) 
        bool has_unsigned = false;
        bool has_long = false;
        bool has_longlong = false;
        bool has_size = false;
        
        // Can be in any order: u, l, ll, z (and combinations)
        for (int i = 0; i < 3; i++) {
            char c = peek();
            if ((c == 'u' || c == 'U') && !has_unsigned) {
                has_unsigned = true;
                advance();
            } else if ((c == 'l' || c == 'L') && !has_long && !has_longlong) {
                advance();
                if (peek() == 'l' || peek() == 'L') {
                    has_longlong = true;
                    advance();
                } else {
                    has_long = true;
                }
            } else if ((c == 'z' || c == 'Z') && !has_size) {
                has_size = true;
                advance();
            } else {
                break;
            }
        }
        
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

void Lexer::scan_string_with_prefix() {
    // Called after we've seen u", U", u8", or L"
    // The prefix has already been consumed, start includes it
    advance(); // Opening "

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

bool Lexer::is_hex_digit(char c) const {
    return std::isxdigit(static_cast<unsigned char>(c));
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
        {"concept", TokenType::Concept}, {"const", TokenType::Const}, {"continue", TokenType::Continue}, {"decltype", TokenType::Decltype}, {"do", TokenType::Do}, {"else", TokenType::Else},
        {"enum", TokenType::Enum}, {"explicit", TokenType::Explicit}, {"final", TokenType::Final}, {"for", TokenType::For},
        {"func", TokenType::Func}, {"if", TokenType::If}, {"import", TokenType::Import}, {"in", TokenType::In},
        {"inspect", TokenType::Inspect}, {"interface", TokenType::Interface}, {"is", TokenType::Is}, {"implicit", TokenType::Implicit}, {"let", TokenType::Let},
        {"module", TokenType::Module}, {"mut", TokenType::Mut}, {"namespace", TokenType::Namespace}, {"next", TokenType::Next}, {"operator", TokenType::Operator},
        {"private", TokenType::Private}, {"public", TokenType::Public}, {"protected", TokenType::Protected}, {"post", TokenType::ContractPost}, {"return", TokenType::Return},
        {"requires", TokenType::Requires}, {"struct", TokenType::Struct}, {"super", TokenType::Super}, {"switch", TokenType::Switch},
        {"this", TokenType::This}, {"throw", TokenType::Throw}, {"throws", TokenType::Throws}, {"try", TokenType::Try}, {"type", TokenType::Type}, {"union", TokenType::Union},
        {"while", TokenType::While}, {"when", TokenType::When}, {"true", TokenType::True}, {"false", TokenType::False},
        {"pre", TokenType::ContractPre}, {"post", TokenType::ContractPost}, {"assert", TokenType::ContractAssert}, {"meta", TokenType::Meta},
        {"using", TokenType::Using}, {"template", TokenType::Template},
        {"virtual", TokenType::Virtual}, {"override", TokenType::Override},
        {"inout", TokenType::Inout}, {"out", TokenType::Out}, {"move", TokenType::Move}, {"forward", TokenType::Forward}, {"copy", TokenType::Copy},
        {"in_ref", TokenType::InRef}, {"forward_ref", TokenType::ForwardRef},
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
