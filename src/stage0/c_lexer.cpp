#include "c_parser.h"
#include <cctype>
#include <unordered_map>

namespace cppfort::c {

// C keyword mapping
static const std::unordered_map<std::string, CTokenType> keywords = {
    {"auto", CTokenType::AUTO}, {"break", CTokenType::BREAK}, {"case", CTokenType::CASE},
    {"char", CTokenType::CHAR}, {"const", CTokenType::CONST}, {"continue", CTokenType::CONTINUE},
    {"default", CTokenType::DEFAULT}, {"do", CTokenType::DO}, {"double", CTokenType::DOUBLE},
    {"else", CTokenType::ELSE}, {"enum", CTokenType::ENUM}, {"extern", CTokenType::EXTERN},
    {"float", CTokenType::FLOAT}, {"for", CTokenType::FOR}, {"goto", CTokenType::GOTO},
    {"if", CTokenType::IF}, {"inline", CTokenType::INLINE}, {"int", CTokenType::INT},
    {"long", CTokenType::LONG}, {"register", CTokenType::REGISTER}, {"restrict", CTokenType::RESTRICT},
    {"return", CTokenType::RETURN}, {"short", CTokenType::SHORT}, {"signed", CTokenType::SIGNED},
    {"sizeof", CTokenType::SIZEOF}, {"static", CTokenType::STATIC}, {"struct", CTokenType::STRUCT},
    {"switch", CTokenType::SWITCH}, {"typedef", CTokenType::TYPEDEF}, {"union", CTokenType::UNION},
    {"unsigned", CTokenType::UNSIGNED}, {"void", CTokenType::VOID}, {"volatile", CTokenType::VOLATILE},
    {"while", CTokenType::WHILE},
    {"_Alignas", CTokenType::_ALIGNAS}, {"_Alignof", CTokenType::_ALIGNOF},
    {"_Atomic", CTokenType::_ATOMIC}, {"_Bool", CTokenType::_BOOL},
    {"_Complex", CTokenType::_COMPLEX}, {"_Generic", CTokenType::_GENERIC},
    {"_Imaginary", CTokenType::_IMAGINARY}, {"_Noreturn", CTokenType::_NORETURN},
    {"_Static_assert", CTokenType::_STATIC_ASSERT}, {"_Thread_local", CTokenType::_THREAD_LOCAL},
    {"defined", CTokenType::DEFINED}
};

char CLexer::peek() const {
    if (_pos >= _source.length()) return '\0';
    return _source[_pos];
}

char CLexer::advance() {
    if (_pos >= _source.length()) return '\0';
    char ch = _source[_pos++];
    if (ch == '\n') {
        _line++;
        _column = 1;
    } else {
        _column++;
    }
    return ch;
}

void CLexer::skipWhitespace() {
    while (std::isspace(peek())) {
        advance();
    }
}

void CLexer::skipLineComment() {
    // Skip "//"
    advance();
    advance();
    while (peek() != '\n' && peek() != '\0') {
        advance();
    }
}

void CLexer::skipBlockComment() {
    // Skip "/*"
    advance();
    advance();
    while (true) {
        if (peek() == '\0') break;
        if (peek() == '*' && _source[_pos + 1] == '/') {
            advance(); // *
            advance(); // /
            break;
        }
        advance();
    }
}

CToken CLexer::lexIdentifierOrKeyword() {
    int startLine = _line;
    int startColumn = _column;
    std::string text;

    while (std::isalnum(peek()) || peek() == '_') {
        text += advance();
    }

    auto it = keywords.find(text);
    CTokenType type = (it != keywords.end()) ? it->second : CTokenType::IDENTIFIER;

    return CToken(type, text, _filename, startLine, startColumn);
}

CToken CLexer::lexNumber() {
    int startLine = _line;
    int startColumn = _column;
    std::string text;
    CTokenType type = CTokenType::INTEGER_LITERAL;

    // Handle hex/octal prefixes
    if (peek() == '0') {
        text += advance();
        if (peek() == 'x' || peek() == 'X') {
            // Hexadecimal
            text += advance();
            while (std::isxdigit(peek())) {
                text += advance();
            }
        } else if (std::isdigit(peek())) {
            // Octal
            while (peek() >= '0' && peek() <= '7') {
                text += advance();
            }
        }
    } else {
        // Decimal
        while (std::isdigit(peek())) {
            text += advance();
        }
    }

    // Check for float suffix or decimal point
    if (peek() == '.' || peek() == 'e' || peek() == 'E') {
        type = CTokenType::FLOAT_LITERAL;
        if (peek() == '.') {
            text += advance();
            while (std::isdigit(peek())) {
                text += advance();
            }
        }
        if (peek() == 'e' || peek() == 'E') {
            text += advance();
            if (peek() == '+' || peek() == '-') {
                text += advance();
            }
            while (std::isdigit(peek())) {
                text += advance();
            }
        }
    }

    // Integer/float suffixes (u, l, ll, f, L)
    while (peek() == 'u' || peek() == 'U' || peek() == 'l' || peek() == 'L' || peek() == 'f' || peek() == 'F') {
        text += advance();
    }

    return CToken(type, text, _filename, startLine, startColumn);
}

CToken CLexer::lexString() {
    int startLine = _line;
    int startColumn = _column;
    std::string text;

    advance(); // Skip opening "

    while (peek() != '"' && peek() != '\0') {
        if (peek() == '\\') {
            text += advance(); // backslash
            if (peek() != '\0') {
                text += advance(); // escaped character
            }
        } else {
            text += advance();
        }
    }

    if (peek() == '"') {
        advance(); // Skip closing "
    }

    return CToken(CTokenType::STRING_LITERAL, text, _filename, startLine, startColumn);
}

CToken CLexer::lexChar() {
    int startLine = _line;
    int startColumn = _column;
    std::string text;

    advance(); // Skip opening '

    if (peek() == '\\') {
        text += advance(); // backslash
        if (peek() != '\0') {
            text += advance(); // escaped character
        }
    } else if (peek() != '\'') {
        text += advance();
    }

    if (peek() == '\'') {
        advance(); // Skip closing '
    }

    return CToken(CTokenType::CHAR_LITERAL, text, _filename, startLine, startColumn);
}

CToken CLexer::lexOperator() {
    int startLine = _line;
    int startColumn = _column;
    char ch = peek();

    switch (ch) {
        case '+':
            advance();
            if (peek() == '+') { advance(); return CToken(CTokenType::INCREMENT, "++", _filename, startLine, startColumn); }
            if (peek() == '=') { advance(); return CToken(CTokenType::PLUS_ASSIGN, "+=", _filename, startLine, startColumn); }
            return CToken(CTokenType::PLUS, "+", _filename, startLine, startColumn);

        case '-':
            advance();
            if (peek() == '-') { advance(); return CToken(CTokenType::DECREMENT, "--", _filename, startLine, startColumn); }
            if (peek() == '=') { advance(); return CToken(CTokenType::MINUS_ASSIGN, "-=", _filename, startLine, startColumn); }
            if (peek() == '>') { advance(); return CToken(CTokenType::ARROW, "->", _filename, startLine, startColumn); }
            return CToken(CTokenType::MINUS, "-", _filename, startLine, startColumn);

        case '*':
            advance();
            if (peek() == '=') { advance(); return CToken(CTokenType::STAR_ASSIGN, "*=", _filename, startLine, startColumn); }
            return CToken(CTokenType::STAR, "*", _filename, startLine, startColumn);

        case '/':
            advance();
            if (peek() == '=') { advance(); return CToken(CTokenType::SLASH_ASSIGN, "/=", _filename, startLine, startColumn); }
            return CToken(CTokenType::SLASH, "/", _filename, startLine, startColumn);

        case '%':
            advance();
            if (peek() == '=') { advance(); return CToken(CTokenType::PERCENT_ASSIGN, "%=", _filename, startLine, startColumn); }
            return CToken(CTokenType::PERCENT, "%", _filename, startLine, startColumn);

        case '&':
            advance();
            if (peek() == '&') { advance(); return CToken(CTokenType::AND, "&&", _filename, startLine, startColumn); }
            if (peek() == '=') { advance(); return CToken(CTokenType::AMPERSAND_ASSIGN, "&=", _filename, startLine, startColumn); }
            return CToken(CTokenType::AMPERSAND, "&", _filename, startLine, startColumn);

        case '|':
            advance();
            if (peek() == '|') { advance(); return CToken(CTokenType::OR, "||", _filename, startLine, startColumn); }
            if (peek() == '=') { advance(); return CToken(CTokenType::PIPE_ASSIGN, "|=", _filename, startLine, startColumn); }
            return CToken(CTokenType::PIPE, "|", _filename, startLine, startColumn);

        case '^':
            advance();
            if (peek() == '=') { advance(); return CToken(CTokenType::CARET_ASSIGN, "^=", _filename, startLine, startColumn); }
            return CToken(CTokenType::CARET, "^", _filename, startLine, startColumn);

        case '<':
            advance();
            if (peek() == '<') {
                advance();
                if (peek() == '=') { advance(); return CToken(CTokenType::LSHIFT_ASSIGN, "<<=", _filename, startLine, startColumn); }
                return CToken(CTokenType::LSHIFT, "<<", _filename, startLine, startColumn);
            }
            if (peek() == '=') { advance(); return CToken(CTokenType::LE, "<=", _filename, startLine, startColumn); }
            return CToken(CTokenType::LT, "<", _filename, startLine, startColumn);

        case '>':
            advance();
            if (peek() == '>') {
                advance();
                if (peek() == '=') { advance(); return CToken(CTokenType::RSHIFT_ASSIGN, ">>=", _filename, startLine, startColumn); }
                return CToken(CTokenType::RSHIFT, ">>", _filename, startLine, startColumn);
            }
            if (peek() == '=') { advance(); return CToken(CTokenType::GE, ">=", _filename, startLine, startColumn); }
            return CToken(CTokenType::GT, ">", _filename, startLine, startColumn);

        case '=':
            advance();
            if (peek() == '=') { advance(); return CToken(CTokenType::EQ, "==", _filename, startLine, startColumn); }
            return CToken(CTokenType::ASSIGN, "=", _filename, startLine, startColumn);

        case '!':
            advance();
            if (peek() == '=') { advance(); return CToken(CTokenType::NE, "!=", _filename, startLine, startColumn); }
            return CToken(CTokenType::EXCLAIM, "!", _filename, startLine, startColumn);

        case '~': advance(); return CToken(CTokenType::TILDE, "~", _filename, startLine, startColumn);
        case '?': advance(); return CToken(CTokenType::QUESTION, "?", _filename, startLine, startColumn);
        case ':': advance(); return CToken(CTokenType::COLON, ":", _filename, startLine, startColumn);
        case ';': advance(); return CToken(CTokenType::SEMICOLON, ";", _filename, startLine, startColumn);
        case ',': advance(); return CToken(CTokenType::COMMA, ",", _filename, startLine, startColumn);
        case '.': advance(); return CToken(CTokenType::DOT, ".", _filename, startLine, startColumn);
        case '{': advance(); return CToken(CTokenType::LBRACE, "{", _filename, startLine, startColumn);
        case '}': advance(); return CToken(CTokenType::RBRACE, "}", _filename, startLine, startColumn);
        case '(': advance(); return CToken(CTokenType::LPAREN, "(", _filename, startLine, startColumn);
        case ')': advance(); return CToken(CTokenType::RPAREN, ")", _filename, startLine, startColumn);
        case '[': advance(); return CToken(CTokenType::LBRACKET, "[", _filename, startLine, startColumn);
        case ']': advance(); return CToken(CTokenType::RBRACKET, "]", _filename, startLine, startColumn);

        case '#':
            advance();
            if (peek() == '#') { advance(); return CToken(CTokenType::HASH_HASH, "##", _filename, startLine, startColumn); }
            return CToken(CTokenType::HASH, "#", _filename, startLine, startColumn);

        default:
            advance();
            return CToken(CTokenType::INVALID, std::string(1, ch), _filename, startLine, startColumn);
    }
}

std::vector<CToken> CLexer::tokenize() {
    std::vector<CToken> tokens;

    while (peek() != '\0') {
        skipWhitespace();

        if (peek() == '\0') break;

        // Comments
        if (peek() == '/' && _source[_pos + 1] == '/') {
            skipLineComment();
            continue;
        }
        if (peek() == '/' && _source[_pos + 1] == '*') {
            skipBlockComment();
            continue;
        }

        // Identifiers and keywords
        if (std::isalpha(peek()) || peek() == '_') {
            tokens.push_back(lexIdentifierOrKeyword());
            continue;
        }

        // Numbers
        if (std::isdigit(peek())) {
            tokens.push_back(lexNumber());
            continue;
        }

        // String literals
        if (peek() == '"') {
            tokens.push_back(lexString());
            continue;
        }

        // Character literals
        if (peek() == '\'') {
            tokens.push_back(lexChar());
            continue;
        }

        // Operators and punctuation
        tokens.push_back(lexOperator());
    }

    tokens.push_back(CToken(CTokenType::EOF_TOKEN, "", _filename, _line, _column));
    return tokens;
}

} // namespace cppfort::c
