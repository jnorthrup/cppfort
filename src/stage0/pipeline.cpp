#include "pipeline.h"

#include <memory>
#include <unordered_map>
#include <string_view>
#include <cctype>
#include <algorithm>

#include "emitter.h"
#include "parser.h"
#include "wide_scanner.h"
#include "pattern_scanner.h"

namespace cppfort::stage0 {
namespace {

// CPP2 keyword lookup table
static const ::std::unordered_map<::std::string_view, TokenType> KEYWORDS = {
    {"return", TokenType::KeywordReturn},
    {"if", TokenType::KeywordIf},
    {"else", TokenType::KeywordElse},
    {"for", TokenType::KeywordFor},
    {"while", TokenType::KeywordWhile},
    {"do", TokenType::KeywordDo},
    {"next", TokenType::KeywordNext},
    {"assert", TokenType::KeywordAssert},
    {"pre", TokenType::KeywordPre},
    {"post", TokenType::KeywordPost},
    {"auto", TokenType::KeywordAuto},
    {"using", TokenType::KeywordUsing},
    {"namespace", TokenType::KeywordNamespace},
    {"type", TokenType::KeywordType},
    {"public", TokenType::KeywordPublic},
    {"protected", TokenType::KeywordProtected},
    {"private", TokenType::KeywordPrivate},
    {"virtual", TokenType::KeywordVirtual},
    {"override", TokenType::KeywordOverride},
    {"final", TokenType::KeywordFinal},
    {"in", TokenType::KeywordIn},
    {"inout", TokenType::KeywordInout},
    {"out", TokenType::KeywordOut},
    {"copy", TokenType::KeywordCopy},
    {"move", TokenType::KeywordMove},
    {"forward", TokenType::KeywordForward},
};

// Helper: Calculate source location (line/column) from byte offset
static SourceLocation calculate_location(const ::std::string& source,
                                          const ::std::string& filename,
                                          size_t offset) {
    size_t line = 1;
    size_t column = 1;
    for (size_t i = 0; i < offset && i < source.size(); ++i) {
        if (source[i] == '\n') {
            ++line;
            column = 1;
        } else {
            ++column;
        }
    }
    return SourceLocation(filename, line, column);
}

// Helper: Trim leading/trailing whitespace
static ::std::string trim(const ::std::string& str) {
    auto start = ::std::find_if_not(str.begin(), str.end(),
                                     [](unsigned char ch) { return ::std::isspace(ch); });
    auto end = ::std::find_if_not(str.rbegin(), str.rend(),
                                   [](unsigned char ch) { return ::std::isspace(ch); }).base();
    return (start < end) ? ::std::string(start, end) : ::std::string();
}

// Helper: Validate identifier characters with comprehensive checks
static bool validate_identifier_chars(const ::std::string& text, size_t& error_pos, ::std::string& error_msg) {
    using namespace cppfort::stage0;

    for (size_t i = 0; i < text.size(); ++i) {
        char ch = text[i];

        // Check for trigraph sequences (ALL 9 trigraphs) - REJECT
        if (i + 1 < text.size()) {
            TrigraphType trig = IdentifierCharEvidence::detectTrigraph(ch, text[i + 1]);
            if (trig != TrigraphType::None) {
                error_pos = i;
                error_msg = "Trigraph sequence detected and rejected: ??" + ::std::string(1, text[i + 1]);
                return false;
            }
        }

        // Validate character using enhanced evidence-based validation
        auto ev = IdentifierCharEvidence::validate(ch, i, i > 0);

        if (!ev.is_legal) {
            if (ev.is_trigraph) {
                error_pos = i;
                error_msg = ev.error_message;
                return false;
            }

            if (ev.is_unicode) {
                // UTF-8 multi-byte sequence detected - skip validation for now
                // In full implementation, validate UTF-8 XID_Start/XID_Continue
                continue;
            }

            error_pos = i;
            error_msg = ev.error_message;
            return false;
        }
    }
    return true;
}

// Helper: Validate numeric literal with prefix detection
static bool validate_numeric_literal(const ::std::string& text, size_t& error_pos, ::std::string& error_msg) {
    using namespace cppfort::stage0;

    if (text.empty()) return true;

    // Detect prefix
    NumericPrefix prefix = NumericPrefix::None;
    size_t start_pos = 0;

    if (text.size() >= 2 && text[0] == '0') {
        prefix = NumericCharEvidence::detectPrefix(text[0], text[1]);
        if (prefix == NumericPrefix::Binary || prefix == NumericPrefix::Hex) {
            start_pos = 2;  // Skip 0b or 0x
        } else if (prefix == NumericPrefix::Octal) {
            start_pos = 1;  // Skip leading 0
        }
    } else {
        prefix = NumericPrefix::Decimal;
    }

    // Validate each character
    for (size_t i = start_pos; i < text.size(); ++i) {
        char ch = text[i];

        auto ev = NumericCharEvidence::validate(ch, i, prefix);

        if (!ev.error_message.empty()) {
            error_pos = i;
            error_msg = ev.error_message;
            return false;
        }

        // Allow decimal point and sign in appropriate contexts
        if (ch == '.' || ch == '+' || ch == '-') {
            continue;  // Floating point or exponent sign
        }
    }

    return true;
}

// Helper: Classify a text segment as keyword, number, string, or identifier
static TokenType classify_segment(const ::std::string& text) {
    if (text.empty()) {
        return TokenType::Identifier;
    }

    // Check if it's a keyword
    auto it = KEYWORDS.find(text);
    if (it != KEYWORDS.end()) {
        return it->second;
    }

    // Check if it starts with a digit (number literal)
    if (::std::isdigit(static_cast<unsigned char>(text[0]))) {
        // Validate numeric literal with prefix detection
        size_t error_pos;
        ::std::string error_msg;
        if (!validate_numeric_literal(text, error_pos, error_msg)) {
            // Invalid numeric literal - could log error or throw
            // For now, still classify as Number but flag the error
        }
        return TokenType::Number;
    }

    // Check if it's a string literal
    if (text[0] == '"') {
        // TODO: Validate escape sequences within string
        return TokenType::String;
    }

    // Check if it's a char literal
    if (text[0] == '\'') {
        // TODO: Validate escape sequences within char
        return TokenType::Char;
    }

    // Validate identifier characters (comprehensive: trigraphs, unicode, etc.)
    size_t error_pos;
    ::std::string error_msg;
    if (!validate_identifier_chars(text, error_pos, error_msg)) {
        // Invalid identifier - could throw or log error
        // For now, return as identifier (error handling TBD)
        return TokenType::Identifier;
    }

    // Default: identifier
    return TokenType::Identifier;
}

} // anonymous namespace

/**
 * Scanner-based tokenization - NO LEGACY LEXER
 * This is the ONLY tokenization mechanism
 */
ScanResult Pipeline::scan(const ::std::string& source, const ::std::string& filename) {
        // Step 1: Wide scanner - get delimiter boundaries
        auto anchors = cppfort::ir::WideScanner::generateAlternatingAnchors(source);
        auto boundaries = cppfort::ir::WideScanner::scanAnchorsSIMD(source, anchors);

        // Step 2: Extract and classify text segments between boundaries
        ScanResult result;
        result.source = source;
        result.filename = filename;

        // Sort boundaries by position (should already be sorted, but ensure it)
        ::std::vector<size_t> positions;
        positions.reserve(boundaries.size());
        for (const auto& b : boundaries) {
            if (b.is_delimiter) {
                positions.push_back(b.position);
            }
        }
        ::std::sort(positions.begin(), positions.end());

        // Extract text segments between delimiters
        size_t prev_pos = 0;
        for (size_t curr_pos : positions) {
            // Extract text between prev_pos and curr_pos
            if (curr_pos > prev_pos) {
                ::std::string segment = source.substr(prev_pos, curr_pos - prev_pos);
                ::std::string trimmed = trim(segment);

                if (!trimmed.empty()) {
                    // Create token for the segment
                    Token tok;
                    tok.type = classify_segment(trimmed);
                    tok.lexeme = trimmed;
                    tok.offset = prev_pos;
                    tok.length = curr_pos - prev_pos;
                    tok.location = calculate_location(source, filename, prev_pos);
                    result.tokens.push_back(tok);
                }
            }

            // Create token for the delimiter itself
            if (curr_pos < source.size()) {
                char delim = source[curr_pos];
                Token tok;
                tok.offset = curr_pos;
                tok.length = 1;
                tok.lexeme = ::std::string(1, delim);
                tok.location = calculate_location(source, filename, curr_pos);

                // Map delimiter to token type
                switch (delim) {
                    case ';': tok.type = TokenType::Semicolon; break;
                    case ',': tok.type = TokenType::Comma; break;
                    case '{': tok.type = TokenType::LBrace; break;
                    case '}': tok.type = TokenType::RBrace; break;
                    case '(': tok.type = TokenType::LParen; break;
                    case ')': tok.type = TokenType::RParen; break;
                    case '[': tok.type = TokenType::LBracket; break;
                    case ']': tok.type = TokenType::RBracket; break;
                    case ':': tok.type = TokenType::Colon; break;
                    case '=': tok.type = TokenType::Equals; break;
                    case '<': tok.type = TokenType::Less; break;
                    case '>': tok.type = TokenType::Greater; break;
                    case '*': tok.type = TokenType::Star; break;
                    case '+': tok.type = TokenType::Plus; break;
                    case '-': tok.type = TokenType::Minus; break;
                    case '/': tok.type = TokenType::Slash; break;
                    case '&': tok.type = TokenType::Ampersand; break;
                    case '|': tok.type = TokenType::Pipe; break;
                    case '^': tok.type = TokenType::Caret; break;
                    case '!': tok.type = TokenType::Exclamation; break;
                    case '?': tok.type = TokenType::Question; break;
                    case '.': tok.type = TokenType::Dot; break;
                    case '%': tok.type = TokenType::Percent; break;
                    default:
                        // Skip unknown delimiters
                        prev_pos = curr_pos + 1;
                        continue;
                }

                result.tokens.push_back(tok);
            }

            prev_pos = curr_pos + 1;
        }

        // Handle any remaining text after the last delimiter
        if (prev_pos < source.size()) {
            ::std::string segment = source.substr(prev_pos);
            ::std::string trimmed = trim(segment);

            if (!trimmed.empty()) {
                Token tok;
                tok.type = classify_segment(trimmed);
                tok.lexeme = trimmed;
                tok.offset = prev_pos;
                tok.length = source.size() - prev_pos;
                tok.location = calculate_location(source, filename, prev_pos);
                result.tokens.push_back(tok);
            }
        }

        // Add EOF token
        Token eof;
        eof.type = TokenType::EndOfFile;
        eof.offset = source.size();
        eof.length = 0;
        eof.location = calculate_location(source, filename, source.size());
        result.tokens.push_back(eof);

        return result;
}

/**
 * Parse scanner tokens into AST
 */
TranslationUnit Pipeline::parse(ScanResult scan_result) {
    Parser parser(::std::move(scan_result.tokens), ::std::move(scan_result.source));
    return parser.parse();
}

/**
 * Emit C++ from AST
 */
::std::string Pipeline::emit(const TranslationUnit& unit, const EmitOptions& options) {
    Emitter emitter;
    return emitter.emit(unit, options);
}

/**
 * Full pipeline: scan -> parse -> emit
 */
::std::string Pipeline::transpile(const ::std::string& source,
                                 const ::std::string& filename,
                                 const EmitOptions& options) {
    auto scan_result = scan(source, filename);
    auto ast = parse(::std::move(scan_result));
    return emit(ast, options);
}

} // namespace cppfort::stage0
