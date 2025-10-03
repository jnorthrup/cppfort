#include "pipeline.h"

#include <memory>
#include <unordered_map>
#include <string_view>
#include <cctype>
#include <algorithm>

#include "emitter.h"
#include "parser.h"
#include "wide_scanner.h"

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
        return TokenType::Number;
    }

    // Check if it's a string literal
    if (text[0] == '"') {
        return TokenType::String;
    }

    // Check if it's a char literal
    if (text[0] == '\'') {
        return TokenType::Char;
    }

    // Default: identifier
    return TokenType::Identifier;
}

// Scanner-based lexer pass - complete tokenizer implementation
class ScannerLexerPass final : public LexerPass {
  public:
    LexResult run(const ::std::string& source, const ::std::string& filename) const override {
        // Step 1: Wide scanner - get delimiter boundaries
        auto anchors = cppfort::ir::WideScanner::generateAlternatingAnchors(source);
        auto boundaries = cppfort::ir::WideScanner::scanAnchorsSIMD(source, anchors);

        // Step 2: Extract and classify text segments between boundaries
        LexResult result;
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
};

class DefaultParserPass final : public ParserPass {
  public:
    TranslationUnit run(LexResult lex_result) const override {
        Parser parser(::std::move(lex_result.tokens), ::std::move(lex_result.source));
        return parser.parse();
    }
};

class DefaultEmitterPass final : public EmitterPass {
  public:
    ::std::string run(const TranslationUnit& unit, const EmitOptions& options) const override {
        Emitter emitter;
        return emitter.emit(unit, options);
    }
};

} // namespace

Pipeline make_default_pipeline() {
    Pipeline pipeline;
    pipeline.lexer = ::std::make_shared<ScannerLexerPass>();
    pipeline.parser = ::std::make_shared<DefaultParserPass>();
    pipeline.emitter = ::std::make_shared<DefaultEmitterPass>();
    return pipeline;
}

} // namespace cppfort::stage0
