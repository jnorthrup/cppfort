#include "pipeline.h"

#include <memory>

#include "emitter.h"
#include "parser.h"
#include "wide_scanner.h"

namespace cppfort::stage0 {
namespace {

// Scanner-based lexer pass - minimal viable implementation
class ScannerLexerPass final : public LexerPass {
  public:
    LexResult run(const ::std::string& source, const ::std::string& filename) const override {
        // Step 1: Wide scanner - get alternating anchors with SIMD
        auto anchors = cppfort::ir::WideScanner::generateAlternatingAnchors(source);
        auto boundaries = cppfort::ir::WideScanner::scanAnchorsSIMD(source, anchors);

        // Step 2: Convert boundaries into tokens
        LexResult result;
        result.source = source;
        result.filename = filename;

        // Generate tokens from boundaries
        for (const auto& boundary : boundaries) {
            Token tok;
            tok.offset = boundary.position;
            tok.length = 1;
            tok.location = SourceLocation(filename, 0, 0); // TODO: calculate line/col

            if (boundary.is_delimiter) {
                // Map delimiter to token type
                switch (boundary.delimiter) {
                    case ';': tok.type = TokenType::Semicolon; break;
                    case ',': tok.type = TokenType::Comma; break;
                    case '{': tok.type = TokenType::LBrace; break;
                    case '}': tok.type = TokenType::RBrace; break;
                    case '(': tok.type = TokenType::LParen; break;
                    case ')': tok.type = TokenType::RParen; break;
                    case '[': tok.type = TokenType::LBracket; break;
                    case ']': tok.type = TokenType::RBracket; break;
                    default: continue; // Skip unknown delimiters
                }
                tok.lexeme = ::std::string(1, boundary.delimiter);
                result.tokens.push_back(tok);
            }
        }

        // Add EOF token
        Token eof;
        eof.type = TokenType::EndOfFile;
        eof.offset = source.size();
        eof.location = SourceLocation(filename, 0, 0);
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
