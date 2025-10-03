#include "pipeline.h"

#include <memory>

#include "emitter.h"
#include "parser.h"
#include "pattern_scanner.h"
#include "wide_scanner.h"
#include "../orbit_scanner.h"

namespace cppfort::stage0 {
namespace {

// Scanner-based lexer pass with speculative lookahead
class ScannerLexerPass final : public LexerPass {
  private:
    mutable PatternScanner m_pattern_scanner;

  public:
    LexResult run(const ::std::string& source, const ::std::string& filename) const override {
        // Step 1: Wide scanner - get alternating anchors with SIMD
        auto anchors = cppfort::ir::WideScanner::generateAlternatingAnchors(source);
        auto boundaries = cppfort::ir::WideScanner::scanAnchorsSIMD(source, anchors);

        // Step 2: Pattern scanner - detect grammar and patterns speculatively
        auto pattern_results = m_pattern_scanner.scanWindowed(source);

        // Step 3: Convert boundaries and patterns into tokens
        LexResult result;
        result.source = source;
        result.filename = filename;

        // Generate tokens from boundaries (speculative tokenization)
        for (const auto& boundary : boundaries) {
            Token tok;
            tok.location.offset = boundary.position;
            tok.location.filename = filename;

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
                    default: tok.type = TokenType::Unknown; break;
                }
                tok.value = ::std::string(1, boundary.delimiter);
            } else {
                tok.type = TokenType::Unknown;
            }

            result.tokens.push_back(tok);
        }

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
    pipeline.lexer = ::std::make_shared<DefaultLexerPass>();
    pipeline.parser = ::std::make_shared<DefaultParserPass>();
    pipeline.emitter = ::std::make_shared<DefaultEmitterPass>();
    return pipeline;
}

} // namespace cppfort::stage0
