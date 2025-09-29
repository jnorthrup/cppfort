#include "pipeline.h"

#include <memory>

#include "emitter.h"
#include "lexer.h"
#include "parser.h"

namespace cppfort::stage0 {
namespace {

class DefaultLexerPass final : public LexerPass {
  public:
    LexResult run(const std::string& source, const std::string& filename) const override {
        Lexer lexer(source, filename);
        auto tokens = lexer.tokenize();
        LexResult result;
        result.tokens = std::move(tokens);
        result.source = lexer.source();
        result.filename = filename;
        return result;
    }
};

class DefaultParserPass final : public ParserPass {
  public:
    TranslationUnit run(LexResult lex_result) const override {
        Parser parser(std::move(lex_result.tokens), std::move(lex_result.source));
        return parser.parse();
    }
};

class DefaultEmitterPass final : public EmitterPass {
  public:
    std::string run(const TranslationUnit& unit, const EmitOptions& options) const override {
        Emitter emitter;
        return emitter.emit(unit, options);
    }
};

} // namespace

Pipeline make_default_pipeline() {
    Pipeline pipeline;
    pipeline.lexer = std::make_shared<DefaultLexerPass>();
    pipeline.parser = std::make_shared<DefaultParserPass>();
    pipeline.emitter = std::make_shared<DefaultEmitterPass>();
    return pipeline;
}

} // namespace cppfort::stage0
