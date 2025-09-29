#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ast.h"
#include "emitter.h"
#include "token.h"

namespace cppfort::stage0 {

struct LexResult {
    std::vector<Token> tokens;
    std::string source;
    std::string filename;
};

class LexerPass {
  public:
    virtual ~LexerPass() = default;
    virtual LexResult run(const std::string& source, const std::string& filename) const = 0;
};

class ParserPass {
  public:
    virtual ~ParserPass() = default;
    virtual TranslationUnit run(LexResult lex_result) const = 0;
};

class EmitterPass {
  public:
    virtual ~EmitterPass() = default;
    virtual std::string run(const TranslationUnit& unit, const EmitOptions& options) const = 0;
};

struct Pipeline {
    std::shared_ptr<LexerPass> lexer;
    std::shared_ptr<ParserPass> parser;
    std::shared_ptr<EmitterPass> emitter;
};

Pipeline make_default_pipeline();

} // namespace cppfort::stage0
