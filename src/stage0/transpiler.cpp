#include "transpiler.h"

#include <stdexcept>
#include <utility>

namespace cppfort::stage0 {

Transpiler::Transpiler()
    : Transpiler(make_default_pipeline()) {}

Transpiler::Transpiler(Pipeline pipeline)
    : m_pipeline(::std::move(pipeline)) {
    if (!m_pipeline.lexer) {
        throw ::std::invalid_argument("Transpiler pipeline requires a lexer pass");
    }
    if (!m_pipeline.parser) {
        throw ::std::invalid_argument("Transpiler pipeline requires a parser pass");
    }
}

TranslationUnit Transpiler::parse(const ::std::string& source, const ::std::string& filename) const {
    auto lex_result = m_pipeline.lexer->run(source, filename);
    return m_pipeline.parser->run(::std::move(lex_result));
}

::std::string Transpiler::transpile(const ::std::string& source, const ::std::string& filename,
    const TranspileOptions& options) const {
    auto unit = parse(source, filename);
    if (!m_pipeline.emitter) {
        throw ::std::runtime_error("Transpiler pipeline requires an emitter pass to produce output");
    }
    return m_pipeline.emitter->run(unit, options.emit_options);
}

} // namespace cppfort::stage0
