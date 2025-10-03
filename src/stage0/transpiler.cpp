#include "transpiler.h"

#include <stdexcept>
#include <utility>
#include <iostream>

#include "orbit_ring.h"
#include "mmap_source.h"

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

OrbitTranslationUnit Transpiler::parseWithOrbits(const ::std::string& source, const ::std::string& filename) const {
    // For now, just return a basic orbit translation unit
    // The orbit scanner implementation is incomplete, so we'll use legacy parsing
    auto legacy_unit = parse(source, filename);

    OrbitTranslationUnit orbit_unit;

    // Convert includes
    for (const auto& inc : legacy_unit.includes) {
        OrbitIncludeDecl orbit_inc;
        orbit_inc.path = inc.path;
        orbit_inc.is_system = inc.is_system;
        orbit_inc.location = inc.location;
        orbit_unit.includes.push_back(orbit_inc);
    }

    // Convert functions
    for (const auto& func : legacy_unit.functions) {
        OrbitFunctionDecl orbit_func;
        orbit_func.name = func.name;
        orbit_func.return_type = func.return_type;
        orbit_func.location = func.location;

        // Convert parameters
        for (const auto& param : func.parameters) {
            OrbitParameter orbit_param;
            orbit_param.name = param.name;
            orbit_param.type = param.type;
            orbit_param.kind = param.kind;
            orbit_param.location = param.location;
            orbit_func.parameters.push_back(orbit_param);
        }

        // For now, create a simple expression body
        // TODO: Implement proper orbit-based body parsing
        OrbitExpressionBody body;
        body.expression = "// TODO: Implement orbit-based function body";
        body.location = func.location;
        orbit_func.body = body;

        orbit_unit.functions.push_back(orbit_func);
    }

    // Set basic metadata
    orbit_unit.source_grammar = "CPP2";  // Assume CPP2 for now

    return orbit_unit;
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
