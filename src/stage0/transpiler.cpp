#include "transpiler.h"

#include <stdexcept>
#include <utility>
#include <iostream>

#include "orbit_scanner.h"
#include "orbit_context.h"
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
    // Step 1: Run orbit scanner with Rabin-Karp multi-grammar detection
    OrbitScanner scanner;
    auto detection_result = scanner.scan(source, filename);
    
    // Step 2: Create orbit context and analyze matches
    OrbitContext context(source, filename);
    for (const auto& match : detection_result.matches) {
        context.addMatch(match);
    }
    
    // Step 3: Determine best grammar based on confidence scores
    auto best_grammar = detection_result.detected_grammar;
    if (best_grammar.empty()) {
        best_grammar = scanner.determineBestGrammar(detection_result.matches);
    }
    
    // Step 4: Parse with legacy parser to get base AST
    auto legacy_unit = parse(source, filename);

    // Step 5: Build OrbitTranslationUnit with ring candidates
    OrbitTranslationUnit orbit_unit;
    orbit_unit.source_grammar = best_grammar;
    orbit_unit.detection_confidence = detection_result.confidence;
    orbit_unit.detection_result = detection_result;
    
    // Create orbit rings for each match
    for (const auto& match : detection_result.matches) {
        OrbitRing ring;
        ring.pattern_name = match.pattern_name;
        ring.confidence = match.confidence;
        ring.start_offset = match.start;
        ring.end_offset = match.end;
        ring.grammar = match.grammar;
        
        // Create candidates for this ring (winner vs alternatives)
        OrbitCandidate winner;
        winner.grammar = best_grammar;
        winner.confidence = match.confidence;
        winner.is_selected = true;
        ring.candidates.push_back(winner);
        
        // Add alternative grammar candidates if multi-grammar detected
        for (const auto& alt_match : detection_result.matches) {
            if (alt_match.grammar != best_grammar && 
                alt_match.start == match.start && 
                alt_match.end == match.end) {
                OrbitCandidate alt_candidate;
                alt_candidate.grammar = alt_match.grammar;
                alt_candidate.confidence = alt_match.confidence;
                alt_candidate.is_selected = false;
                ring.candidates.push_back(alt_candidate);
            }
        }
        
        orbit_unit.orbit_rings.push_back(ring);
    }

    // Step 6: Convert includes with orbit metadata
    for (const auto& inc : legacy_unit.includes) {
        OrbitIncludeDecl orbit_inc;
        orbit_inc.path = inc.path;
        orbit_inc.is_system = inc.is_system;
        orbit_inc.location = inc.location;
        orbit_unit.includes.push_back(orbit_inc);
    }

    // Step 7: Convert functions with orbit ring assignments
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

        // Assign orbit rings based on location overlap
        for (const auto& ring : orbit_unit.orbit_rings) {
            if (func.location.line >= ring.start_offset && 
                func.location.line <= ring.end_offset) {
                orbit_func.associated_rings.push_back(&ring);
            }
        }
        
        orbit_unit.functions.push_back(orbit_func);
    }

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
