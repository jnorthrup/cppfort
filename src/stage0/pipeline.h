#pragma once

#include <string>
#include <vector>

#include "ast.h"
#include "emitter.h"
#include "token.h"

namespace cppfort::stage0 {

/**
 * Scanner-based tokenization result
 * No more legacy lexer - scanner is the only tokenization mechanism
 */
struct ScanResult {
    ::std::vector<Token> tokens;
    ::std::string source;
    ::std::string filename;
};

/**
 * Simplified pipeline - no virtual dispatch overhead
 * Scanner is directly integrated (no abstraction layer)
 */
struct Pipeline {
    /**
     * Tokenize source using WideScanner + pattern detection
     */
    static ScanResult scan(const ::std::string& source, const ::std::string& filename);

    /**
     * Parse tokens into AST
     */
    static TranslationUnit parse(ScanResult scan_result);

    /**
     * Emit C++ from AST
     */
    static ::std::string emit(const TranslationUnit& unit, const EmitOptions& options);

    /**
     * Full pipeline: scan -> parse -> emit
     */
    static ::std::string transpile(const ::std::string& source,
                                   const ::std::string& filename,
                                   const EmitOptions& options);
};

} // namespace cppfort::stage0
