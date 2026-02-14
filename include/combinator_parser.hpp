#pragma once
// ============================================================================
// Combinator Parser - Public API
// ============================================================================
// Spirit-like combinator parser that produces slim ParseTree.
// No legacy AST - this is the only parser interface.
// ============================================================================

#include "slim_ast.hpp"
#include "core/tokens.hpp"
#include <span>

namespace cpp2::parser {

// Parse tokens into ParseTree using combinator grammar
// Defined in src/parser.cpp
cpp2::ast::ParseTree parse(std::span<const cpp2_transpiler::Token> tokens);

} // namespace cpp2::parser
