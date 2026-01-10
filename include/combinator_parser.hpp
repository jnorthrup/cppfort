#pragma once
// ============================================================================
// Combinator Parser - Public API
// ============================================================================
// Spirit-like combinator parser that produces slim ParseTree.
// Include this header to use the new parser.
// ============================================================================

#include "slim_ast.hpp"
#include "parse_tree_converter.hpp"

// Include the implementation (all inline, header-only)
#include "../src/parser.cpp"

namespace cpp2::parser {

// Parse tokens into ParseTree using combinator grammar
// Note: parse() is defined in src/parser.cpp

// Parse and convert to legacy AST in one step
inline std::unique_ptr<cpp2_transpiler::AST> parse_to_ast(
        std::span<const cpp2_transpiler::Token> tokens) {
    auto tree = parse(tokens);
    return convert_to_ast(tree);
}

} // namespace cpp2::parser
