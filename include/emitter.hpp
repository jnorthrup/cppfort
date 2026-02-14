#pragma once

#include <string>
#include <span>
#include "slim_ast.hpp"

// Generates C++ code from a parsed Cpp2 AST
std::string generate_from_tree(const cpp2::ast::ParseTree& tree,
                               std::span<const cpp2_transpiler::Token> tokens);
