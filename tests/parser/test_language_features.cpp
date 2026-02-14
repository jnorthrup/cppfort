#include "combinator_parser.hpp"
#include "lexer.hpp"
#include "slim_ast.hpp"
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

using namespace cpp2::ast;

// Helper to check AST nodes
bool check_ast_contains(const ParseTree &tree, NodeKind kind) {
  for (const auto &node : tree.nodes) {
    if (node.kind == kind)
      return true;
  }
  return false;
}

void test_contracts_parsing() {
  std::cout << "Testing Contracts Parsing...\n";

  // Test assert statement
  {
    std::string code = "main: () = { assert(x > 0); }";
    cpp2_transpiler::Lexer lexer(code);
    auto tokens = lexer.tokenize();
    auto tree = cpp2::parser::parse(tokens);
    if (!check_ast_contains(tree, NodeKind::AssertStatement)) {
      std::cerr << "FAIL: assert statement not parsed correctly in: " << code
                << "\n";
      std::exit(1);
    }
  }

  // Test pre/post statement
  {
    std::string code = "main: () = { pre(x > 0); post(result); }";
    cpp2_transpiler::Lexer lexer(code);
    auto tokens = lexer.tokenize();
    auto tree = cpp2::parser::parse(tokens);
    if (!check_ast_contains(tree, NodeKind::AssertStatement)) {
      std::cerr << "FAIL: pre/post statement not parsed correctly\n";
      std::exit(1);
    }
  }

  // Test pre/post in function suffix
  {
    std::string code =
        "f: (x: int) -> int pre(x > 0) post(result > 0) = { return x; }";
    cpp2_transpiler::Lexer lexer(code);
    auto tokens = lexer.tokenize();
    auto tree = cpp2::parser::parse(tokens);
    if (!check_ast_contains(tree, NodeKind::ContractClause)) {
      std::cerr
          << "FAIL: contract clause not parsed correctly in function suffix\n";
      std::exit(1);
    }
  }

  std::cout << "PASS: Contracts parsing\n";
}

void test_pattern_matching_parsing() {
  std::cout << "Testing Pattern Matching Parsing...\n";

  // Test inspect expression
  {
    std::string code = "main: () = { x := inspect(y) { is int = { return 1; } "
                       "is float = { return 2; } }; }";
    cpp2_transpiler::Lexer lexer(code);
    auto tokens = lexer.tokenize();
    auto tree = cpp2::parser::parse(tokens);

    // Debug output if fails
    if (!check_ast_contains(tree, NodeKind::InspectExpression)) {
      std::cerr << "FAIL: inspect expression not parsed correctly\n";
      std::exit(1);
    }
    if (!check_ast_contains(tree, NodeKind::IsPattern)) {
      std::cerr << "FAIL: is pattern not parsed correctly\n";
      std::exit(1);
    }
    if (!check_ast_contains(tree, NodeKind::InspectArm)) {
      std::cerr << "FAIL: inspect arm not parsed correctly\n";
      std::exit(1);
    }
  }

  std::cout << "PASS: Pattern Matching parsing\n";
}

int main() {
  test_contracts_parsing();
  test_pattern_matching_parsing();
  std::cout << "All language feature tests passed\n";
  return 0;
}
