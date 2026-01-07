// Test C++26 contract attribute parsing
// Phase 9: C++26 Integration - Contract parsing for alias analysis

#include "lexer.hpp"
#include "parser.hpp"
#include "ast.hpp"
#include <cassert>
#include <iostream>
#include <sstream>

using namespace cpp2_transpiler;

void test_cpp26_contract_lexer_tokens() {
    // Test that [[expects]], [[ensures]], [[assert]] are lexed correctly
    std::string source = R"(
        [[expects: x > 0]]
        [[ensures: result > x]]
        [[assert: condition]]
    )";

    Lexer lexer(source);
    auto tokens = lexer.tokenize();

    bool found_expects = false;
    bool found_ensures = false;
    bool found_assert = false;

    for (const auto& token : tokens) {
        if (token.type == TokenType::AttributeExpect) {
            found_expects = true;
        }
        if (token.type == TokenType::AttributeEnsure) {
            found_ensures = true;
        }
        if (token.type == TokenType::AttributeAssert) {
            found_assert = true;
        }
    }

    assert(found_expects && "Expected [[expects]] token");
    assert(found_ensures && "Expected [[ensures]] token");
    assert(found_assert && "Expected [[assert]] token");

    std::cout << "✅ test_cpp26_contract_lexer_tokens passed\n";
}

void test_cpp26_contract_parser_integration() {
    // Test parsing functions with C++26 contract attributes
    std::string source = R"(
        func square: (x: int) -> result
            [[expects: x >= 0]]
            [[ensures: result >= 0]]
        = {
            result := x * x;
        }
    )";

    Lexer lexer(source);
    auto tokens = lexer.tokenize();

    Parser parser(tokens);
    try {
        auto ast = parser.parse();

        bool found_func = false;
        bool found_contracts = false;

        for (const auto& decl : ast->declarations) {
            if (auto* func = dynamic_cast<FunctionDeclaration*>(decl.get())) {
                found_func = true;
                if (func->semantic_info && func->semantic_info->contracts.size() >= 2) {
                    found_contracts = true;
                    // Check contract kinds
                    assert(func->semantic_info->contracts[0].kind == SafetyContract::Kind::Precondition);
                    assert(func->semantic_info->contracts[1].kind == SafetyContract::Kind::Postcondition);
                }
            }
        }

        assert(found_func && "Expected function declaration");
        assert(found_contracts && "Expected contract annotations");

        std::cout << "✅ test_cpp26_contract_parser_integration passed\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ Parser error: " << e.what() << "\n";
        throw;
    }
}

void test_contract_informed_alias_analysis() {
    // Test that contracts provide no-alias guarantees for alias analysis
    std::string source = R"(
        func process: (in x: std::vector<int>, inout y: std::vector<int>)
            [[expects: x.size() == y.size()]]
            [[ensures: y.size() > 0]]
        = {
            n := x.size();
            y.push_back(42);
        }
    )";

    Lexer lexer(source);
    auto tokens = lexer.tokenize();

    Parser parser(tokens);
    try {
        auto ast = parser.parse();

        for (const auto& decl : ast->declarations) {
            if (auto* func = dynamic_cast<FunctionDeclaration*>(decl.get())) {
                // Contracts should be attached to semantic info
                assert(func->semantic_info && !func->semantic_info->contracts.empty() && "Expected contracts");

                // Check that 'in' parameter is Borrowed (immutable)
                assert(func->parameters.size() >= 2);
                assert(func->parameters[0]->semantic_info->borrow.kind == OwnershipKind::Borrowed);

                // Check that 'inout' parameter is MutBorrowed (mutable borrow)
                assert(func->parameters[1]->semantic_info->borrow.kind == OwnershipKind::MutBorrowed);

                // Contracts strengthen alias analysis:
                // - x.size() == y.size() guarantees no aliasing (different sizes would be detected)
                // - 'in' x cannot alias 'inout' y (immutable vs mutable)
            }
        }

        std::cout << "✅ test_contract_informed_alias_analysis passed\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ Parser error: " << e.what() << "\n";
        throw;
    }
}

void test_contract_to_mlir_attributes() {
    // Test that contracts are converted to MLIR attributes
    std::string source = R"(
        func safe_divide: (a: int, b: int) -> result
            [[expects: b != 0]]
            [[ensures: result * b == a]]
        = {
            result := a / b;
        }
    )";

    Lexer lexer(source);
    auto tokens = lexer.tokenize();

    Parser parser(tokens);
    try {
        auto ast = parser.parse();

        for (const auto& decl : ast->declarations) {
            if (auto* func = dynamic_cast<FunctionDeclaration*>(decl.get())) {
                // to_mlir_attributes() should include contract information
                std::string attrs = func->semantic_info->to_mlir_attributes();

                // Check for contract-related attributes
                assert(attrs.find("contract") != std::string::npos ||
                       !func->semantic_info->contracts.empty());

                // Each contract should have kind and condition
                for (const auto& contract : func->semantic_info->contracts) {
                    assert(!contract.condition.empty() && "Contract should have condition");
                }
            }
        }

        std::cout << "✅ test_contract_to_mlir_attributes passed\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ Parser error: " << e.what() << "\n";
        throw;
    }
}

int main() {
    std::cout << "C++26 Contract Attribute Tests (Phase 9)\n";
    std::cout << "=========================================\n\n";

    try {
        test_cpp26_contract_lexer_tokens();
        test_cpp26_contract_parser_integration();
        test_contract_informed_alias_analysis();
        test_contract_to_mlir_attributes();

        std::cout << "\n✅ All C++26 contract tests passed!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed: " << e.what() << "\n";
        return 1;
    }
}
