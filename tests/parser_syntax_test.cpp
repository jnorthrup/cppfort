#include "../include/combinators/operators.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <string>

// Use new operators namespace
using namespace cpp2::parser::operators;
// We don't need 'using namespace cpp2::combinators::ebnf' because factories handle it

// Mock Token Types
enum class TokenType {
    Identifier,
    Integer,
    Plus,
    Minus,
    LParen,
    RParen
};

struct Token {
    TokenType type;
    int value = 0; // Payload
};

// Mock Input Stream
struct MockInput {
    std::vector<Token> tokens;
    size_t pos = 0;

    using Iterator = std::vector<Token>::const_iterator;

    explicit MockInput(std::vector<Token> t) : tokens(std::move(t)) {}
    explicit MockInput(std::vector<TokenType> types) {
        for(auto t : types) tokens.push_back({t});
    }

    [[nodiscard]] bool empty() const { return pos >= tokens.size(); }
    
    // ebnf::Token parser expects peek().type
    const Token& peek() const { return tokens[pos]; }
    
    // ebnf::Token parser expects advance() to return T (decltype(*begin))
    Token advance() { return tokens[pos++]; }
    
    Iterator begin() const { return tokens.begin() + pos; }
};

// Mock Parsers for Terminals using new factories
constexpr auto identifier = token(TokenType::Identifier);
constexpr auto integer = token(TokenType::Integer);
constexpr auto plus = token(TokenType::Plus);
constexpr auto minus = token(TokenType::Minus);
constexpr auto lparen = token(TokenType::LParen);
constexpr auto rparen = token(TokenType::RParen);

namespace rules {
    // EBNF: rule = integer { "+" integer } ;
    constexpr auto simple_expr = 
        integer >> *(plus >> integer);

    // EBNF: list = identifier % "," ; (using plus as comma for test)
    constexpr auto list_expr = 
        identifier % plus;
    
    // EBNF: optional = [ integer ] ;
    constexpr auto opt_expr = 
        -integer;
    
    // EBNF: one_or_more = { integer }+ ;
    constexpr auto some_expr = 
        +integer;

    // EBNF: transform = integer -> f
    constexpr auto transform_expr = 
        integer[ ([](auto&& val) { return 42; }) ];
        
    // EBNF: difference = integer - plus (silly example, matches integer if not plus? No, integer matches integer.)
    // Better difference test: (integer | plus) - plus  -> matches integer only
    constexpr auto diff_expr =
        (integer | plus) - plus;
}

int main() {
    std::cout << "Compiling parser syntax test...\n";
    
    // Test 1: Compile-time check is implicit
    
    // Test 2: Sequence and Many
    {
        MockInput input({TokenType::Integer, TokenType::Plus, TokenType::Integer});
        auto result = rules::simple_expr.parse(input);
        assert(result.success());
        std::cout << "Test 1 Passed: Sequence and Many\n";
    }

    // Test 3: List (sep_by)
    {
        MockInput input({TokenType::Identifier, TokenType::Plus, TokenType::Identifier});
        auto result = rules::list_expr.parse(input);
        assert(result.success());
        std::cout << "Test 2 Passed: List operator%\n";
    }

    // Test 4: Optional
    {
        MockInput input(std::vector<TokenType>{}); // Empty
        auto result = rules::opt_expr.parse(input);
        assert(result.success()); // Optional always succeeds
        assert(!result.value().has_value());
        std::cout << "Test 3 Passed: Optional operator-\n";
    }

    // Test 5: Map/Transform
    {
        MockInput input({TokenType::Integer});
        auto result = rules::transform_expr.parse(input);
        assert(result.success());
        assert(result.value() == 42);
        std::cout << "Test 4 Passed: Transform operator[]\n";
    }

    // Test 6: Difference
    {
        // Case A: Input is Integer -> Matches
        MockInput input1({TokenType::Integer});
        auto r1 = rules::diff_expr.parse(input1);
        assert(r1.success());
        
        // Case B: Input is Plus -> Fails
        MockInput input2({TokenType::Plus});
        auto r2 = rules::diff_expr.parse(input2);
        assert(!r2.success()); // Should fail because it matches 'plus' (RHS)
        
        std::cout << "Test 5 Passed: Difference operator-\n";
    }
    
    std::cout << "All syntax tests passed.\n";
    return 0;
}
