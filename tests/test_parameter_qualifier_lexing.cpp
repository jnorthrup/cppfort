#include <iostream>
#include <cassert>
#include <sstream>
#include <string_view>

#include "../include/lexer.hpp"

using namespace cpp2_transpiler;

// Test 1: Lex 'inout' keyword
void test_lex_inout() {
    std::cout << "Test: Lex 'inout' keyword\n";

    std::string_view source = "inout";
    Lexer lexer(source);
    auto tokens = lexer.tokenize();

    assert(tokens.size() >= 2 && "Should have at least 2 tokens (inout + EOF)");
    assert(tokens[0].type == TokenType::Inout && "First token should be Inout");
    assert(tokens[0].lexeme == "inout" && "Lexeme should be 'inout'");

    std::cout << "  Token type: " << static_cast<int>(tokens[0].type) << "\n";
    std::cout << "  Lexeme: " << tokens[0].lexeme << "\n";
    std::cout << "✓ Test passed\n";
}

// Test 2: Lex 'out' keyword
void test_lex_out() {
    std::cout << "Test: Lex 'out' keyword\n";

    std::string_view source = "out";
    Lexer lexer(source);
    auto tokens = lexer.tokenize();

    assert(tokens.size() >= 2 && "Should have at least 2 tokens (out + EOF)");
    assert(tokens[0].type == TokenType::Out && "First token should be Out");
    assert(tokens[0].lexeme == "out" && "Lexeme should be 'out'");

    std::cout << "  Token type: " << static_cast<int>(tokens[0].type) << "\n";
    std::cout << "  Lexeme: " << tokens[0].lexeme << "\n";
    std::cout << "✓ Test passed\n";
}

// Test 3: Lex 'move' keyword
void test_lex_move() {
    std::cout << "Test: Lex 'move' keyword\n";

    std::string_view source = "move";
    Lexer lexer(source);
    auto tokens = lexer.tokenize();

    assert(tokens.size() >= 2 && "Should have at least 2 tokens (move + EOF)");
    assert(tokens[0].type == TokenType::Move && "First token should be Move");
    assert(tokens[0].lexeme == "move" && "Lexeme should be 'move'");

    std::cout << "  Token type: " << static_cast<int>(tokens[0].type) << "\n";
    std::cout << "  Lexeme: " << tokens[0].lexeme << "\n";
    std::cout << "✓ Test passed\n";
}

// Test 4: Lex 'forward' keyword
void test_lex_forward() {
    std::cout << "Test: Lex 'forward' keyword\n";

    std::string_view source = "forward";
    Lexer lexer(source);
    auto tokens = lexer.tokenize();

    assert(tokens.size() >= 2 && "Should have at least 2 tokens (forward + EOF)");
    assert(tokens[0].type == TokenType::Forward && "First token should be Forward");
    assert(tokens[0].lexeme == "forward" && "Lexeme should be 'forward'");

    std::cout << "  Token type: " << static_cast<int>(tokens[0].type) << "\n";
    std::cout << "  Lexeme: " << tokens[0].lexeme << "\n";
    std::cout << "✓ Test passed\n";
}

// Test 5: Lex 'virtual' keyword (already exists, verify)
void test_lex_virtual() {
    std::cout << "Test: Lex 'virtual' keyword\n";

    std::string_view source = "virtual";
    Lexer lexer(source);
    auto tokens = lexer.tokenize();

    assert(tokens.size() >= 2 && "Should have at least 2 tokens (virtual + EOF)");
    assert(tokens[0].type == TokenType::Virtual && "First token should be Virtual");
    assert(tokens[0].lexeme == "virtual" && "Lexeme should be 'virtual'");

    std::cout << "  Token type: " << static_cast<int>(tokens[0].type) << "\n";
    std::cout << "  Lexeme: " << tokens[0].lexeme << "\n";
    std::cout << "✓ Test passed\n";
}

// Test 6: Lex 'override' keyword (already exists, verify)
void test_lex_override() {
    std::cout << "Test: Lex 'override' keyword\n";

    std::string_view source = "override";
    Lexer lexer(source);
    auto tokens = lexer.tokenize();

    assert(tokens.size() >= 2 && "Should have at least 2 tokens (override + EOF)");
    assert(tokens[0].type == TokenType::Override && "First token should be Override");
    assert(tokens[0].lexeme == "override" && "Lexeme should be 'override'");

    std::cout << "  Token type: " << static_cast<int>(tokens[0].type) << "\n";
    std::cout << "  Lexeme: " << tokens[0].lexeme << "\n";
    std::cout << "✓ Test passed\n";
}

// Test 7: Lex function with multiple qualifiers in parameter list
void test_lex_function_with_qualifiers() {
    std::cout << "Test: Lex function with parameter qualifiers\n";

    // func foo: (inout x: int, out y: int, move z: int) -> int
    std::string_view source = "func foo: (inout x: int, out y: int, move z: int) -> int";
    Lexer lexer(source);
    auto tokens = lexer.tokenize();

    // Find qualifier tokens
    bool found_inout = false;
    bool found_out = false;
    bool found_move = false;

    for (const auto& token : tokens) {
        if (token.type == TokenType::Inout) found_inout = true;
        if (token.type == TokenType::Out) found_out = true;
        if (token.type == TokenType::Move) found_move = true;
    }

    assert(found_inout && "Should find 'inout' qualifier");
    assert(found_out && "Should find 'out' qualifier");
    assert(found_move && "Should find 'move' qualifier");

    std::cout << "  Found inout: " << (found_inout ? "yes" : "no") << "\n";
    std::cout << "  Found out: " << (found_out ? "yes" : "no") << "\n";
    std::cout << "  Found move: " << (found_move ? "yes" : "no") << "\n";
    std::cout << "✓ Test passed\n";
}

// Test 8: Ensure identifier named 'inout' doesn't conflict (if not a keyword)
// Actually, in Cpp2 these are keywords, so this test verifies they are keywords
void test_qualifiers_are_keywords() {
    std::cout << "Test: Parameter qualifiers are keywords, not identifiers\n";

    std::string_view source = "inout out move forward virtual override";
    Lexer lexer(source);
    auto tokens = lexer.tokenize();

    // All should be keyword tokens, not identifiers
    assert(tokens[0].type == TokenType::Inout && "'inout' should be keyword");
    assert(tokens[1].type == TokenType::Out && "'out' should be keyword");
    assert(tokens[2].type == TokenType::Move && "'move' should be keyword");
    assert(tokens[3].type == TokenType::Forward && "'forward' should be keyword");
    assert(tokens[4].type == TokenType::Virtual && "'virtual' should be keyword");
    assert(tokens[5].type == TokenType::Override && "'override' should be keyword");

    std::cout << "  All qualifiers recognized as keywords\n";
    std::cout << "✓ Test passed\n";
}

int main() {
    std::cout << "=== Parameter Qualifier Lexing Tests ===\n\n";

    test_lex_inout();
    test_lex_out();
    test_lex_move();
    test_lex_forward();
    test_lex_virtual();
    test_lex_override();
    test_lex_function_with_qualifiers();
    test_qualifiers_are_keywords();

    std::cout << "\n=== All tests passed ===\n";
    return 0;
}
