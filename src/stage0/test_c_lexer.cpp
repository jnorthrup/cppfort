#include "c_parser.h"
#include <iostream>
#include <cassert>

using namespace cppfort::c;

void test_keywords() {
    std::cout << "=== Testing C Keywords ===" << std::endl;

    CLexer lexer("int main void return if else while for", "test.c");
    auto tokens = lexer.tokenize();

    assert(tokens.size() == 9); // 8 keywords + EOF
    assert(tokens[0].type == CTokenType::INT);
    assert(tokens[1].type == CTokenType::IDENTIFIER); // main is identifier, not keyword
    assert(tokens[2].type == CTokenType::VOID);
    assert(tokens[3].type == CTokenType::RETURN);
    assert(tokens[4].type == CTokenType::IF);
    assert(tokens[5].type == CTokenType::ELSE);
    assert(tokens[6].type == CTokenType::WHILE);
    assert(tokens[7].type == CTokenType::FOR);
    assert(tokens[8].type == CTokenType::EOF_TOKEN);

    std::cout << "✓ Keywords recognized correctly" << std::endl;
}

void test_operators() {
    std::cout << "=== Testing C Operators ===" << std::endl;

    CLexer lexer("+ - * / % ++ -- += -= == != < > <= >= && || & | ^ ~ << >> = !", "test.c");
    auto tokens = lexer.tokenize();

    assert(tokens[0].type == CTokenType::PLUS);
    assert(tokens[1].type == CTokenType::MINUS);
    assert(tokens[2].type == CTokenType::STAR);
    assert(tokens[3].type == CTokenType::SLASH);
    assert(tokens[4].type == CTokenType::PERCENT);
    assert(tokens[5].type == CTokenType::INCREMENT);
    assert(tokens[6].type == CTokenType::DECREMENT);
    assert(tokens[7].type == CTokenType::PLUS_ASSIGN);
    assert(tokens[8].type == CTokenType::MINUS_ASSIGN);
    assert(tokens[9].type == CTokenType::EQ);
    assert(tokens[10].type == CTokenType::NE);
    assert(tokens[11].type == CTokenType::LT);
    assert(tokens[12].type == CTokenType::GT);
    assert(tokens[13].type == CTokenType::LE);
    assert(tokens[14].type == CTokenType::GE);
    assert(tokens[15].type == CTokenType::AND);
    assert(tokens[16].type == CTokenType::OR);
    assert(tokens[17].type == CTokenType::AMPERSAND);
    assert(tokens[18].type == CTokenType::PIPE);
    assert(tokens[19].type == CTokenType::CARET);
    assert(tokens[20].type == CTokenType::TILDE);
    assert(tokens[21].type == CTokenType::LSHIFT);
    assert(tokens[22].type == CTokenType::RSHIFT);
    assert(tokens[23].type == CTokenType::ASSIGN);
    assert(tokens[24].type == CTokenType::EXCLAIM);

    std::cout << "✓ Operators tokenized correctly" << std::endl;
}

void test_literals() {
    std::cout << "=== Testing C Literals ===" << std::endl;

    CLexer lexer("42 0x2A 077 3.14 1.5e-10 \"hello\" 'c'", "test.c");
    auto tokens = lexer.tokenize();

    assert(tokens[0].type == CTokenType::INTEGER_LITERAL);
    assert(tokens[0].text == "42");

    assert(tokens[1].type == CTokenType::INTEGER_LITERAL);
    assert(tokens[1].text == "0x2A");

    assert(tokens[2].type == CTokenType::INTEGER_LITERAL);
    assert(tokens[2].text == "077");

    assert(tokens[3].type == CTokenType::FLOAT_LITERAL);
    assert(tokens[3].text == "3.14");

    assert(tokens[4].type == CTokenType::FLOAT_LITERAL);
    assert(tokens[4].text.find("1.5e-10") != std::string::npos);

    assert(tokens[5].type == CTokenType::STRING_LITERAL);
    assert(tokens[5].text == "hello");

    assert(tokens[6].type == CTokenType::CHAR_LITERAL);
    assert(tokens[6].text == "c");

    std::cout << "✓ Literals parsed correctly" << std::endl;
}

void test_comments() {
    std::cout << "=== Testing C Comments ===" << std::endl;

    CLexer lexer("int x; // line comment\nint y; /* block comment */ int z;", "test.c");
    auto tokens = lexer.tokenize();

    // Comments should be stripped - only tokens should remain
    assert(tokens[0].type == CTokenType::INT);
    assert(tokens[1].type == CTokenType::IDENTIFIER);
    assert(tokens[1].text == "x");
    assert(tokens[2].type == CTokenType::SEMICOLON);
    assert(tokens[3].type == CTokenType::INT);
    assert(tokens[4].type == CTokenType::IDENTIFIER);
    assert(tokens[4].text == "y");
    assert(tokens[5].type == CTokenType::SEMICOLON);
    assert(tokens[6].type == CTokenType::INT);
    assert(tokens[7].type == CTokenType::IDENTIFIER);
    assert(tokens[7].text == "z");
    assert(tokens[8].type == CTokenType::SEMICOLON);

    std::cout << "✓ Comments stripped correctly" << std::endl;
}

void test_simple_function() {
    std::cout << "=== Testing Simple C Function ===" << std::endl;

    const char* source = R"(
int add(int a, int b) {
    return a + b;
}
)";

    CLexer lexer(source, "test.c");
    auto tokens = lexer.tokenize();

    // Verify structure: int add ( int a , int b ) { return a + b ; }
    assert(tokens[0].type == CTokenType::INT);
    assert(tokens[1].type == CTokenType::IDENTIFIER && tokens[1].text == "add");
    assert(tokens[2].type == CTokenType::LPAREN);
    assert(tokens[3].type == CTokenType::INT);
    assert(tokens[4].type == CTokenType::IDENTIFIER && tokens[4].text == "a");
    assert(tokens[5].type == CTokenType::COMMA);
    assert(tokens[6].type == CTokenType::INT);
    assert(tokens[7].type == CTokenType::IDENTIFIER && tokens[7].text == "b");
    assert(tokens[8].type == CTokenType::RPAREN);
    assert(tokens[9].type == CTokenType::LBRACE);
    assert(tokens[10].type == CTokenType::RETURN);
    assert(tokens[11].type == CTokenType::IDENTIFIER && tokens[11].text == "a");
    assert(tokens[12].type == CTokenType::PLUS);
    assert(tokens[13].type == CTokenType::IDENTIFIER && tokens[13].text == "b");
    assert(tokens[14].type == CTokenType::SEMICOLON);
    assert(tokens[15].type == CTokenType::RBRACE);

    std::cout << "✓ Simple function tokenized correctly" << std::endl;
}

void test_pointer_and_array_syntax() {
    std::cout << "=== Testing Pointer/Array Syntax ===" << std::endl;

    CLexer lexer("int *ptr; int arr[10]; ptr->field; arr[i]; &x; *p;", "test.c");
    auto tokens = lexer.tokenize();

    // Verify pointer/array operators
    bool hasArrow = false, hasBrackets = false, hasAmpersand = false, hasStar = false;

    for (const auto& tok : tokens) {
        if (tok.type == CTokenType::ARROW) hasArrow = true;
        if (tok.type == CTokenType::LBRACKET) hasBrackets = true;
        if (tok.type == CTokenType::AMPERSAND) hasAmpersand = true;
        if (tok.type == CTokenType::STAR) hasStar = true;
    }

    assert(hasArrow && "Missing -> operator");
    assert(hasBrackets && "Missing [] operator");
    assert(hasAmpersand && "Missing & operator");
    assert(hasStar && "Missing * operator");

    std::cout << "✓ Pointer/array operators recognized" << std::endl;
}

int main() {
    std::cout << "Running C Lexer Tests" << std::endl;
    std::cout << "=====================" << std::endl << std::endl;

    test_keywords();
    test_operators();
    test_literals();
    test_comments();
    test_simple_function();
    test_pointer_and_array_syntax();

    std::cout << std::endl << "All C lexer tests passed! ✓" << std::endl;
    return 0;
}
