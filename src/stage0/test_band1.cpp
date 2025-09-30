// Band 1: Control flow - if/region/phi/comparisons
#include "node.h"
#include "son_parser.h"
#include <cassert>
#include <iostream>

using namespace cppfort::ir;

static void test_comparisons() {
    SoNParser p;
    Node* n = p.parse("return 3==3;");
    auto ret = dynamic_cast<ReturnNode*>(n);
    assert(ret);
    auto c = dynamic_cast<ConstantNode*>(ret->value());
    assert(c && c->_value == 1);

    n = p.parse("return 2<1;");
    ret = dynamic_cast<ReturnNode*>(n);
    c = dynamic_cast<ConstantNode*>(ret->value());
    assert(c && c->_value == 0);
}

static void test_simple_if() {
    SoNParser p;
    Node* n = p.parse("if(1==1) return 7; else return 9;");
    assert(n != nullptr); // parsing succeeded
}

static void test_if_with_vars() {
    SoNParser p;
    Node* n = p.parse("int x=1; if(0<1){ x=2; } else { x=3; } return x;");
    assert(n != nullptr);
}

static void test_nested_if() {
    SoNParser p;
    Node* n = p.parse("int x=1; if(1==1){ if(0<1){ x=5; } } return x;");
    assert(n != nullptr);
}

static void test_if_no_else() {
    SoNParser p;
    Node* n = p.parse("int x=1; if(0<1){ x=2; } return x;");
    assert(n != nullptr);
}

int main() {
    try { test_comparisons(); std::cout << "comparisons ok\n"; } catch (const std::exception& e) { std::cerr << "comparisons fail: " << e.what() << "\n"; return 1; }
    try { test_simple_if(); std::cout << "simple_if ok\n"; } catch (const std::exception& e) { std::cerr << "simple_if fail: " << e.what() << "\n"; return 1; }
    try { test_if_with_vars(); std::cout << "if_with_vars ok\n"; } catch (const std::exception& e) { std::cerr << "if_with_vars fail: " << e.what() << "\n"; return 1; }
    try { test_nested_if(); std::cout << "nested_if ok\n"; } catch (const std::exception& e) { std::cerr << "nested_if fail: " << e.what() << "\n"; return 1; }
    try { test_if_no_else(); std::cout << "if_no_else ok\n"; } catch (const std::exception& e) { std::cerr << "if_no_else fail: " << e.what() << "\n"; return 1; }
    std::cout << "Band 1 tests passed\n";
    return 0;
}
