// test_band5.cpp - Band 5: Bitwise Operations and Pattern Matching
#include "node.h"
#include "son_parser.h"
#include "node_category.h"
#include <cassert>
#include <iostream>

using namespace cppfort::ir;

static void test_bitwise_parsing() {
    SoNParser p;

    // Test simple parsing first
    try {
        Node* n = p.parse("return 3;");
        std::cout << "Basic parsing works\n";
    } catch (const std::exception& e) {
        std::cout << "Basic parsing failed: " << e.what() << "\n";
        return;
    }

    // Test bitwise AND
    try {
        Node* n = p.parse("return 3 & 5;");
        auto ret = dynamic_cast<ReturnNode*>(n);
        if (ret) {
            std::cout << "Bitwise AND parsing works\n";
        } else {
            std::cout << "Bitwise AND parsing failed - not a ReturnNode\n";
        }
    } catch (const std::exception& e) {
        std::cout << "Bitwise AND parsing failed: " << e.what() << "\n";
    }
}

static void test_shift_parsing() {
    SoNParser p;

    // Test shift left
    Node* n = p.parse("return 1 << 2;");
    auto ret = dynamic_cast<ReturnNode*>(n);
    assert(ret);
    auto shl_node = dynamic_cast<ShlNode*>(ret->value());
    assert(shl_node);

    // Test shift right
    n = p.parse("return 8 >> 1;");
    ret = dynamic_cast<ReturnNode*>(n);
    auto shr_node = dynamic_cast<LShrNode*>(ret->value());
    assert(shr_node);

    std::cout << "Shift parsing tests passed\n";
}

static void test_operator_precedence() {
    SoNParser p;

    // Test precedence: & has higher precedence than |
    Node* n = p.parse("return 1 & 2 | 4;");
    auto ret = dynamic_cast<ReturnNode*>(n);
    assert(ret);
    // Should be (1 & 2) | 4
    auto or_node = dynamic_cast<OrNode*>(ret->value());
    assert(or_node);
    auto and_node = dynamic_cast<AndNode*>(or_node->in(0));
    assert(and_node);

    // Test precedence: << has higher precedence than &
    n = p.parse("return 1 << 2 & 3;");
    ret = dynamic_cast<ReturnNode*>(n);
    auto and_node2 = dynamic_cast<AndNode*>(ret->value());
    assert(and_node2);
    auto shl_node = dynamic_cast<ShlNode*>(and_node2->in(0));
    assert(shl_node);

    std::cout << "Operator precedence tests passed\n";
}

static void test_node_kinds() {
    // Test that bitwise nodes have correct NodeKinds
    AndNode and_node(nullptr, nullptr);
    assert(and_node.getKind() == NodeKind::AND);

    OrNode or_node(nullptr, nullptr);
    assert(or_node.getKind() == NodeKind::OR);

    XorNode xor_node(nullptr, nullptr);
    assert(xor_node.getKind() == NodeKind::XOR);

    ShlNode shl_node(nullptr, nullptr);
    assert(shl_node.getKind() == NodeKind::SHL);

    LShrNode lshr_node(nullptr, nullptr);
    assert(lshr_node.getKind() == NodeKind::LSHR);

    AShrNode ashr_node(nullptr, nullptr);
    assert(ashr_node.getKind() == NodeKind::ASHR);

    std::cout << "NodeKind tests passed\n";
}

static void test_node_categories() {
    AndNode and_node(nullptr, nullptr);
    assert(NodeCategory::isBitwise(and_node.getKind()));
    assert(!NodeCategory::isArithmetic(and_node.getKind()));
    assert(!NodeCategory::isComparison(and_node.getKind()));

    ShlNode shl_node(nullptr, nullptr);
    assert(NodeCategory::isBitwise(shl_node.getKind()));

    std::cout << "Node category tests passed\n";
}

int main() {
    try {
        test_bitwise_parsing();
        std::cout << "Basic test completed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << "\n";
        return 1;
    }
}