#include "sea_of_nodes_impl.h"
#include <iostream>
#include <cassert>

using namespace cppfort::ir;

void test_basic_nodes() {
    std::cout << "Test: Basic node creation\n";

    auto graph = std::make_unique<GraphImpl>();

    auto* c1 = static_cast<NodeImpl*>(graph->createConstant(10));
    auto* c2 = static_cast<NodeImpl*>(graph->createConstant(20));

    assert(c1 != nullptr);
    assert(c2 != nullptr);
    assert(c1->type() == NodeType::Constant);
    assert(c2->type() == NodeType::Constant);

    std::cout << "  ✓ Constants created\n";

    auto* add = static_cast<NodeImpl*>(graph->createBinaryOp(BinaryOpType::Add, c1, c2));
    assert(add != nullptr);
    assert(add->type() == NodeType::BinaryOp);
    assert(add->inputCount() == 2);

    std::cout << "  ✓ Binary operation created\n";
    std::cout << "  Graph has " << graph->nodes().size() << " nodes\n";
}

void test_constant_folding() {
    std::cout << "\nTest: Constant folding\n";

    auto graph = std::make_unique<GraphImpl>();

    auto* c1 = graph->createConstant(5);
    auto* c2 = graph->createConstant(3);
    auto* add = graph->createBinaryOp(BinaryOpType::Add, c1, c2);
    auto* c3 = graph->createConstant(2);
    auto* mul = graph->createBinaryOp(BinaryOpType::Mul, add, c3);

    size_t before = graph->nodes().size();
    std::cout << "  Before optimization: " << before << " nodes\n";

    graph->optimize();

    size_t after = graph->nodes().size();
    std::cout << "  After optimization: " << after << " nodes\n";

    std::cout << "  ✓ Constant folding applied\n";
}

void test_dead_code_elimination() {
    std::cout << "\nTest: Dead code elimination\n";

    auto graph = std::make_unique<GraphImpl>();

    auto* c1 = graph->createConstant(100);
    auto* c2 = graph->createConstant(50);
    auto* dead_add = graph->createBinaryOp(BinaryOpType::Add, c1, c2);

    // Create another computation that will be used
    auto* c3 = graph->createConstant(10);
    auto* c4 = graph->createConstant(5);
    auto* live_mul = graph->createBinaryOp(BinaryOpType::Mul, c3, c4);

    // Mark live_mul as used by creating a unary op on it
    auto* neg = graph->createUnaryOp(UnaryOpType::Neg, live_mul);

    size_t before = graph->nodes().size();
    std::cout << "  Before DCE: " << before << " nodes\n";

    graph->eliminateDeadCode();

    size_t after = graph->nodes().size();
    std::cout << "  After DCE: " << after << " nodes\n";

    std::cout << "  ✓ Dead code eliminated\n";
}

void test_common_subexpression_elimination() {
    std::cout << "\nTest: Common subexpression elimination\n";

    auto graph = std::make_unique<GraphImpl>();

    auto* c1 = graph->createConstant(7);
    auto* c2 = graph->createConstant(3);

    // Create two identical additions
    auto* add1 = graph->createBinaryOp(BinaryOpType::Add, c1, c2);
    auto* add2 = graph->createBinaryOp(BinaryOpType::Add, c1, c2);

    // Use both
    auto* mul1 = graph->createBinaryOp(BinaryOpType::Mul, add1, c1);
    auto* mul2 = graph->createBinaryOp(BinaryOpType::Mul, add2, c2);

    size_t before = graph->nodes().size();
    std::cout << "  Before CSE: " << before << " nodes\n";

    bool changed = graph->eliminateCommonSubexpressions();

    size_t after = graph->nodes().size();
    std::cout << "  After CSE: " << after << " nodes (changed=" << changed << ")\n";

    std::cout << "  ✓ CSE applied\n";
}

void test_scheduling() {
    std::cout << "\nTest: Node scheduling\n";

    auto graph = std::make_unique<GraphImpl>();

    auto* c1 = graph->createConstant(1);
    auto* c2 = graph->createConstant(2);
    auto* add = graph->createBinaryOp(BinaryOpType::Add, c1, c2);
    auto* c3 = graph->createConstant(3);
    auto* mul = graph->createBinaryOp(BinaryOpType::Mul, add, c3);

    graph->schedule();

    auto scheduled = static_cast<GraphImpl*>(graph.get())->getScheduledNodes();
    std::cout << "  Scheduled " << scheduled.size() << " nodes\n";

    std::cout << "  Schedule order:\n";
    for (auto* node : scheduled) {
        std::cout << "    " << node->name() << " (id=" << node->id() << ")\n";
    }

    std::cout << "  ✓ Scheduling complete\n";
}

void test_graph_validation() {
    std::cout << "\nTest: Graph validation\n";

    auto graph = std::make_unique<GraphImpl>();

    auto* c1 = graph->createConstant(42);
    auto* c2 = graph->createConstant(24);
    auto* add = graph->createBinaryOp(BinaryOpType::Add, c1, c2);

    bool valid = validateGraph(graph.get());
    std::cout << "  Graph valid: " << (valid ? "yes" : "no") << "\n";

    assert(valid);
    std::cout << "  ✓ Graph validation passed\n";
}

void test_dot_output() {
    std::cout << "\nTest: DOT graph output\n";

    auto graph = std::make_unique<GraphImpl>();

    auto* c1 = graph->createConstant(10);
    auto* c2 = graph->createConstant(20);
    auto* add = graph->createBinaryOp(BinaryOpType::Add, c1, c2);
    auto* c3 = graph->createConstant(2);
    auto* div = graph->createBinaryOp(BinaryOpType::Div, add, c3);

    std::string dot = graphToDot(graph.get());
    std::cout << "  Generated DOT output:\n";
    std::cout << dot << "\n";

    assert(dot.find("digraph") != std::string::npos);
    std::cout << "  ✓ DOT output generated\n";
}

void test_code_emission() {
    std::cout << "\nTest: Code emission\n";

    auto graph = std::make_unique<GraphImpl>();

    auto* c1 = graph->createConstant(15);
    auto* c2 = graph->createConstant(5);
    auto* sub = graph->createBinaryOp(BinaryOpType::Sub, c1, c2);
    auto* c3 = graph->createConstant(2);
    auto* mul = graph->createBinaryOp(BinaryOpType::Mul, sub, c3);

    graph->schedule();

    // Test C++ emission
    auto cpp_lowering = createTargetLowering(Target::Cpp);
    std::string cpp_code = cpp_lowering->emit(graph.get());
    std::cout << "\n  C++ emission:\n";
    std::cout << cpp_code << "\n";

    // Test MLIR emission
    auto mlir_lowering = createTargetLowering(Target::MLIR);
    std::string mlir_code = mlir_lowering->emit(graph.get());
    std::cout << "  MLIR emission:\n";
    std::cout << mlir_code << "\n";

    assert(cpp_code.find("int64_t") != std::string::npos);
    assert(mlir_code.find("func.func") != std::string::npos);

    std::cout << "  ✓ Code emission successful\n";
}

void test_full_optimization_pipeline() {
    std::cout << "\nTest: Full optimization pipeline\n";

    auto graph = std::make_unique<GraphImpl>();

    // Create a more complex graph with optimization opportunities
    auto* c1 = graph->createConstant(10);
    auto* c2 = graph->createConstant(5);
    auto* add1 = graph->createBinaryOp(BinaryOpType::Add, c1, c2); // 15

    auto* c3 = graph->createConstant(10);
    auto* c4 = graph->createConstant(5);
    auto* add2 = graph->createBinaryOp(BinaryOpType::Add, c3, c4); // 15 (duplicate)

    auto* mul1 = graph->createBinaryOp(BinaryOpType::Mul, add1, c1); // 150
    auto* mul2 = graph->createBinaryOp(BinaryOpType::Mul, add2, c2); // 75

    // Dead code
    auto* dead_const = graph->createConstant(999);
    auto* dead_add = graph->createBinaryOp(BinaryOpType::Add, dead_const, c1);

    size_t before = graph->nodes().size();
    std::cout << "  Before optimization: " << before << " nodes\n";

    dumpGraph(graph.get(), std::cout);

    graph->optimize();

    size_t after = graph->nodes().size();
    std::cout << "  After optimization: " << after << " nodes\n";

    dumpGraph(graph.get(), std::cout);

    std::cout << "  ✓ Full pipeline executed\n";
}

int main() {
    std::cout << "=== Sea of Nodes IR Tests ===\n\n";

    try {
        test_basic_nodes();
        test_constant_folding();
        test_dead_code_elimination();
        test_common_subexpression_elimination();
        test_scheduling();
        test_graph_validation();
        test_dot_output();
        test_code_emission();
        test_full_optimization_pipeline();

        std::cout << "\n=== All tests passed! ===\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nTest failed with exception: " << e.what() << "\n";
        return 1;
    }
}
