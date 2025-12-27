#include "../include/Cpp2SONDialect.h"
#include "../include/mlir_cpp2_dialect.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <cassert>
#include <iostream>

using namespace mlir;
using namespace mlir::sond;
using namespace cppfort::mlir_son;

// Forward declare deserialization function we'll implement
mlir::ModuleOp deserializeSONDialect(const CRDTGraph& graph, mlir::MLIRContext* context);

void test_deserialize_constant() {
    std::cout << "Test: Deserialize CRDT graph with constant node\n";

    MLIRContext context;
    context.loadDialect<Cpp2SONDialect>();
    context.loadDialect<func::FuncDialect>();

    // Create CRDT graph with constant node
    CRDTGraph graph;

    Node constant_node{Node::Kind::Constant, 1};
    constant_node.value = int64_t(42);
    constant_node.timestamp = 100;

    Patch add_node;
    add_node.operation = Patch::Op::AddNode;
    add_node.data = constant_node;
    graph.apply_patch(add_node);

    // Deserialize to MLIR
    auto module = deserializeSONDialect(graph, &context);

    // Verify module contains constant operation
    bool found_constant = false;
    module.walk([&](sond::ConstantOp op) {
        found_constant = true;
        auto attr = op.getValue();
        if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
            assert(intAttr.getInt() == 42 && "Constant value should be 42");
        }
    });

    assert(found_constant && "Module should contain sond.constant operation");

    std::cout << "✓ Deserialize constant test passed\n\n";
}

void test_deserialize_arithmetic() {
    std::cout << "Test: Deserialize CRDT graph with arithmetic operations\n";

    MLIRContext context;
    context.loadDialect<Cpp2SONDialect>();
    context.loadDialect<func::FuncDialect>();

    // Create CRDT graph: %result = add %c1, %c2
    CRDTGraph graph;

    // Constant 1
    Node const1{Node::Kind::Constant, 1};
    const1.value = int64_t(10);
    const1.timestamp = 100;

    Patch add_const1;
    add_const1.operation = Patch::Op::AddNode;
    add_const1.data = const1;
    graph.apply_patch(add_const1);

    // Constant 2
    Node const2{Node::Kind::Constant, 2};
    const2.value = int64_t(32);
    const2.timestamp = 101;

    Patch add_const2;
    add_const2.operation = Patch::Op::AddNode;
    add_const2.data = const2;
    graph.apply_patch(add_const2);

    // Add operation
    Node add_node{Node::Kind::Add, 3};
    add_node.inputs = {1, 2};
    add_node.timestamp = 102;

    Patch add_add;
    add_add.operation = Patch::Op::AddNode;
    add_add.data = add_node;
    graph.apply_patch(add_add);

    // Add edges
    Patch edge1;
    edge1.operation = Patch::Op::AddEdge;
    edge1.data = std::make_pair(NodeID(1), NodeID(3));
    graph.apply_patch(edge1);

    Patch edge2;
    edge2.operation = Patch::Op::AddEdge;
    edge2.data = std::make_pair(NodeID(2), NodeID(3));
    graph.apply_patch(edge2);

    // Deserialize
    auto module = deserializeSONDialect(graph, &context);

    // Verify operations
    int const_count = 0;
    int add_count = 0;

    module.walk([&](sond::ConstantOp op) { const_count++; });
    module.walk([&](sond::AddOp op) { add_count++; });

    assert(const_count == 2 && "Should have 2 constant operations");
    assert(add_count == 1 && "Should have 1 add operation");

    std::cout << "✓ Deserialize arithmetic test passed\n\n";
}

void test_deserialize_edges() {
    std::cout << "Test: Deserialize preserves SSA use-def chains\n";

    MLIRContext context;
    context.loadDialect<Cpp2SONDialect>();
    context.loadDialect<func::FuncDialect>();

    // Create graph with edges
    CRDTGraph graph;

    Node const1{Node::Kind::Constant, 1};
    const1.value = int64_t(5);
    const1.timestamp = 100;

    Node const2{Node::Kind::Constant, 2};
    const2.value = int64_t(3);
    const2.timestamp = 101;

    Node add_node{Node::Kind::Add, 3};
    add_node.inputs = {1, 2};
    add_node.timestamp = 102;

    Patch p1;
    p1.operation = Patch::Op::AddNode;
    p1.data = const1;
    graph.apply_patch(p1);

    Patch p2;
    p2.operation = Patch::Op::AddNode;
    p2.data = const2;
    graph.apply_patch(p2);

    Patch p3;
    p3.operation = Patch::Op::AddNode;
    p3.data = add_node;
    graph.apply_patch(p3);

    Patch e1;
    e1.operation = Patch::Op::AddEdge;
    e1.data = std::make_pair(NodeID(1), NodeID(3));
    graph.apply_patch(e1);

    Patch e2;
    e2.operation = Patch::Op::AddEdge;
    e2.data = std::make_pair(NodeID(2), NodeID(3));
    graph.apply_patch(e2);

    // Deserialize
    auto module = deserializeSONDialect(graph, &context);

    // Verify add operation has correct operands
    bool verified = false;
    module.walk([&](sond::AddOp op) {
        // Add should have 2 operands
        assert(op.getLhs() != nullptr && "Add should have lhs operand");
        assert(op.getRhs() != nullptr && "Add should have rhs operand");

        // Operands should be results of constant operations
        auto lhs_def = op.getLhs().getDefiningOp();
        auto rhs_def = op.getRhs().getDefiningOp();

        assert(isa<sond::ConstantOp>(lhs_def) && "LHS should be from constant");
        assert(isa<sond::ConstantOp>(rhs_def) && "RHS should be from constant");

        verified = true;
    });

    assert(verified && "Add operation verification should succeed");

    std::cout << "✓ Deserialize edges test passed\n\n";
}

void test_deserialize_roundtrip() {
    std::cout << "Test: Serialize-deserialize roundtrip preserves structure\n";

    MLIRContext context;
    context.loadDialect<Cpp2SONDialect>();
    context.loadDialect<func::FuncDialect>();

    // Create original MLIR module
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    auto original_module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(original_module.getBody());

    auto funcType = builder.getFunctionType({}, {builder.getI32Type()});
    auto funcOp = builder.create<func::FuncOp>(loc, "test", funcType);

    Block* entry = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    auto c1 = builder.create<sond::ConstantOp>(
        loc, builder.getI32Type(), builder.getI32IntegerAttr(7));
    auto c2 = builder.create<sond::ConstantOp>(
        loc, builder.getI32Type(), builder.getI32IntegerAttr(8));
    auto add = builder.create<sond::AddOp>(loc, builder.getI32Type(), c1.getResult(), c2.getResult());

    builder.create<func::ReturnOp>(loc, ValueRange{add.getResult()});

    // Serialize (using existing serialization function)
    CRDTGraph serializeSONDialect(mlir::ModuleOp module);
    CRDTGraph graph = serializeSONDialect(original_module);

    // Deserialize
    auto deserialized_module = deserializeSONDialect(graph, &context);

    // Verify structure matches
    int orig_const_count = 0;
    int orig_add_count = 0;
    int deser_const_count = 0;
    int deser_add_count = 0;

    original_module.walk([&](sond::ConstantOp op) { orig_const_count++; });
    original_module.walk([&](sond::AddOp op) { orig_add_count++; });

    deserialized_module.walk([&](sond::ConstantOp op) { deser_const_count++; });
    deserialized_module.walk([&](sond::AddOp op) { deser_add_count++; });

    assert(orig_const_count == deser_const_count && "Constant count should match");
    assert(orig_add_count == deser_add_count && "Add count should match");

    std::cout << "✓ Roundtrip test passed\n\n";
}

int main() {
    std::cout << "=== SON Dialect Deserialization Tests ===\n\n";

    test_deserialize_constant();
    test_deserialize_arithmetic();
    test_deserialize_edges();
    test_deserialize_roundtrip();

    std::cout << "=== All SON deserialization tests passed! ===\n";
    return 0;
}
