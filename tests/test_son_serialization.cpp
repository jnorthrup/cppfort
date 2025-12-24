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

// Forward declare serialization function we'll implement
CRDTGraph serializeSONDialect(mlir::ModuleOp module);

void test_serialize_constant() {
    std::cout << "Test: Serialize SON constant operation\n";

    // Create MLIR context and load dialects
    MLIRContext context;
    context.loadDialect<Cpp2SONDialect>();
    context.loadDialect<func::FuncDialect>();

    // Create module with a constant operation
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());

    // Create function with constant
    auto funcType = builder.getFunctionType({}, {builder.getI32Type()});
    auto funcOp = builder.create<func::FuncOp>(loc, "test_func", funcType);

    Block* entry = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    auto constOp = builder.create<sond::ConstantOp>(
        loc, builder.getI32Type(), builder.getI32IntegerAttr(42));

    builder.create<func::ReturnOp>(loc, ValueRange{constOp.getResult()});

    // Serialize to CRDT graph
    CRDTGraph graph = serializeSONDialect(module);

    // Verify graph contains the constant node
    bool found_constant = false;
    for (const auto& [id, node] : graph.get_nodes()) {
        if (node.kind == Node::Kind::Constant) {
            found_constant = true;
            assert(std::get<int64_t>(node.value) == 42 && "Constant value should be 42");
            break;
        }
    }

    assert(found_constant && "Graph should contain constant node");

    std::cout << "✓ Serialize constant test passed\n\n";
}

void test_serialize_arithmetic() {
    std::cout << "Test: Serialize SON arithmetic operations\n";

    MLIRContext context;
    context.loadDialect<Cpp2SONDialect>();
    context.loadDialect<func::FuncDialect>();

    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());

    // Create function: %result = sond.add %c1, %c2
    auto funcType = builder.getFunctionType({}, {builder.getI32Type()});
    auto funcOp = builder.create<func::FuncOp>(loc, "add_func", funcType);

    Block* entry = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    auto const1 = builder.create<sond::ConstantOp>(
        loc, builder.getI32Type(), builder.getI32IntegerAttr(10));

    auto const2 = builder.create<sond::ConstantOp>(
        loc, builder.getI32Type(), builder.getI32IntegerAttr(32));

    auto addOp = builder.create<sond::AddOp>(
        loc, const1.getResult(), const2.getResult());

    builder.create<func::ReturnOp>(loc, ValueRange{addOp.getResult()});

    // Serialize
    CRDTGraph graph = serializeSONDialect(module);

    // Verify graph structure
    int constant_count = 0;
    int add_count = 0;

    for (const auto& [id, node] : graph.get_nodes()) {
        if (node.kind == Node::Kind::Constant) {
            constant_count++;
        } else if (node.kind == Node::Kind::Add) {
            add_count++;
            // Add node should have 2 inputs (the two constants)
            assert(node.inputs.size() == 2 && "Add node should have 2 inputs");
        }
    }

    assert(constant_count == 2 && "Should have 2 constant nodes");
    assert(add_count == 1 && "Should have 1 add node");

    std::cout << "✓ Serialize arithmetic test passed\n\n";
}

void test_serialize_edges() {
    std::cout << "Test: Serialize maintains data flow edges\n";

    MLIRContext context;
    context.loadDialect<Cpp2SONDialect>();
    context.loadDialect<func::FuncDialect>();

    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());

    auto funcType = builder.getFunctionType({}, {builder.getI32Type()});
    auto funcOp = builder.create<func::FuncOp>(loc, "edge_test", funcType);

    Block* entry = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    auto const1 = builder.create<sond::ConstantOp>(
        loc, builder.getI32Type(), builder.getI32IntegerAttr(5));

    auto const2 = builder.create<sond::ConstantOp>(
        loc, builder.getI32Type(), builder.getI32IntegerAttr(3));

    auto addOp = builder.create<sond::AddOp>(
        loc, const1.getResult(), const2.getResult());

    builder.create<func::ReturnOp>(loc, ValueRange{addOp.getResult()});

    // Serialize
    CRDTGraph graph = serializeSONDialect(module);

    // Find the add node
    NodeID add_node_id = 0;
    for (const auto& [id, node] : graph.get_nodes()) {
        if (node.kind == Node::Kind::Add) {
            add_node_id = id;
            break;
        }
    }

    assert(add_node_id != 0 && "Add node should exist");

    // Verify add node has edges from both constants
    const Node* add_node = graph.get_node(add_node_id);
    assert(add_node->inputs.size() == 2 && "Add should have 2 input edges");

    // Verify both inputs are constants
    for (NodeID input_id : add_node->inputs) {
        const Node* input = graph.get_node(input_id);
        assert(input != nullptr && "Input node should exist");
        assert(input->kind == Node::Kind::Constant && "Inputs should be constants");
    }

    std::cout << "✓ Serialize edges test passed\n\n";
}

void test_serialize_roundtrip_structure() {
    std::cout << "Test: Serialized graph preserves operation structure\n";

    MLIRContext context;
    context.loadDialect<Cpp2SONDialect>();
    context.loadDialect<func::FuncDialect>();

    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());

    // Create: (a + b) + c
    auto funcType = builder.getFunctionType({}, {builder.getI32Type()});
    auto funcOp = builder.create<func::FuncOp>(loc, "nested_add", funcType);

    Block* entry = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    auto a = builder.create<sond::ConstantOp>(
        loc, builder.getI32Type(), builder.getI32IntegerAttr(1));
    auto b = builder.create<sond::ConstantOp>(
        loc, builder.getI32Type(), builder.getI32IntegerAttr(2));
    auto c = builder.create<sond::ConstantOp>(
        loc, builder.getI32Type(), builder.getI32IntegerAttr(3));

    auto add1 = builder.create<sond::AddOp>(loc, a.getResult(), b.getResult());
    auto add2 = builder.create<sond::AddOp>(loc, add1.getResult(), c.getResult());

    builder.create<func::ReturnOp>(loc, ValueRange{add2.getResult()});

    // Serialize
    CRDTGraph graph = serializeSONDialect(module);

    // Verify structure: 3 constants, 2 adds
    int const_count = 0;
    int add_count = 0;
    NodeID nested_add_id = 0;

    for (const auto& [id, node] : graph.get_nodes()) {
        if (node.kind == Node::Kind::Constant) const_count++;
        if (node.kind == Node::Kind::Add) {
            add_count++;
            // Second add should have an add as one input
            for (NodeID input_id : node.inputs) {
                const Node* input = graph.get_node(input_id);
                if (input && input->kind == Node::Kind::Add) {
                    nested_add_id = id;
                }
            }
        }
    }

    assert(const_count == 3 && "Should have 3 constants");
    assert(add_count == 2 && "Should have 2 add operations");
    assert(nested_add_id != 0 && "Should have nested add structure");

    std::cout << "✓ Roundtrip structure test passed\n\n";
}

int main() {
    std::cout << "=== SON Dialect Serialization Tests ===\n\n";

    test_serialize_constant();
    test_serialize_arithmetic();
    test_serialize_edges();
    test_serialize_roundtrip_structure();

    std::cout << "=== All SON serialization tests passed! ===\n";
    return 0;
}
