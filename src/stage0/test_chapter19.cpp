#include <iostream>
#include <cassert>
#include "node.h"
#include "machine.h"
#include "instruction_selection.h"

namespace cppfort::ir {

// Chapter 19: Instruction Selection Tests
// Tests the MLIR dialect instruction selection patterns

void test_mlir_arith_machine() {
    std::cout << "Testing MLIR Arith Machine patterns...\n";

    MachineRegistry registry;
    MLIRArithMachine* arithMachine = dynamic_cast<MLIRArithMachine*>(registry.getMachine("mlir-arith"));
    assert(arithMachine != nullptr);

    PatternMatcher matcher;
    arithMachine->registerPatterns(matcher);

    // Test that arith machine can handle ADD nodes
    assert(arithMachine->canHandle(NodeKind::ADD));
    assert(arithMachine->canHandle(NodeKind::SUB));
    assert(arithMachine->canHandle(NodeKind::MUL));
    assert(arithMachine->canHandle(NodeKind::DIV));
    assert(arithMachine->canHandle(NodeKind::EQ));
    assert(arithMachine->canHandle(NodeKind::LT));
    assert(arithMachine->canHandle(NodeKind::CONSTANT));

    // Test that it cannot handle control flow
    assert(!arithMachine->canHandle(NodeKind::IF));

    std::cout << "✓ MLIR Arith Machine patterns work\n";
}

void test_mlir_func_machine() {
    std::cout << "Testing MLIR Func Machine patterns...\n";

    MachineRegistry registry;
    MLIRFuncMachine* funcMachine = dynamic_cast<MLIRFuncMachine*>(registry.getMachine("mlir-func"));
    assert(funcMachine != nullptr);

    PatternMatcher matcher;
    funcMachine->registerPatterns(matcher);

    // Test that func machine can handle function nodes
    assert(funcMachine->canHandle(NodeKind::FUN));
    assert(funcMachine->canHandle(NodeKind::CALL));
    assert(funcMachine->canHandle(NodeKind::RETURN));
    assert(funcMachine->canHandle(NodeKind::PARM));

    // Test that it cannot handle arithmetic
    assert(!funcMachine->canHandle(NodeKind::ADD));

    std::cout << "✓ MLIR Func Machine patterns work\n";
}

void test_machine_registry() {
    std::cout << "Testing Machine Registry...\n";

    MachineRegistry registry;

    // Test available machines
    auto machines = registry.availableMachines();
    assert(machines.size() == 5);

    // Test getting machines by name
    assert(registry.getMachine("mlir-arith") != nullptr);
    assert(registry.getMachine("mlir-func") != nullptr);
    assert(registry.getMachine("mlir-cf") != nullptr);
    assert(registry.getMachine("mlir-scf") != nullptr);
    assert(registry.getMachine("mlir-memref") != nullptr);

    // Test non-existent machine
    assert(registry.getMachine("non-existent") == nullptr);

    std::cout << "✓ Machine Registry works\n";
}

void test_instruction_selection() {
    std::cout << "Testing Instruction Selection...\n";

    // Create a simple graph: constant + constant
    StartNode* start = new StartNode();
    ConstantNode* c1 = new ConstantNode(10, start);
    ConstantNode* c2 = new ConstantNode(20, start);
    AddNode* add = new AddNode(c1, c2);
    ReturnNode* ret = new ReturnNode(start, add);

    // Run instruction selection
    InstructionSelection isel({"mlir-arith"});
    Node* result = isel.selectInstructions(start);

    // Graph should still be valid
    assert(result == start);
    assert(ret->value() == add);
    assert(add->in(0) == c1);
    assert(add->in(1) == c2);

    std::cout << "✓ Instruction Selection works\n";
}

} // namespace cppfort::ir

int main() {
    std::cout << "=== Chapter 19: Instruction Selection Test Suite ===\n\n";

    try {
        cppfort::ir::test_mlir_arith_machine();
        cppfort::ir::test_mlir_func_machine();
        cppfort::ir::test_machine_registry();
        cppfort::ir::test_instruction_selection();

        std::cout << "\n=== All Chapter 19 tests passed! ===\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Test failed: " << ex.what() << std::endl;
        return 1;
    }
}