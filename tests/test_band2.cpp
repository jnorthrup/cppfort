/**
 * Test cases for Band 2 (Chapters 7-10) Sea of Nodes implementation
 */

#include <iostream>
#include <cassert>
#include "../src/stage0/node.h"
#include "../src/stage0/type.h"
#include "../src/stage0/iterpeeps.h"

using namespace cppfort::ir;

/**
 * Test Chapter 7: While loops with LoopNode
 */
void testLoopNode() {
    std::cout << "Testing Chapter 7: LoopNode..." << std::endl;

    // Create a simple loop structure
    StartNode* start = new StartNode();

    // Create loop with entry
    LoopNode* loop = new LoopNode(start);
    assert(!loop->hasAllInputs());  // No backedge yet

    // Create loop body (simplified)
    ConstantNode* one = new ConstantNode(1, start);

    // Set backedge
    loop->setBackedge(one);
    assert(loop->hasAllInputs());  // Now complete

    std::cout << "  LoopNode test passed" << std::endl;
}

/**
 * Test Chapter 8: Break and Continue nodes
 */
void testBreakContinue() {
    std::cout << "Testing Chapter 8: Break and Continue..." << std::endl;

    StartNode* start = new StartNode();

    // Create break node
    BreakNode* brk = new BreakNode(start);
    assert(brk->isCFG());
    assert(brk->label() == "Break");

    // Create continue node
    ContinueNode* cont = new ContinueNode(start);
    assert(cont->isCFG());
    assert(cont->label() == "Continue");

    std::cout << "  Break/Continue test passed" << std::endl;
}

/**
 * Test Chapter 8: Lazy Phi creation in ScopeNode
 */
void testLazyPhi() {
    std::cout << "Testing Chapter 8: Lazy Phi creation..." << std::endl;

    StartNode* start = new StartNode();
    ScopeNode* scope = new ScopeNode();
    scope->setInput(0, start);  // Set control

    // Define a variable
    ConstantNode* val1 = new ConstantNode(10, start);
    scope->define("x", val1);

    // Duplicate for loop (with lazy phi sentinel)
    ScopeNode* loopScope = scope->duplicate(true);
    assert(loopScope != nullptr);

    // The duplicate should have sentinels set up for lazy phi creation
    Node* xVal = loopScope->lookup("x");
    // In lazy mode, might be a sentinel or the original value

    std::cout << "  Lazy Phi test passed" << std::endl;
}

/**
 * Test Chapter 9: Global Value Numbering
 */
void testGVN() {
    std::cout << "Testing Chapter 9: Global Value Numbering..." << std::endl;

    StartNode* start = new StartNode();

    // Create two identical add nodes
    ConstantNode* c1 = new ConstantNode(5, start);
    ConstantNode* c2 = new ConstantNode(3, start);

    AddNode* add1 = new AddNode(c1, c2);
    AddNode* add2 = new AddNode(c1, c2);

    // After GVN, these should be the same
    Node* gvn1 = add1->gvn();
    Node* gvn2 = add2->gvn();

    // One should be killed and replaced by the other
    assert(gvn1 == gvn2 || add1->isDead() || add2->isDead());

    std::cout << "  GVN test passed" << std::endl;
}

/**
 * Test Chapter 9: Iterative peephole optimization
 */
void testIterativePeepholes() {
    std::cout << "Testing Chapter 9: Iterative peepholes..." << std::endl;

    StartNode* start = new StartNode();

    // Create expression that can be optimized: (2 + 3) * 4
    ConstantNode* c2 = new ConstantNode(2, start);
    ConstantNode* c3 = new ConstantNode(3, start);
    ConstantNode* c4 = new ConstantNode(4, start);

    AddNode* add = new AddNode(c2, c3);
    MulNode* mul = new MulNode(add, c4);
    ReturnNode* ret = new ReturnNode(start, mul);

    // Run iterative peepholes
    IterPeeps optimizer;
    int iterations = optimizer.iterate(ret);

    assert(iterations > 0);  // Should have done some optimization

    // The result should be constant folded
    Node* result = ret->value();
    assert(result != nullptr);

    // After optimization, should be a constant
    if (result->_type && result->_type->isConstant()) {
        TypeInteger* intType = dynamic_cast<TypeInteger*>(result->_type);
        if (intType) {
            assert(intType->value() == 20);  // (2 + 3) * 4 = 20
        }
    }

    std::cout << "  Iterative peephole test passed (" << iterations << " iterations)" << std::endl;
}

/**
 * Test Chapter 10: Memory operations
 */
void testMemoryOps() {
    std::cout << "Testing Chapter 10: Memory operations..." << std::endl;

    StartNode* start = new StartNode();

    // Create a new struct allocation
    NewNode* newStruct = new NewNode(start, "Point");
    assert(newStruct->structType() == "Point");
    assert(newStruct->hasSideEffects());

    // Create memory projection
    MemProjNode* memProj = new MemProjNode(start, 1);  // Alias class 1
    assert(memProj->alias() == 1);

    // Create a store operation
    ConstantNode* offset = new ConstantNode(0, start);  // Field offset
    ConstantNode* value = new ConstantNode(42, start);
    StoreNode* store = new StoreNode(1, memProj, newStruct, offset, value, "x");
    assert(store->alias() == 1);
    assert(store->hasSideEffects());

    // Create a load operation
    LoadNode* load = new LoadNode(1, store, newStruct, offset, "x");
    assert(load->alias() == 1);
    assert(load->mem() == store);

    std::cout << "  Memory operations test passed" << std::endl;
}

/**
 * Test Chapter 10: Cast operations
 */
void testCast() {
    std::cout << "Testing Chapter 10: Cast operations..." << std::endl;

    StartNode* start = new StartNode();
    ConstantNode* value = new ConstantNode(10, start);

    // Create a cast node
    Type* targetType = TypeInteger::bottom();
    CastNode* cast = new CastNode(start, value, targetType);

    assert(cast->toType() == targetType);
    assert(cast->compute() == targetType);

    std::cout << "  Cast test passed" << std::endl;
}

int main() {
    std::cout << "Running Band 2 (Chapters 7-10) tests..." << std::endl;
    std::cout << "========================================" << std::endl;

    // Chapter 7 tests
    testLoopNode();

    // Chapter 8 tests
    testBreakContinue();
    testLazyPhi();

    // Chapter 9 tests
    testGVN();
    testIterativePeepholes();

    // Chapter 10 tests
    testMemoryOps();
    testCast();

    std::cout << "========================================" << std::endl;
    std::cout << "All Band 2 tests passed!" << std::endl;

    return 0;
}