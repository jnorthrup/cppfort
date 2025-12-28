//===- test_sccp_control_flow.cpp - SCCP Control Flow Analysis Tests ---------===//
///
/// Tests for control flow reachability analysis in SCCP.
/// Verifies tracking of executable blocks and dead branch detection.
///
//===----------------------------------------------------------------------===//

#include "../include/LatticeValue.h"
#include "../include/DataflowAnalysis.h"
#include <cassert>
#include <iostream>
#include <unordered_set>

using namespace cppfort::sccp;

void test_control_flow_initially_all_unreachable() {
    std::cout << "Test: Control flow initially all unreachable\n";

    DataflowAnalysis analysis;

    // Initially, no blocks are marked as reachable
    assert(analysis.getReachableBlockCount() == 0 && "Initial reachable count should be 0");

    // Check specific block is not reachable
    void* block = (void*)0x1000;
    assert(!analysis.isBlockReachable(block) && "Block should not be reachable initially");

    std::cout << "✓ Control flow initially all unreachable test passed\n\n";
}

void test_mark_block_reachable() {
    std::cout << "Test: Mark block reachable\n";

    DataflowAnalysis analysis;
    void* block = (void*)0x2000;

    assert(!analysis.isBlockReachable(block) && "Block should not be reachable initially");

    analysis.markBlockReachable(block);

    assert(analysis.isBlockReachable(block) && "Block should be reachable after marking");
    assert(analysis.getReachableBlockCount() == 1 && "Reachable count should be 1");

    std::cout << "✓ Mark block reachable test passed\n\n";
}

void test_mark_block_reachable_idempotent() {
    std::cout << "Test: Mark block reachable is idempotent\n";

    DataflowAnalysis analysis;
    void* block = (void*)0x3000;

    analysis.markBlockReachable(block);
    size_t count1 = analysis.getReachableBlockCount();

    analysis.markBlockReachable(block);
    size_t count2 = analysis.getReachableBlockCount();

    assert(count1 == count2 && "Marking same block twice should not change count");
    assert(count1 == 1 && "Count should be 1");

    std::cout << "✓ Mark block reachable idempotent test passed\n\n";
}

void test_mark_multiple_blocks_reachable() {
    std::cout << "Test: Mark multiple blocks reachable\n";

    DataflowAnalysis analysis;
    void* block1 = (void*)0x4000;
    void* block2 = (void*)0x5000;
    void* block3 = (void*)0x6000;

    analysis.markBlockReachable(block1);
    analysis.markBlockReachable(block2);
    analysis.markBlockReachable(block3);

    assert(analysis.getReachableBlockCount() == 3 && "All 3 blocks should be reachable");
    assert(analysis.isBlockReachable(block1) && "Block1 should be reachable");
    assert(analysis.isBlockReachable(block2) && "Block2 should be reachable");
    assert(analysis.isBlockReachable(block3) && "Block3 should be reachable");

    std::cout << "✓ Mark multiple blocks reachable test passed\n\n";
}

void test_branch_condition_constant_true() {
    std::cout << "Test: Branch condition constant true\n";

    DataflowAnalysis analysis;
    void* condition = (void*)0x7000;
    void* trueBlock = (void*)0x8000;
    void* falseBlock = (void*)0x9000;

    // Set condition to true
    analysis.setLatticeValue(condition, LatticeValue::getConstant(true));

    // Evaluate branch
    analysis.evaluateBranch(condition, trueBlock, falseBlock);

    assert(analysis.isBlockReachable(trueBlock) && "True block should be reachable");
    assert(!analysis.isBlockReachable(falseBlock) && "False block should not be reachable");

    std::cout << "✓ Branch condition constant true test passed\n\n";
}

void test_branch_condition_constant_false() {
    std::cout << "Test: Branch condition constant false\n";

    DataflowAnalysis analysis;
    void* condition = (void*)0xA000;
    void* trueBlock = (void*)0xB000;
    void* falseBlock = (void*)0xC000;

    // Set condition to false
    analysis.setLatticeValue(condition, LatticeValue::getConstant(false));

    // Evaluate branch
    analysis.evaluateBranch(condition, trueBlock, falseBlock);

    assert(!analysis.isBlockReachable(trueBlock) && "True block should not be reachable");
    assert(analysis.isBlockReachable(falseBlock) && "False block should be reachable");

    std::cout << "✓ Branch condition constant false test passed\n\n";
}

void test_branch_condition_top() {
    std::cout << "Test: Branch condition Top (unknown)\n";

    DataflowAnalysis analysis;
    void* condition = (void*)0xD000;
    void* trueBlock = (void*)0xE000;
    void* falseBlock = (void*)0xF000;

    // Condition is Top (unknown) - should not mark any blocks
    analysis.evaluateBranch(condition, trueBlock, falseBlock);

    assert(!analysis.isBlockReachable(trueBlock) && "True block should not be reachable");
    assert(!analysis.isBlockReachable(falseBlock) && "False block should not be reachable");

    std::cout << "✓ Branch condition Top test passed\n\n";
}

void test_clear_reachable_blocks() {
    std::cout << "Test: Clear reachable blocks\n";

    DataflowAnalysis analysis;
    void* block1 = (void*)0x10000;
    void* block2 = (void*)0x11000;

    analysis.markBlockReachable(block1);
    analysis.markBlockReachable(block2);

    assert(analysis.getReachableBlockCount() == 2 && "Should have 2 reachable blocks");

    analysis.clearReachableBlocks();

    assert(analysis.getReachableBlockCount() == 0 && "Should have 0 reachable blocks after clear");
    assert(!analysis.isBlockReachable(block1) && "Block1 should not be reachable");
    assert(!analysis.isBlockReachable(block2) && "Block2 should not be reachable");

    std::cout << "✓ Clear reachable blocks test passed\n\n";
}

void test_dead_branch_detection() {
    std::cout << "Test: Dead branch detection\n";

    DataflowAnalysis analysis;
    void* condition = (void*)0x12000;
    void* trueBlock = (void*)0x13000;
    void* falseBlock = (void*)0x14000;

    // Set condition to true - false block is dead
    analysis.setLatticeValue(condition, LatticeValue::getConstant(true));
    analysis.evaluateBranch(condition, trueBlock, falseBlock);

    assert(analysis.isBlockReachable(trueBlock) && "True block should be reachable");
    assert(!analysis.isBlockReachable(falseBlock) && "False block should be dead (unreachable)");

    std::cout << "✓ Dead branch detection test passed\n\n";
}

void test_entry_block_reachable() {
    std::cout << "Test: Entry block reachable\n";

    DataflowAnalysis analysis;
    void* entryBlock = (void*)0x15000;

    // Mark entry block as reachable to start analysis
    analysis.markBlockReachable(entryBlock);

    assert(analysis.isBlockReachable(entryBlock) && "Entry block should be reachable");

    std::cout << "✓ Entry block reachable test passed\n\n";
}

int main() {
    std::cout << "=== SCCP Control Flow Analysis Tests ===\n\n";

    test_control_flow_initially_all_unreachable();
    test_mark_block_reachable();
    test_mark_block_reachable_idempotent();
    test_mark_multiple_blocks_reachable();
    test_branch_condition_constant_true();
    test_branch_condition_constant_false();
    test_branch_condition_top();
    test_clear_reachable_blocks();
    test_dead_branch_detection();
    test_entry_block_reachable();

    std::cout << "=== All Control Flow Analysis tests passed! ===\n";
    return 0;
}
