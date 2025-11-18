#include "mlir_region_node.h"
#include <iostream>

int main() {
    using namespace cppfort::ir::mlir;

    // Test 1: Parent/child and nesting levels
    RegionNode fn_region(RegionNode::RegionType::FUNCTION, "foo");
    auto block = std::make_unique<RegionNode>(RegionNode::RegionType::BLOCK, "entry");
    fn_region.addChild(std::move(block));
    if (fn_region.getChildren().size() != 1) {
        std::cerr << "FAILED: expected 1 child in function region\n";
        return 1;
    }
    RegionNode* child = fn_region.getChild(0);
    if (!child) {
        std::cerr << "FAILED: child pointer is null\n";
        return 1;
    }
    if (child->getParent() != &fn_region) {
        std::cerr << "FAILED: child's parent mismatch\n";
        return 1;
    }
    if (child->getNestingLevel() != fn_region.getNestingLevel() + 1) {
        std::cerr << "FAILED: child nesting level incorrect\n";
        return 1;
    }

    // Test 2: Operations and values
    RegionNode test_region(RegionNode::RegionType::BLOCK, "block");
    RegionNode::Operation op("arith.addi");
    op.result_index = SIZE_MAX;
    size_t op_idx = test_region.addOperation(op);

    RegionNode::Value val("i32", "a");
    val.defining_op = op_idx;
    size_t val_idx = test_region.addValue(val);

    // Ensure operation and value retrieval works
    const auto* got_op = test_region.getOperation(op_idx);
    const auto* got_val = test_region.getValue(val_idx);
    if (!got_op || got_op->name != "arith.addi") {
        std::cerr << "FAILED: operation not found or name mismatch\n";
        return 1;
    }
    if (!got_val || got_val->name != "a" || got_val->type != "i32") {
        std::cerr << "FAILED: value not found or properties mismatch\n";
        return 1;
    }

    // Test 3: isFunctionRegion by type and by op name
    RegionNode f1(RegionNode::RegionType::FUNCTION, "f1");
    if (!f1.isFunctionRegion()) {
        std::cerr << "FAILED: f1 should be recognized as function region by type\n";
        return 1;
    }
    RegionNode f2(RegionNode::RegionType::UNKNOWN);
    RegionNode::Operation funcop("func.func");
    f2.addOperation(funcop);
    if (!f2.isFunctionRegion()) {
        std::cerr << "FAILED: f2 should be recognized as function region by op name\n";
        return 1;
    }

    // Test 4: isBlockRegion
    RegionNode b1(RegionNode::RegionType::BLOCK);
    if (!b1.isBlockRegion()) {
        std::cerr << "FAILED: b1 should be recognized as block region\n";
        return 1;
    }
    RegionNode b2(RegionNode::RegionType::UNKNOWN);
    b2.addArgument("arg1");
    if (!b2.isBlockRegion()) {
        std::cerr << "FAILED: b2 should be recognized as block region by arguments\n";
        return 1;
    }

    // Test 5: findChildrenByType
    RegionNode root(RegionNode::RegionType::UNKNOWN);
    auto child_block_a = std::make_unique<RegionNode>(RegionNode::RegionType::BLOCK, "a");
    auto child_func = std::make_unique<RegionNode>(RegionNode::RegionType::FUNCTION, "f");
    auto nested_block = std::make_unique<RegionNode>(RegionNode::RegionType::BLOCK, "nested");
    child_func->addChild(std::move(nested_block));
    root.addChild(std::move(child_block_a));
    root.addChild(std::move(child_func));

    auto blocks = root.findChildrenByType(RegionNode::RegionType::BLOCK);
    if (blocks.size() != 2) {
        std::cerr << "FAILED: expected to find 2 blocks (direct + nested)\n";
        return 1;
    }

    // Test 6: Validation rules
    RegionNode valid_func(RegionNode::RegionType::FUNCTION);
    auto only_block = std::make_unique<RegionNode>(RegionNode::RegionType::BLOCK);
    valid_func.addChild(std::move(only_block));
    if (!valid_func.validate()) {
        std::cerr << "FAILED: valid_func should be valid (one block child)\n";
        return 1;
    }
    RegionNode bad_func(RegionNode::RegionType::FUNCTION);
    bad_func.addChild(std::make_unique<RegionNode>(RegionNode::RegionType::BLOCK));
    bad_func.addChild(std::make_unique<RegionNode>(RegionNode::RegionType::BLOCK));
    if (bad_func.validate()) {
        std::cerr << "FAILED: bad_func should be invalid (two block children)\n";
        return 1;
    }

    // Test source location
    RegionNode sr; sr.setSourceLocation(10, 20);
    if (sr.getSourceStart() != 10 || sr.getSourceEnd() != 20 || sr.getSourceLength() != 10) {
        std::cerr << "FAILED: source location getters mismatch\n";
        return 1;
    }

    // Test 7: toString/printTree declarations
    // These functions are declared in the header; if they're not implemented we'll
    // get a link-time failure which will be the RED part of TDD and drive the
    // subsequent implementation.
    std::string s = test_region.toString(2);
    if (s.empty()) {
        std::cerr << "FAILED: toString should return non-empty string (even for an empty block)\n";
        return 1;
    }
    test_region.printTree(0);

    std::cout << "All RegionNode basic tests passed\n";
    return 0;
}
