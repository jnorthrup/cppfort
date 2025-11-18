#include "mlir_region_node.h"
#include <iostream>

int main() {
    using namespace cppfort::ir::mlir;

    // Test A: Add operation first, then add value defining it - expect linking
    RegionNode r;
    RegionNode::Operation op("arith.addi");
    op.operand_indices = {}; // no operands for simplicity
    op.result_index = SIZE_MAX;
    size_t op_idx = r.addOperation(op);

    RegionNode::Value val("i32", "res");
    val.defining_op = op_idx; // says this value is defined by op
    size_t val_idx = r.addValue(val);

    const auto* got_op = r.getOperation(op_idx);
    const auto* got_val = r.getValue(val_idx);

    if (!got_op || !got_val) {
        std::cerr << "FAILED: op or val missing\n";
        return 1;
    }
    if (got_op->result_index != val_idx) {
        std::cerr << "FAILED: op.result_index expected " << val_idx << " got " << got_op->result_index << "\n";
        return 1;
    }
    if (got_val->defining_op != op_idx) {
        std::cerr << "FAILED: val.defining_op expected " << op_idx << " got " << got_val->defining_op << "\n";
        return 1;
    }

    // Test B: Add values first, then an op with those values as operands - expect uses
    RegionNode r2;
    RegionNode::Value a("i32", "a");
    RegionNode::Value b("i32", "b");
    size_t a_idx = r2.addValue(a);
    size_t b_idx = r2.addValue(b);

    RegionNode::Operation op2("arith.addi");
    op2.operand_indices = {a_idx, b_idx};
    op2.result_index = SIZE_MAX;
    size_t op2_idx = r2.addOperation(op2);

    const auto* a_val = r2.getValue(a_idx);
    const auto* b_val = r2.getValue(b_idx);

    if (!a_val || !b_val) {
        std::cerr << "FAILED: a or b missing\n";
        return 1;
    }
    // Expect `op2` to be recorded as a use for both a and b
    bool a_has_use = false;
    for (auto u : a_val->use_ops) if (u == op2_idx) a_has_use = true;
    bool b_has_use = false;
    for (auto u : b_val->use_ops) if (u == op2_idx) b_has_use = true;

    if (!a_has_use || !b_has_use) {
        std::cerr << "FAILED: operand use not recorded for a or b\n";
        return 1;
    }

    std::cout << "RegionNode SSA linking tests passed\n";
    return 0;
}
