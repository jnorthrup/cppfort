//===- test_sccp_debug_logging.cpp - Test SCCP Debug Logging -------------===//
///
/// Tests that SCCP passes support debug logging via LLVM_DEBUG.
/// Run with: LLVM_DEBUG=fir-sccp ./test_sccp_debug_logging
///
//===----------------------------------------------------------------------===//

#include "../include/LatticeValue.h"
#include "../include/ConstantFolder.h"
#include "../include/DataflowAnalysis.h"
#include "test_timeout.hpp"

#include <iostream>
#include <cassert>
#include <string>

using namespace cppfort::sccp;

void test_lattice_toString() {
    // Test Top
    LatticeValue top = LatticeValue::getTop();
    assert(top.toString() == "Top");

    // Test Bottom
    LatticeValue bottom = LatticeValue::getBottom();
    assert(bottom.toString() == "Bottom");

    // Test integer constant
    LatticeValue intConst = LatticeValue::getConstant(static_cast<int64_t>(42));
    assert(intConst.toString() == "Constant(42)");

    // Test boolean constant
    LatticeValue boolTrue = LatticeValue::getConstant(true);
    assert(boolTrue.toString() == "Constant(true)");

    LatticeValue boolFalse = LatticeValue::getConstant(false);
    assert(boolFalse.toString() == "Constant(false)");

    // Test integer range
    LatticeValue range = LatticeValue::getIntegerRange(0, 100);
    assert(range.toString() == "Range[0, 100]");

    // Test NaN
    LatticeValue nan = LatticeValue::getFloatSpecial(LatticeValue::FloatSpecialValue::NaN);
    assert(nan.toString() == "NaN");

    // Test +Infinity
    LatticeValue posInf = LatticeValue::getFloatSpecial(LatticeValue::FloatSpecialValue::PosInfinity);
    assert(posInf.toString() == "+Infinity");

    // Test -Infinity
    LatticeValue negInf = LatticeValue::getFloatSpecial(LatticeValue::FloatSpecialValue::NegInfinity);
    assert(negInf.toString() == "-Infinity");
}

void test_debug_logging_infrastructure() {
    // Verify that toString() is available for all lattice value types
    // This ensures debug logging won't crash when LLVM_DEBUG is enabled

    LatticeValue top = LatticeValue::getTop();
    LatticeValue bottom = LatticeValue::getBottom();
    LatticeValue constant = LatticeValue::getConstant(static_cast<int64_t>(123));
    LatticeValue range = LatticeValue::getIntegerRange(-10, 10);
    LatticeValue floatSpecial = LatticeValue::getFloatSpecial(LatticeValue::FloatSpecialValue::NaN);

    // All toString() calls should succeed without crashing
    std::string s1 = top.toString();
    std::string s2 = bottom.toString();
    std::string s3 = constant.toString();
    std::string s4 = range.toString();
    std::string s5 = floatSpecial.toString();

    assert(!s1.empty());
    assert(!s2.empty());
    assert(!s3.empty());
    assert(!s4.empty());
    assert(!s5.empty());
}

int main() {
    run_with_timeout("test_lattice_toString", test_lattice_toString);
    run_with_timeout("test_debug_logging_infrastructure", test_debug_logging_infrastructure);

    std::cout << "Debug logging infrastructure tests passed!" << std::endl;
    std::cout << "To enable debug output, run with: LLVM_DEBUG=fir-sccp <test-executable>" << std::endl;

    return 0;
}
