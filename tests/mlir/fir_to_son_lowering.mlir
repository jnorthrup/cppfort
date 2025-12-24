// RUN: cppfort-opt --convert-fir-to-son %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @simple_constant
  cpp2fir.func @simple_constant : () -> i32 {
    // CHECK: [[C42:%.*]] = sond.constant 42 : i32
    %0 = cpp2fir.constant 42 : i32
    // CHECK: return [[C42]] : i32
    cpp2fir.return %0 : i32
  }
}
