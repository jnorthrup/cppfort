// RUN: cppfort-opt %s | FileCheck %s

module {
  // CHECK-LABEL: cpp2fir.func @main
  cpp2fir.func @main : () -> i32 {
    // CHECK: cpp2fir.constant 42
    %0 = cpp2fir.constant 42 : i32
    // CHECK: cpp2fir.return
    cpp2fir.return %0 : i32
  }
}
