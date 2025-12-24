// RUN: cppfort-opt %s | FileCheck %s

module {
  // CHECK: sond.constant 42
  %c42 = sond.constant 42 : i32

  // CHECK: sond.constant 10
  %c10 = sond.constant 10 : i32

  // CHECK: sond.add
  %sum = sond.add %c42, %c10 : i32
}
