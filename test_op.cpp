

//=== Cpp2 type declarations ====================================================


#include "cpp2util.h"

#line 1 "/tmp/test_op.cpp2"


//=== Cpp2 type definitions and function declarations ===========================

#line 1 "/tmp/test_op.cpp2"
auto operator+(cpp2::impl::in<int> a, cpp2::impl::in<int> b) -> int;

//=== Cpp2 function definitions =================================================

#line 1 "/tmp/test_op.cpp2"
[[nodiscard]] auto operator+(cpp2::impl::in<int> a, cpp2::impl::in<int> b) -> int { return a + b;  }

