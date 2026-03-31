

//=== Cpp2 type declarations ====================================================


#include "cpp2util.h"

#line 1 "/tmp/test_inout.cpp2"


//=== Cpp2 type definitions and function declarations ===========================

#line 1 "/tmp/test_inout.cpp2"
auto foo(int& x) -> void;

#line 5 "/tmp/test_inout.cpp2"
[[nodiscard]] auto main() -> int;

//=== Cpp2 function definitions =================================================

#line 1 "/tmp/test_inout.cpp2"
auto foo(int& x) -> void{
#line 2 "/tmp/test_inout.cpp2"
    x += 1;
}

#line 5 "/tmp/test_inout.cpp2"
[[nodiscard]] auto main() -> int{
    int y {5}; 
    foo(y);
    return y; 
}

