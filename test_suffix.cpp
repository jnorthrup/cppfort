

//=== Cpp2 type declarations ====================================================


#include "cpp2util.h"

#line 1 "/tmp/test_suffix.cpp2"


//=== Cpp2 type definitions and function declarations ===========================

#line 1 "/tmp/test_suffix.cpp2"
// Try suffix literal - 1j meaning j(1)
operator"" j: (a: int) -> int = a + 1;

#line 4 "/tmp/test_suffix.cpp2"
[[nodiscard]] auto main() -> int;

//=== Cpp2 function definitions =================================================

#line 1 "/tmp/test_suffix.cpp2"

#line 4 "/tmp/test_suffix.cpp2"
[[nodiscard]] auto main() -> int{
    int x {1j}; 
    return x; 
}

