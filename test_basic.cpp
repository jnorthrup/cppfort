#include <string_view>
#include <vector>
#include <cstdint>
#include <string>


//=== Cpp2 type declarations ====================================================


#include "cpp2util.h"

#line 1 "/tmp/test_basic.cpp2"

#line 6 "/tmp/test_basic.cpp2"
namespace cpp2 {

#line 12 "/tmp/test_basic.cpp2"
}


//=== Cpp2 type definitions and function declarations ===========================

#line 1 "/tmp/test_basic.cpp2"

#line 6 "/tmp/test_basic.cpp2"
namespace cpp2 {

[[nodiscard]] auto foo() -> int;

#line 12 "/tmp/test_basic.cpp2"
}


//=== Cpp2 function definitions =================================================

#line 1 "/tmp/test_basic.cpp2"

#line 6 "/tmp/test_basic.cpp2"
namespace cpp2 {

#line 8 "/tmp/test_basic.cpp2"
[[nodiscard]] auto foo() -> int{
    return 42; 
}

}

