

//=== Cpp2 type declarations ====================================================


#include "cpp2util.h"

#line 1 "/tmp/test_assert.cpp2"


//=== Cpp2 type definitions and function declarations ===========================

#line 1 "/tmp/test_assert.cpp2"
auto main() -> int;

//=== Cpp2 function definitions =================================================

#line 1 "/tmp/test_assert.cpp2"
auto main() -> int{
#line 2 "/tmp/test_assert.cpp2"
    if (cpp2::cpp2_default.is_active() && !(1 != 2) ) { cpp2::cpp2_default.report_violation(CPP2_CONTRACT_MSG("message")); }
}

