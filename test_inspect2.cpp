

//=== Cpp2 type declarations ====================================================


#include "cpp2util.h"

#line 1 "/tmp/test_inspect2.cpp2"
namespace cpp2 {

#line 11 "/tmp/test_inspect2.cpp2"
}


//=== Cpp2 type definitions and function declarations ===========================

#line 1 "/tmp/test_inspect2.cpp2"
namespace cpp2 {
#line 2 "/tmp/test_inspect2.cpp2"
[[nodiscard]] auto f(cpp2::impl::in<char> c) -> int;

#line 8 "/tmp/test_inspect2.cpp2"
[[nodiscard]] auto main() -> int;

#line 11 "/tmp/test_inspect2.cpp2"
}


//=== Cpp2 function definitions =================================================

#line 1 "/tmp/test_inspect2.cpp2"
namespace cpp2 {
#line 2 "/tmp/test_inspect2.cpp2"
[[nodiscard]] auto f(cpp2::impl::in<char> c) -> int{
    return [&] () -> int { auto&& _expr = c;
        if (cpp2::impl::is(_expr, 'a')) { if constexpr( requires{1;} ) if constexpr( std::is_convertible_v<CPP2_TYPEOF((1)),int> ) return 1; else return int{}; else return int{}; }
        else return 0; }
    (); 
}
#line 8 "/tmp/test_inspect2.cpp2"
[[nodiscard]] auto main() -> int{
    return f('a'); 
}
}

