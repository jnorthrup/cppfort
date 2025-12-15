

//=== Cpp2 type declarations ====================================================


#include "cpp2util.h"

#line 1 "/Users/jim/work/cppfort/corpus/inputs/pure2-bugfix-for-max-munch.cpp2"


//=== Cpp2 type definitions and function declarations ===========================

#line 1 "/Users/jim/work/cppfort/corpus/inputs/pure2-bugfix-for-max-munch.cpp2"
template<typename T> auto inline constexpr v{ 0 };
#line 2 "/Users/jim/work/cppfort/corpus/inputs/pure2-bugfix-for-max-munch.cpp2"
auto main() -> int;

//=== Cpp2 function definitions =================================================

#line 1 "/Users/jim/work/cppfort/corpus/inputs/pure2-bugfix-for-max-munch.cpp2"

#line 2 "/Users/jim/work/cppfort/corpus/inputs/pure2-bugfix-for-max-munch.cpp2"
auto main() -> int { static_cast<void>(v<cpp2::i32> == v<cpp2::i64>);  }

