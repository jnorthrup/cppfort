

//=== Cpp2 type declarations ====================================================


#include "cpp2util.h"

#line 1 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.UYGSvXULa3/pure2-bugfix-for-non-local-function-expression.cpp2"

#line 9 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.UYGSvXULa3/pure2-bugfix-for-non-local-function-expression.cpp2"
class t;
    

//=== Cpp2 type definitions and function declarations ===========================

#line 1 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.UYGSvXULa3/pure2-bugfix-for-non-local-function-expression.cpp2"
//  Clang note: When using Clang as the Cpp1 compiler, the following uses of an unnamed function
//  require Clang 13 or higher; Clang 12 complains "lambda expression in an unevaluated operand"
//  Standalone Cpp1 repro: https://godbolt.org/z/dznnYTvc6

#line 5 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.UYGSvXULa3/pure2-bugfix-for-non-local-function-expression.cpp2"
template<typename T> concept v = []() -> bool { return true;  }(); 

using u = decltype([]() -> void{});

class t: public decltype([]() -> void{}) {

};

auto main() -> int;

//=== Cpp2 function definitions =================================================

#line 1 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.UYGSvXULa3/pure2-bugfix-for-non-local-function-expression.cpp2"

#line 13 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.UYGSvXULa3/pure2-bugfix-for-non-local-function-expression.cpp2"
auto main() -> int{}

