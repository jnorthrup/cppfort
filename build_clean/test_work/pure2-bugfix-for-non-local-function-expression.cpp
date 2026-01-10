

//=== Cpp2 type declarations ====================================================




class t;
    

//=== Cpp2 type definitions and function declarations ===========================

//  Clang note: When using Clang as the Cpp1 compiler, the following uses of an unnamed function
//  require Clang 13 or higher; Clang 12 complains "lambda expression in an unevaluated operand"
//  Standalone Cpp1 repro: https://godbolt.org/z/dznnYTvc6

template<typename T> concept v = []() -> bool { return true;  }(); 

using u = decltype([]() -> void{});

class t: public decltype([]() -> void{}) {

};

auto main() -> int;

//=== Cpp2 function definitions =================================================


auto main() -> int{}

