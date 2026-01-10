

//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================

template<typename T> [[nodiscard]] auto f() -> decltype(auto)
CPP2_REQUIRES (std::regular<T>) ;
auto main() -> int;

//=== Cpp2 function definitions =================================================

template<typename T> [[nodiscard]] auto f() -> decltype(auto)
requires (std::regular<T>)  { return g(T());  }
auto main() -> int                 {}

