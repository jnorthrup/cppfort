

//=== Cpp2 type declarations ====================================================


#include "cpp2util.h"

#line 1 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.aYZRLhgWni/pure2-bugfix-for-requires-clause-unbraced-function-initializer.cpp2"


//=== Cpp2 type definitions and function declarations ===========================

#line 1 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.aYZRLhgWni/pure2-bugfix-for-requires-clause-unbraced-function-initializer.cpp2"
template<typename T> [[nodiscard]] auto f() -> decltype(auto)
CPP2_REQUIRES (std::regular<T>) ;
#line 2 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.aYZRLhgWni/pure2-bugfix-for-requires-clause-unbraced-function-initializer.cpp2"
auto main() -> int;

//=== Cpp2 function definitions =================================================

#line 1 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.aYZRLhgWni/pure2-bugfix-for-requires-clause-unbraced-function-initializer.cpp2"
template<typename T> [[nodiscard]] auto f() -> decltype(auto)
requires (std::regular<T>)  { return g(T());  }
#line 2 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.aYZRLhgWni/pure2-bugfix-for-requires-clause-unbraced-function-initializer.cpp2"
auto main() -> int                 {}

