

//=== Cpp2 type declarations ====================================================


#include "cpp2util.h"

#line 1 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.0lDXWztqMb/pure2-concept-definition.cpp2"


//=== Cpp2 type definitions and function declarations ===========================

#line 1 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.0lDXWztqMb/pure2-concept-definition.cpp2"
template<typename T> concept arithmetic = std::integral<T> || std::floating_point<T>; 
#line 2 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.0lDXWztqMb/pure2-concept-definition.cpp2"
auto main() -> int;

//=== Cpp2 function definitions =================================================

#line 1 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.0lDXWztqMb/pure2-concept-definition.cpp2"

#line 2 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.0lDXWztqMb/pure2-concept-definition.cpp2"
auto main() -> int      {
  if (cpp2::testing.is_active() && !(arithmetic<cpp2::i32>) ) { cpp2::testing.report_violation(""); }
  if (cpp2::testing.is_active() && !(arithmetic<float>) ) { cpp2::testing.report_violation(""); }
}

