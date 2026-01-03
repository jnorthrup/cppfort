

//=== Cpp2 type declarations ====================================================


#include "cpp2util.h"

#line 1 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.HX0XdSLcnD/pure2-bugfix-for-variable-template.cpp2"


//=== Cpp2 type definitions and function declarations ===========================

#line 1 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.HX0XdSLcnD/pure2-bugfix-for-variable-template.cpp2"
template<auto V> extern int const v0;
#line 2 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.HX0XdSLcnD/pure2-bugfix-for-variable-template.cpp2"
[[nodiscard]] auto main() -> int;

//=== Cpp2 function definitions =================================================

#line 1 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.HX0XdSLcnD/pure2-bugfix-for-variable-template.cpp2"
template<auto V> int const v0 {0}; 
#line 2 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.HX0XdSLcnD/pure2-bugfix-for-variable-template.cpp2"
[[nodiscard]] auto main() -> int { return v0<0>; }

