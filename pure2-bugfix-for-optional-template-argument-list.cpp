

//=== Cpp2 type declarations ====================================================


#include "cpp2util.h"

#line 1 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.WhjvusyEf0/pure2-bugfix-for-optional-template-argument-list.cpp2"


//=== Cpp2 type definitions and function declarations ===========================

#line 1 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.WhjvusyEf0/pure2-bugfix-for-optional-template-argument-list.cpp2"
extern std::plus<> const plus;
#line 2 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.WhjvusyEf0/pure2-bugfix-for-optional-template-argument-list.cpp2"
[[nodiscard]] auto main() -> int;

//=== Cpp2 function definitions =================================================

#line 1 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.WhjvusyEf0/pure2-bugfix-for-optional-template-argument-list.cpp2"
std::plus<> const plus {}; 
#line 2 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.WhjvusyEf0/pure2-bugfix-for-optional-template-argument-list.cpp2"
[[nodiscard]] auto main() -> int { return std::plus<>()(0, 0); }

