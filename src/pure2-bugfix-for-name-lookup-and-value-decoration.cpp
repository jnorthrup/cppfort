

//=== Cpp2 type declarations ====================================================


#include "cpp2util.h"

#line 1 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.dbAzUYAUnl/pure2-bugfix-for-name-lookup-and-value-decoration.cpp2"


//=== Cpp2 type definitions and function declarations ===========================

#line 1 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.dbAzUYAUnl/pure2-bugfix-for-name-lookup-and-value-decoration.cpp2"

using vals_ret = int;
[[nodiscard]] auto vals() -> vals_ret;

#line 6 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.dbAzUYAUnl/pure2-bugfix-for-name-lookup-and-value-decoration.cpp2"
[[nodiscard]] auto main() -> int;

//=== Cpp2 function definitions =================================================

#line 1 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.dbAzUYAUnl/pure2-bugfix-for-name-lookup-and-value-decoration.cpp2"
[[nodiscard]] auto vals() -> vals_ret{
        cpp2::impl::deferred_init<int> i;
#line 2 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.dbAzUYAUnl/pure2-bugfix-for-name-lookup-and-value-decoration.cpp2"
    i.construct(42);
    return std::move(i.value()); 
}

#line 6 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.dbAzUYAUnl/pure2-bugfix-for-name-lookup-and-value-decoration.cpp2"
[[nodiscard]] auto main() -> int{
    auto v {vals()}; 
    static_cast<void>(cpp2::move(v));
}

