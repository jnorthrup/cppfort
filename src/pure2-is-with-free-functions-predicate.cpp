

//=== Cpp2 type declarations ====================================================


#include "cpp2util.h"

#line 1 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.bvly2ngV4j/pure2-is-with-free-functions-predicate.cpp2"


//=== Cpp2 type definitions and function declarations ===========================

#line 1 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.bvly2ngV4j/pure2-is-with-free-functions-predicate.cpp2"
auto fun(auto const& v) -> void;

#line 15 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.bvly2ngV4j/pure2-is-with-free-functions-predicate.cpp2"
[[nodiscard]] auto main() -> int;

#line 21 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.bvly2ngV4j/pure2-is-with-free-functions-predicate.cpp2"
[[nodiscard]] auto pred_i(cpp2::impl::in<int> x) -> bool;

#line 25 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.bvly2ngV4j/pure2-is-with-free-functions-predicate.cpp2"
[[nodiscard]] auto pred_d(cpp2::impl::in<double> x) -> bool;

#line 29 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.bvly2ngV4j/pure2-is-with-free-functions-predicate.cpp2"
[[nodiscard]] auto pred_(auto const& x) -> bool;

//=== Cpp2 function definitions =================================================

#line 1 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.bvly2ngV4j/pure2-is-with-free-functions-predicate.cpp2"
auto fun(auto const& v) -> void{
#line 2 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.bvly2ngV4j/pure2-is-with-free-functions-predicate.cpp2"
    if (cpp2::impl::is(v, (pred_i))) {
        std::cout << "" + cpp2::to_string(v) + " is integer bigger than 3" << std::endl;
    }

    if (cpp2::impl::is(v, (pred_d))) {
        std::cout << "" + cpp2::to_string(v) + " is double bigger than 3" << std::endl;
    }

    if (cpp2::impl::is(v, (pred_))) {
        std::cout << "" + cpp2::to_string(v) + " is bigger than 3" << std::endl;
    }
}

#line 15 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.bvly2ngV4j/pure2-is-with-free-functions-predicate.cpp2"
[[nodiscard]] auto main() -> int{
    fun(3.14);
    fun(42);
    fun('a');
}

#line 21 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.bvly2ngV4j/pure2-is-with-free-functions-predicate.cpp2"
[[nodiscard]] auto pred_i(cpp2::impl::in<int> x) -> bool{
    return cpp2::impl::cmp_greater(x,3); 
}

#line 25 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.bvly2ngV4j/pure2-is-with-free-functions-predicate.cpp2"
[[nodiscard]] auto pred_d(cpp2::impl::in<double> x) -> bool{
    return cpp2::impl::cmp_greater(x,3); 
}

#line 29 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.bvly2ngV4j/pure2-is-with-free-functions-predicate.cpp2"
[[nodiscard]] auto pred_(auto const& x) -> bool{
    return cpp2::impl::cmp_greater(x,3); 
}

