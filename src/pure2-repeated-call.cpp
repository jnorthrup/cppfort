

//=== Cpp2 type declarations ====================================================


#include "cpp2util.h"

#line 1 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.ayDc73iAjo/pure2-repeated-call.cpp2"


//=== Cpp2 type definitions and function declarations ===========================

#line 1 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.ayDc73iAjo/pure2-repeated-call.cpp2"
[[nodiscard]] auto f0() -> auto;
#line 2 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.ayDc73iAjo/pure2-repeated-call.cpp2"
[[nodiscard]] auto f1() -> auto;
[[nodiscard]] auto f2() -> auto;
[[nodiscard]] auto f3() -> auto;
[[nodiscard]] auto f4() -> auto;

[[nodiscard]] auto main() -> int;

//=== Cpp2 function definitions =================================================

#line 1 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.ayDc73iAjo/pure2-repeated-call.cpp2"
[[nodiscard]] auto f0() -> auto{return 42; }
#line 2 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.ayDc73iAjo/pure2-repeated-call.cpp2"
[[nodiscard]] auto f1() -> auto{return f0; }
#line 3 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.ayDc73iAjo/pure2-repeated-call.cpp2"
[[nodiscard]] auto f2() -> auto{return f1; }
#line 4 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.ayDc73iAjo/pure2-repeated-call.cpp2"
[[nodiscard]] auto f3() -> auto{return f2; }
#line 5 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.ayDc73iAjo/pure2-repeated-call.cpp2"
[[nodiscard]] auto f4() -> auto{return f3; }

#line 7 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.ayDc73iAjo/pure2-repeated-call.cpp2"
[[nodiscard]] auto main() -> int{
    std::cout << f4()()()()() << std::endl;
    return 0; 
}

