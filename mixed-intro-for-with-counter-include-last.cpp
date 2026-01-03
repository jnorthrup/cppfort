

//=== Cpp2 type declarations ====================================================


#include "cpp2util.h"

#line 1 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.WW5HK3JRPO/mixed-intro-for-with-counter-include-last.cpp2"


//=== Cpp2 type definitions and function declarations ===========================

#line 1 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.WW5HK3JRPO/mixed-intro-for-with-counter-include-last.cpp2"

#line 2 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.WW5HK3JRPO/mixed-intro-for-with-counter-include-last.cpp2"
[[nodiscard]] auto main() -> int;
#line 10 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.WW5HK3JRPO/mixed-intro-for-with-counter-include-last.cpp2"

#include <vector>
#include <span>
#include <iostream>


//=== Cpp2 function definitions =================================================

#line 1 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.WW5HK3JRPO/mixed-intro-for-with-counter-include-last.cpp2"

#line 2 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.WW5HK3JRPO/mixed-intro-for-with-counter-include-last.cpp2"
[[nodiscard]] auto main() -> int
{
    std::vector<int> v {1, 2, 3, 4, 5}; 
    auto counter {42}; 
    for (                         auto const& i : cpp2::move(v) )  { do {
        std::cout << i << " " << counter << "\n";
    } while (false); counter *= 2; }
}

