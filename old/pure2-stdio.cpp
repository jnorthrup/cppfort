

//=== Cpp2 type declarations ====================================================


#include "cpp2util.h"

#line 1 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.LSfLCqd0aM/pure2-stdio.cpp2"


//=== Cpp2 type definitions and function declarations ===========================

#line 1 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.LSfLCqd0aM/pure2-stdio.cpp2"

//  "A better C than C" ... ?
//
#line 4 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.LSfLCqd0aM/pure2-stdio.cpp2"
[[nodiscard]] auto main() -> int;
#line 10 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.LSfLCqd0aM/pure2-stdio.cpp2"

#line 1 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.LSfLCqd0aM/pure2-stdio.cpp2"

//=== Cpp2 function definitions =================================================

#line 1 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.LSfLCqd0aM/pure2-stdio.cpp2"

#line 4 "/var/folders/kd/t4gdhlf137n0zbd6x34z1hcm0000gn/T/tmp.LSfLCqd0aM/pure2-stdio.cpp2"
[[nodiscard]] auto main() -> int{
    std::string s {"Fred"}; 
    auto myfile {fopen("xyzzy", "w")}; 
    static_cast<void>(CPP2_UFCS(fprintf)(myfile, "Hello %s with UFCS!", CPP2_UFCS(c_str)(cpp2::move(s))));
    static_cast<void>(CPP2_UFCS(fclose)(cpp2::move(myfile)));
}

