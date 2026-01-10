

//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================


//  "A better C than C" ... ?
//
[[nodiscard]] auto main() -> int;

//=== Cpp2 function definitions =================================================


[[nodiscard]] auto main() -> int{
    std::string s {"Fred"}; 
    auto myfile {cpp2::fopen("xyzzy", "w")}; 
    static_cast<void>(CPP2_UFCS(fprintf)(cpp2::move(myfile), "Hello %s with UFCS!", CPP2_UFCS(c_str)(cpp2::move(s))));
}

