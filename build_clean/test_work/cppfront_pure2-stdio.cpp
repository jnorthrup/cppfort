

//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================


//  "A better C than C" ... ?
//
[[nodiscard]] auto main() -> int;


//=== Cpp2 function definitions =================================================


[[nodiscard]] auto main() -> int{
    std::string s {"Fred"}; 
    auto myfile {fopen("xyzzy", "w")}; 
    static_cast<void>(CPP2_UFCS(fprintf)(myfile, "Hello %s with UFCS!", CPP2_UFCS(c_str)(cpp2::move(s))));
    static_cast<void>(CPP2_UFCS(fclose)(cpp2::move(myfile)));
}

