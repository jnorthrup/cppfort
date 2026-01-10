

//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================

auto main() -> int;

//=== Cpp2 function definitions =================================================

auto main() -> int{
    if (cpp2::cpp2_default.is_active() && !(true) ) { cpp2::cpp2_default.report_violation(CPP2_CONTRACT_MSG("some_potentially_long_string")); }
}

