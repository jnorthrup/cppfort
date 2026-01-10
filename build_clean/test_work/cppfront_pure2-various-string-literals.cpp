

//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================

auto main() -> int;

//=== Cpp2 function definitions =================================================

auto main() -> int{
    auto s1 {u"u\""}; 
    auto s2 {U"U\""}; 
    auto s3 {u8"u8\""}; 
    auto s4 {L"L\""}; 
    auto s5 {R"(R")"}; 
    auto s6 {uR"(uR")"}; 
    auto s7 {u8R"(u8R")"}; 
    auto s8 {UR"(UR")"}; 
    auto s9 {LR"(LR")"}; 
    static_cast<void>(cpp2::move(s1));
    static_cast<void>(cpp2::move(s2));
    static_cast<void>(cpp2::move(s3));
    static_cast<void>(cpp2::move(s4));
    static_cast<void>(cpp2::move(s5));
    static_cast<void>(cpp2::move(s6));
    static_cast<void>(cpp2::move(s7));
    static_cast<void>(cpp2::move(s8));
    static_cast<void>(cpp2::move(s9));
}

