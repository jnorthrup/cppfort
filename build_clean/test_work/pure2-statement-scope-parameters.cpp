

//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================


auto main(int const argc_, char** argv_) -> int;

//=== Cpp2 function definitions =================================================


auto main(int const argc_, char** argv_) -> int{
    auto const args = cpp2::make_args(argc_, argv_); 
    auto local_int {42}; 
{
auto const& i{local_int};

    //  'in' (read-only) statement scope variable
    for ( [[maybe_unused]] auto const& unnamed_param_1 : args ) {
        std::cout << i << "\n";     // prints 42
    }
}
{
auto& i{local_int};

    //  'inout' (read-write) statement scope variable
    {
        ++i;
    }
}
    std::cout << cpp2::move(local_int) << "\n";// prints 43

    std::chrono::duration<cpp2::i32> auto_1 {0}; 
{
[[maybe_unused]] std::chrono::duration<cpp2::i32> unnamed_param_1{0};
    {}
}
    std::in_place_t auto_2 {}; 
{
[[maybe_unused]] std::in_place_t unnamed_param_1{};
    {}
}
}

