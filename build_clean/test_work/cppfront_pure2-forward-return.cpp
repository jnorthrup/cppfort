

//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================


[[nodiscard]] auto first(auto&& rng) -> decltype(auto);

extern int const global;
[[nodiscard]] auto f() -> int const&;

[[nodiscard]] auto main() -> int;

//=== Cpp2 function definitions =================================================


[[nodiscard]] auto first(auto&& rng) -> decltype(auto) { 
    if (cpp2::bounds_safety.is_active() && !(!(std::empty(rng))) ) { cpp2::bounds_safety.report_violation(""); }

    return *cpp2::impl::assert_not_null(std::begin(CPP2_FORWARD(rng)));  }

int const global {42}; 
[[nodiscard]] auto f() -> int const&{return global; }

[[nodiscard]] auto main() -> int{
    std::vector v {1, 2, 3}; 
    first(v) = 4;
    std::cout << first(cpp2::move(v)) << std::endl;// prints: 4
    std::cout << f() << std::endl; // prints: 42
}

