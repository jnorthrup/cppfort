

//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================

[[nodiscard]] auto f(cpp2::impl::in<int> a, cpp2::impl::in<int> b) -> int;

[[nodiscard]] auto main() -> int;

//=== Cpp2 function definitions =================================================

[[nodiscard]] auto f(cpp2::impl::in<int> a, cpp2::impl::in<int> b) -> int{
    auto x {a}; 
    x >>= b * 2;
    return cpp2::move(x) >> 1; 
}

[[nodiscard]] auto main() -> int{
    std::cout << f(32, 1) << "\n";
}

