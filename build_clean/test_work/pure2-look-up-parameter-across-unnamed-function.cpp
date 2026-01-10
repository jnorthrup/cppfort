

//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================


using f_ret = int;
[[nodiscard]] auto f() -> f_ret;
using g_ret = int;


[[nodiscard]] auto g() -> g_ret;

[[nodiscard]] auto main() -> int;

//=== Cpp2 function definitions =================================================


[[nodiscard]] auto f() -> f_ret{
    int ri {0};
    auto pred {[](auto const& e) -> decltype(auto) { return e == 1;  }}; 
    ri = 42;
    cpp2::move(pred)(ri);
    return ri; // "return;" is implicit"
}

[[nodiscard]] auto g() -> g_ret{
        cpp2::impl::deferred_init<int> ri;
    ri.construct(0);
    auto pred {[](auto const& e) -> decltype(auto) { return e == 1;  }}; 
    ri.value() = 42;
    cpp2::move(pred)(ri.value());
    return std::move(ri.value()); 
}

[[nodiscard]] auto main() -> int{
    std::cout << f() + g() << "\n";
}

