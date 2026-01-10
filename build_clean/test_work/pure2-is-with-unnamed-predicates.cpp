

//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================

auto fun(auto const& v) -> void;

[[nodiscard]] auto main() -> int;

//=== Cpp2 function definitions =================================================

auto fun(auto const& v) -> void{
    if (cpp2::impl::is(v, ([](cpp2::impl::in<int> x) -> decltype(auto) { return cpp2::impl::cmp_greater(x,3); }))) {
        std::cout << "" + cpp2::to_string(v) + " is integer bigger than 3" << std::endl;
    }

    if (cpp2::impl::is(v, ([](cpp2::impl::in<double> x) -> decltype(auto) { return cpp2::impl::cmp_greater(x,3); }))) {
        std::cout << "" + cpp2::to_string(v) + " is double bigger than 3" << std::endl;
    }

    if (cpp2::impl::is(v, ([](auto const& x) -> decltype(auto) { return cpp2::impl::cmp_greater(x,3); }))) {
        std::cout << "" + cpp2::to_string(v) + " is bigger than 3" << std::endl;
    }
}

[[nodiscard]] auto main() -> int{
    fun(3.14);
    fun(42);
    fun('a');
}

