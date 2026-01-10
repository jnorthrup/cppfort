

//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================

auto fun(auto const& v) -> void;

[[nodiscard]] auto main() -> int;

[[nodiscard]] auto pred_i(cpp2::impl::in<int> x) -> bool;

[[nodiscard]] auto pred_d(cpp2::impl::in<double> x) -> bool;

[[nodiscard]] auto pred_(auto const& x) -> bool;

//=== Cpp2 function definitions =================================================

auto fun(auto const& v) -> void{
    if (cpp2::impl::is(v, (pred_i))) {
        std::cout << "" + cpp2::to_string(v) + " is integer bigger than 3" << std::endl;
    }

    if (cpp2::impl::is(v, (pred_d))) {
        std::cout << "" + cpp2::to_string(v) + " is double bigger than 3" << std::endl;
    }

    if (cpp2::impl::is(v, (pred_))) {
        std::cout << "" + cpp2::to_string(v) + " is bigger than 3" << std::endl;
    }
}

[[nodiscard]] auto main() -> int{
    fun(3.14);
    fun(42);
    fun('a');
}

[[nodiscard]] auto pred_i(cpp2::impl::in<int> x) -> bool{
    return cpp2::impl::cmp_greater(x,3); 
}

[[nodiscard]] auto pred_d(cpp2::impl::in<double> x) -> bool{
    return cpp2::impl::cmp_greater(x,3); 
}

[[nodiscard]] auto pred_(auto const& x) -> bool{
    return cpp2::impl::cmp_greater(x,3); 
}

