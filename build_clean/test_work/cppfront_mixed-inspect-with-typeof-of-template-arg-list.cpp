

//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================


template <int i, int j>
auto calc() {
    return i + j;
}

[[nodiscard]] auto fun(auto const& v) -> int;

[[nodiscard]] auto main() -> int;

//=== Cpp2 function definitions =================================================


[[nodiscard]] auto fun(auto const& v) -> int{
    return [&] () -> int { auto&& _expr = v;
        if (cpp2::impl::is<int>(_expr)) { if constexpr( requires{calc<1,2>();} ) if constexpr( std::is_convertible_v<CPP2_TYPEOF((calc<1,2>())),int> ) return calc<1,2>(); else return int{}; else return int{}; }
        else return 0; }
    (); 
}

[[nodiscard]] auto main() -> int{
    return fun(42); 
}

