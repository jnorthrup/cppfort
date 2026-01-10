

//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================


using vals_ret = int;
[[nodiscard]] auto vals() -> vals_ret;

[[nodiscard]] auto main() -> int;

//=== Cpp2 function definitions =================================================

[[nodiscard]] auto vals() -> vals_ret{
        cpp2::impl::deferred_init<int> i;
    i.construct(42);
    return std::move(i.value()); 
}

[[nodiscard]] auto main() -> int{
    auto v {vals()}; 
    static_cast<void>(cpp2::move(v));
}

