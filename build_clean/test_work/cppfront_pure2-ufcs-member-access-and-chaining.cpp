

//=== Cpp2 type declarations ====================================================




class mytype;
    

//=== Cpp2 type definitions and function declarations ===========================

[[nodiscard]] auto main() -> int;

auto no_return([[maybe_unused]] auto const& unnamed_param_1) -> void;

[[nodiscard]] auto ufcs(cpp2::impl::in<int> i) -> int;
using fun_ret = int;


[[nodiscard]] auto fun() -> fun_ret;

[[nodiscard]] auto get_i(auto const& r) -> int;

//  And a test for non-local UFCS, which shouldn't do a [&] capture
[[nodiscard]] auto f([[maybe_unused]] auto const& unnamed_param_1) -> int;
extern int y;

class mytype {
    public: static auto hun([[maybe_unused]] auto const& unnamed_param_1) -> void;
    public: mytype() = default;
    public: mytype(mytype const&) = delete; /* No 'that' constructor, suppress copy */
    public: auto operator=(mytype const&) -> void = delete;

};


//=== Cpp2 function definitions =================================================

[[nodiscard]] auto main() -> int{
    auto i {42}; 
    static_cast<void>(CPP2_UFCS(ufcs)(cpp2::move(i)));

    auto j {fun()}; 
    static_cast<void>(CPP2_UFCS(ufcs)(j));

    static_cast<void>(CPP2_UFCS(ufcs)(fun()));

    auto k {fun()}; 
    static_cast<void>(CPP2_UFCS(ufcs)(cpp2::move(k)));

    static_cast<void>(CPP2_UFCS(ufcs)(get_i(j)));

    static_cast<void>(CPP2_UFCS(ufcs)(get_i(fun())));

    auto res {CPP2_UFCS(ufcs)((42))}; 

    static_cast<void>(CPP2_UFCS(ufcs)((cpp2::move(j))));

    CPP2_UFCS(no_return)(42);

    CPP2_UFCS(no_return)(cpp2::move(res));

    mytype obj {}; 
    cpp2::move(obj).hun(42);// explicit non-UFCS
}

auto no_return([[maybe_unused]] auto const& unnamed_param_1) -> void{}

[[nodiscard]] auto ufcs(cpp2::impl::in<int> i) -> int{
    return i + 2; 
}

[[nodiscard]] auto fun() -> fun_ret{
        cpp2::impl::deferred_init<int> i;
    i.construct(42);
    return std::move(i.value()); 
}

[[nodiscard]] auto get_i(auto const& r) -> int{
    return r; 
}

[[nodiscard]] auto f([[maybe_unused]] auto const& unnamed_param_1) -> int { return 0;  }
int y {CPP2_UFCS_NONLOCAL(f)(0)}; 

    auto mytype::hun([[maybe_unused]] auto const& unnamed_param_1) -> void{}

