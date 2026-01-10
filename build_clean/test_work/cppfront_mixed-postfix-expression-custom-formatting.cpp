

//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================


auto call([[maybe_unused]] auto const& unnamed_param_1, [[maybe_unused]] auto const& unnamed_param_2, [[maybe_unused]] auto const& unnamed_param_3, [[maybe_unused]] auto const& unnamed_param_4, [[maybe_unused]] auto const& unnamed_param_5) -> void;

[[nodiscard]] auto test(auto const& a) -> std::string;

[[nodiscard]] auto main() -> int;

//=== Cpp2 function definitions =================================================


auto call([[maybe_unused]] auto const& unnamed_param_1, [[maybe_unused]] auto const& unnamed_param_2, [[maybe_unused]] auto const& unnamed_param_3, [[maybe_unused]] auto const& unnamed_param_4, [[maybe_unused]] auto const& unnamed_param_5) -> void{}

[[nodiscard]] auto test(auto const& a) -> std::string{
    return call(a, 
        ++*cpp2::impl::assert_not_null(CPP2_UFCS(b)(a, a.c)), "hello", /* polite
                          greeting
                          goes here */" there", 
        CPP2_UFCS(e)(a.d, ++CPP2_UFCS(g)(*cpp2::impl::assert_not_null(a.f)), // because f is foobar
              CPP2_UFCS(i)(a.h), 
              CPP2_UFCS(j)(a, a.k, a.l))
        ); 
}

[[nodiscard]] auto main() -> int{}

