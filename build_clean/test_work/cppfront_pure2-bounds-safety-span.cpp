

//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================


[[nodiscard]] auto main() -> int;

auto print_and_decorate(auto const& x) -> void;


//=== Cpp2 function definitions =================================================


[[nodiscard]] auto main() -> int
{
    std::vector<std::string> words {"decorated", "hello", "world"}; 

    std::span<std::string> s {words}; 
    static_cast<void>(cpp2::move(words));

    auto i {0}; 
    for( ; cpp2::impl::cmp_less(i,CPP2_UFCS(ssize)(s)); ++i ) {
        print_and_decorate(CPP2_ASSERT_IN_BOUNDS(s, i));
    }
}

auto print_and_decorate(auto const& x) -> void{
    std::cout << ">> " << x << "\n";
}

