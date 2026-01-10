

//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================


auto print(auto const& x) -> void;

auto decorate_and_print(auto& x) -> void;

[[nodiscard]] auto main() -> int;


//=== Cpp2 function definitions =================================================


auto print(auto const& x) -> void{
    std::cout << ">> " << x << "\n";
}

auto decorate_and_print(auto& x) -> void{
    x = "[" + x + "]";
    print(x);
}

[[nodiscard]] auto main() -> int{
    std::vector<std::string> words {
        "hello", "big", "world"}; 
    std::span<std::string> view {words}; 

    auto i {cpp2_new<int>(0)}; 
    for( ; cpp2::impl::cmp_less(*cpp2::impl::assert_not_null(i),CPP2_UFCS(ssize)(view)); ++*cpp2::impl::assert_not_null(i) ) {
        print(CPP2_ASSERT_IN_BOUNDS(view, *cpp2::impl::assert_not_null(i)));
    }

    do {
        std::cout << std::setw(4) << "**";
    } while ( [&]{ --*cpp2::impl::assert_not_null(i) ; return true; }() && cpp2::impl::cmp_greater(*cpp2::impl::assert_not_null(i),0));

    std::cout << "\n";
    for ( auto& word : cpp2::move(words) ) 
        decorate_and_print(word);

    print(std::string{"end of program"});
}

