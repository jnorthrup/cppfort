

//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================

[[nodiscard]] auto main() -> int;

[[nodiscard]] auto decorate(auto& x) -> int;

auto print_it(auto const& x, auto const& len) -> void;

//=== Cpp2 function definitions =================================================

[[nodiscard]] auto main() -> int{
    std::vector<std::string> vec {
            "hello", "2022"}; 

    for ( auto& str : cpp2::move(vec) ) {
        auto len {decorate(str)}; 
        print_it(str, cpp2::move(len));
    }
}

[[nodiscard]] auto decorate(auto& x) -> int{
    x = "[" + x + "]";
    return CPP2_UFCS(ssize)(x); 
}

auto print_it(auto const& x, auto const& len) -> void{
    std::cout 
        << ">> " << x 
        << " - length " 
        << len << "\n";
}

