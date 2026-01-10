#include <vector>
#include <ranges>
#include <string>
#include <span>
#include <algorithm>
#include <iostream>


//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================


[[nodiscard]] auto main() -> int;

//=== Cpp2 function definitions =================================================


[[nodiscard]] auto main() -> int{
    std::vector<std::string> vec {
            "hello", "2022"}; 

    auto y {"\n"}; 
    std::ranges::for_each
        (vec, [_0 = cpp2::move(y)](auto const& x) mutable -> decltype(auto) { return std::cout << _0 << x << _0;  });

    auto callback {[](auto& x) -> decltype(auto) { return x += "-ish";  }}; 
    std::ranges::for_each(vec, cpp2::move(callback));

    for ( auto const& str : cpp2::move(vec) ) 
        std::cout << str << "\n";
}

