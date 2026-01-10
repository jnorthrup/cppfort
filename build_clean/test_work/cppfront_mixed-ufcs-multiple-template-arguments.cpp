#include <string>
#include <iostream>


//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================


template <auto from, auto to>
auto substr(const std::string& input) -> std::string {
    return input.substr(from, to-from);
}

struct X {
    std::string input;
    X(const std::string& input) : input{input} { }
    template <auto from, auto to>
    auto substr() -> std::string {
        return input.substr(from, to-from);
    }
};

[[nodiscard]] auto main() -> int;

//=== Cpp2 function definitions =================================================


[[nodiscard]] auto main() -> int{
    std::string test_string {"The rain in Spain flows mainly down the drain"}; 
    std::cout << substr<4,8>(test_string) << "\n";

    X x {cpp2::move(test_string)}; 
    std::cout << CPP2_UFCS_TEMPLATE(substr<4,8>)(cpp2::move(x)) << "\n";
}

