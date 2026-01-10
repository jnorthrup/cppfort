#include <iostream>
#include <string>


//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================


[[nodiscard]] auto name() -> std::string;

auto decorate(std::string& s) -> void;

auto main() -> int {
    // name();
    std::cout << "Hello " << name() << "\n";
}


//=== Cpp2 function definitions =================================================


[[nodiscard]] auto name() -> std::string{
    std::string s {"world"}; 
    decorate(s);
    return s; 
}

auto decorate(std::string& s) -> void{
    s = "[" + s + "]";
}

