

//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================


[[nodiscard]] auto main() -> int;

[[nodiscard]] auto name() -> std::string;

auto decorate(std::string& s) -> void;

//=== Cpp2 function definitions =================================================


[[nodiscard]] auto main() -> int{
    std::cout << "Hello " << name() << "\n";
}

[[nodiscard]] auto name() -> std::string{
    std::string s {"world"}; 
    decorate(s);
    return s; 
}

auto decorate(std::string& s) -> void{
    s = "[" + s + "]";
}

