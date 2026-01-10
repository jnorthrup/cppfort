

//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================

auto cxx(int x, std::string y) -> void {
    std::cout << x << " \"" << y << "\"" << "\n";
}

auto cxx2(cpp2::impl::in<int> x, cpp2::impl::in<std::string> y) -> void;

auto main() -> int;

//=== Cpp2 function definitions =================================================


auto cxx2(cpp2::impl::in<int> x, cpp2::impl::in<std::string> y) -> void{
    std::cout << "" + cpp2::to_string(x) + " \"" + cpp2::to_string(y) + "\"\n";
}

auto main() -> int{
 cxx(1, "test");
 cxx({}, {});
 cxx2(1, "test");
 cxx2({}, {});
}

