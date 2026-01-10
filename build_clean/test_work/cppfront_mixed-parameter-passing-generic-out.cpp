#include <iostream>


//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================


struct X {
    X(int) { }
};
auto operator<<(std::ostream& o, X const&) -> std::ostream& {
    o << "exxxx";
    return o;
}

auto f(auto x_) -> void;

[[nodiscard]] auto main() -> int;

//=== Cpp2 function definitions =================================================


auto f(auto x_) -> void{
    auto x = cpp2::impl::out(x_); 
    x.construct(42);
}

[[nodiscard]] auto main() -> int{
    cpp2::impl::deferred_init<int> a; 
    f(cpp2::impl::out(&a));
    std::cout << cpp2::move(a.value()) << "\n";

    cpp2::impl::deferred_init<X> b; 
    f(cpp2::impl::out(&b));
    std::cout << cpp2::move(b.value()) << "\n";
}

