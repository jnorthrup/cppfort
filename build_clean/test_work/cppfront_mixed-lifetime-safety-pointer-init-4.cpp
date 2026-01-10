#include <iostream>
#include <random>


//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================


[[nodiscard]] auto main() -> int;

auto print_and_decorate(auto const& x) -> void;

bool flip_a_coin() {
    // Change std::mt19937 to std::random_device for non-deterministic PRNG
    static std::mt19937 rand; 
    return rand() % 2 == 1;
}


//=== Cpp2 function definitions =================================================


[[nodiscard]] auto main() -> int{
    int x {42}; 
    int y {43}; 
    cpp2::impl::deferred_init<int*> p; 

    // ... more code ...
    if (flip_a_coin()) {
        p.construct(&y);
    }
    else {
        p.construct(&x);
    }

    print_and_decorate(*cpp2::impl::assert_not_null(cpp2::move(p.value())));
}

auto print_and_decorate(auto const& x) -> void{
    std::cout << ">> " << x << "\n";
}

