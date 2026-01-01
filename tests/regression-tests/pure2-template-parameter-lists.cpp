#include "cpp2util.h"

auto f1(T t, U u) -> void;
auto f2(T t, U u) -> void;
auto f3() -> void;
auto f4() -> void;

template<typename T, typename U>
auto f1(T t, U u) -> void {
    t + u;
}

template<typename T, typename U>
auto f2(T t, U u) -> void {
    t + u;
}

template<typename T, typename U>
auto f3() -> void {
    T + U;
}

template<typename T, typename U>
auto f4() -> void {
    T + U;
}

auto main() -> void {
    std::cout << "f1: (f1(1,1))$\n";
    std::cout << "f2: (f2(2,2))$\n";
    std::cout << "f3: (f3<3,3>())$\n";
    std::cout << "f4: (f4<4,4>())$\n";
}

