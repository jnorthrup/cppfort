

//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================


[[nodiscard]] auto f() -> decltype(auto);

auto g2() -> void;
[[nodiscard]] auto g() -> decltype(auto);

[[nodiscard]] auto h() -> decltype(auto);

auto main() -> int;

//=== Cpp2 function definitions =================================================


[[nodiscard]] auto f() -> decltype(auto) { return std::cout << "hi";  }

auto g2() -> void{}
[[nodiscard]] auto g() -> decltype(auto) { return g2();  }

[[nodiscard]] auto h() -> decltype(auto) { return cpp2::impl::cmp_greater(2,0);  }

auto main() -> int{
    f() << " ho";
    static_assert(std::is_same_v<decltype(g()),void>);
    if (h()) {std::cout << " hum"; }
}

