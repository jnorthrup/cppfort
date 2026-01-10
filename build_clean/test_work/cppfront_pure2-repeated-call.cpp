

//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================

[[nodiscard]] auto f0() -> auto;
[[nodiscard]] auto f1() -> auto;
[[nodiscard]] auto f2() -> auto;
[[nodiscard]] auto f3() -> auto;
[[nodiscard]] auto f4() -> auto;

[[nodiscard]] auto main() -> int;

//=== Cpp2 function definitions =================================================

[[nodiscard]] auto f0() -> auto{return 42; }
[[nodiscard]] auto f1() -> auto{return f0; }
[[nodiscard]] auto f2() -> auto{return f1; }
[[nodiscard]] auto f3() -> auto{return f2; }
[[nodiscard]] auto f4() -> auto{return f3; }

[[nodiscard]] auto main() -> int{
    std::cout << f4()()()()() << std::endl;
    return 0; 
}

