

//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================

template<typename T> auto inline constexpr v{ 0 };
auto main() -> int;

//=== Cpp2 function definitions =================================================


auto main() -> int { static_cast<void>(v<cpp2::i32> == v<cpp2::i64>);  }

