

//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================

extern std::plus<> const plus;
[[nodiscard]] auto main() -> int;

//=== Cpp2 function definitions =================================================

std::plus<> const plus {}; 
[[nodiscard]] auto main() -> int { return std::plus<>()(0, 0); }

