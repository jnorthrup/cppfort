

//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================

auto main() -> int;

//=== Cpp2 function definitions =================================================

auto main() -> int { 
    std::cout << "" + cpp2::to_string(std::is_void_v<cpp2::i32*> && std::is_void_v<cpp2::i32 const>) + "\n";  }

