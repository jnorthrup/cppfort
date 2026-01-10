

//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================

template<typename T> concept arithmetic = std::integral<T> || std::floating_point<T>; 
auto main() -> int;

//=== Cpp2 function definitions =================================================


auto main() -> int      {
  if (cpp2::testing.is_active() && !(arithmetic<cpp2::i32>) ) { cpp2::testing.report_violation(""); }
  if (cpp2::testing.is_active() && !(arithmetic<float>) ) { cpp2::testing.report_violation(""); }
}

