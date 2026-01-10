

//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================

auto f([[maybe_unused]] cpp2::impl::in<cpp2::i32> unnamed_param_1) -> void;
auto main() -> int;

//=== Cpp2 function definitions =================================================

auto f([[maybe_unused]] cpp2::impl::in<cpp2::i32> unnamed_param_1) -> void{}
auto main() -> int{
  std::array array_of_functions {f, f}; 
  auto index {0}; 
  cpp2::i32 arguments {0}; 
  CPP2_ASSERT_IN_BOUNDS(array_of_functions, index)(arguments);
  static_cast<void>(cpp2::move(array_of_functions));
  static_cast<void>(cpp2::move(index));
  static_cast<void>(cpp2::move(arguments));
}

