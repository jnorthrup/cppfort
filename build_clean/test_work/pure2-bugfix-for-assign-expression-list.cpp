

//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================

auto main() -> int;

//=== Cpp2 function definitions =================================================

auto main() -> int{
  using vec = std::vector<int>;
  vec v {0}; 
  v                   = {  };
  if (cpp2::cpp2_default.is_active() && !(v == vec{}) ) { cpp2::cpp2_default.report_violation(""); }
  v                   = { 1 };
  if (cpp2::cpp2_default.is_active() && !(v == vec{1}) ) { cpp2::cpp2_default.report_violation(""); }
  v                   = { 2, 3 };
  if (cpp2::cpp2_default.is_active() && !(cpp2::move(v) == vec{2, 3}) ) { cpp2::cpp2_default.report_violation(""); }
}

