

//=== Cpp2 type declarations ====================================================




class t;
  

//=== Cpp2 type definitions and function declarations ===========================

using u = std::array<cpp2::i32,2>;
class t: public std::integral_constant<u,u{17, 29}> {

};
auto main() -> int;

//=== Cpp2 function definitions =================================================


auto main() -> int{
  if (cpp2::testing.is_active() && !(CPP2_ASSERT_IN_BOUNDS_LITERAL(t::value, 0) == 17) ) { cpp2::testing.report_violation(""); }
  if (cpp2::testing.is_active() && !(CPP2_ASSERT_IN_BOUNDS_LITERAL(t::value, 1) == 29) ) { cpp2::testing.report_violation(""); }
}

