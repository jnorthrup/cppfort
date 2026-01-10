

//=== Cpp2 type declarations ====================================================



class t;
  

//=== Cpp2 type definitions and function declarations ===========================

class t {
  public: auto operator[]([[maybe_unused]] auto const& unnamed_param_2) const& -> void;
  public: t() = default;
  public: t(t const&) = delete; /* No 'that' constructor, suppress copy */
  public: auto operator=(t const&) -> void = delete;

};

[[nodiscard]] auto main() -> int;

auto inline constexpr x{ cpp2::i32{0} };
extern cpp2::i32 y;

//=== Cpp2 function definitions =================================================


  auto t::operator[]([[maybe_unused]] auto const& unnamed_param_2) const& -> void{}

[[nodiscard]] auto main() -> int{
{
auto const& x{t()};
  {CPP2_ASSERT_IN_BOUNDS(x, []() -> auto { return 0;  }); }
}
{
auto const& x{t()};
  {CPP2_ASSERT_IN_BOUNDS(x, []() -> auto { return 0;  }); }
}

  if (cpp2::cpp2_default.is_active() && !(!((cpp2::impl::is<int>([]() -> decltype(auto) { return 0; })))) ) { cpp2::cpp2_default.report_violation(""); }

  return cpp2::i32{0}; 
}

cpp2::i32 y {0}; 

