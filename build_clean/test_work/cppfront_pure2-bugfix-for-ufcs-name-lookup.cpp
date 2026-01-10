

//=== Cpp2 type declarations ====================================================



class identity;
  

class t;
  

class u;

namespace ns {

}


//=== Cpp2 type definitions and function declarations ===========================

class identity {
  public: [[nodiscard]] constexpr auto operator()(auto&& x) const& -> decltype(auto);
};

class t {
  public: [[nodiscard]] constexpr auto f() const& -> int;
};

class u {};

namespace ns {
[[nodiscard]] constexpr auto f([[maybe_unused]] auto const& unnamed_param_1) -> int;
} // namespace ns

// v: @struct type = {
//   f :== :(_) 0; // Pending on #706.
//   g: (i) i.f();
// }

auto main() -> int;

//=== Cpp2 function definitions =================================================


  [[nodiscard]] constexpr auto identity::operator()(auto&& x) const& -> decltype(auto) { return CPP2_FORWARD(x);  }

  [[nodiscard]] constexpr auto t::f() const& -> int { return 0;  }

namespace ns {
[[nodiscard]] constexpr auto f([[maybe_unused]] auto const& unnamed_param_1) -> int { return 1;  }
}

auto main() -> int{
  {
    int f {CPP2_UFCS(f)(t())}; 
    if (cpp2::cpp2_default.is_active() && !(cpp2::move(f) == 0) ) { cpp2::cpp2_default.report_violation(""); }
  }
  {
    auto f {t().f()}; 
    if (cpp2::cpp2_default.is_active() && !(cpp2::move(f) == 0) ) { cpp2::cpp2_default.report_violation(""); }
  }
  {
    auto f {t().f()}; 
    if (cpp2::cpp2_default.is_active() && !(cpp2::move(f) == 0) ) { cpp2::cpp2_default.report_violation(""); }
  }
{
auto const& f{t().f()};
  if (cpp2::cpp2_default.is_active() && !(f == 0) ) { cpp2::cpp2_default.report_violation(""); }
}
{
auto const& f{t().f()};
  if (cpp2::cpp2_default.is_active() && !(f == 0) ) { cpp2::cpp2_default.report_violation(""); }
}
  {
    auto constexpr f{ t().f() };
    static_assert(f == 0);
  }
  {
    auto constexpr f{ t().f() };
    static_assert(f == 0);
  }
  {
    auto f {[](auto const& f) -> auto{
      if (cpp2::cpp2_default.is_active() && !(CPP2_UFCS(f)(t()) == 0) ) { cpp2::cpp2_default.report_violation(""); }
      return CPP2_UFCS(f)(u()); 
    }(identity())}; 
    static_cast<void>(cpp2::move(f));
  }
  { // Rejected by MSVC.
    // f := :<f: _> (copy _: std::integral_constant<identity, f>) -> _ = {
    //   assert(t().f() == 0);
    //   return u().f();
    // }(:std::integral_constant<identity, (:identity = ())> = ());
    // _ = f;
  }
  {
    auto f {[]() -> void{
{
cpp2::impl::in<identity> f{identity()};
      if (cpp2::cpp2_default.is_active() && !(CPP2_UFCS(f)(t()) == 0) ) { cpp2::cpp2_default.report_violation(""); }
}
{
cpp2::impl::in<identity> f{identity()};
      static_cast<void>(CPP2_UFCS(f)(u()));
}
    }}; 
    static_cast<void>(cpp2::move(f));
  }
  {
    auto f {[]() -> void{
      using ns::f;
      static_assert(CPP2_UFCS(f)(t()) == 0);
      // static_assert(u().f() == 1);
    }}; 
    static_cast<void>(cpp2::move(f));
  }
  {
    auto f {[]() -> void{
      static_assert(t().f() == 0);
      auto g {[]<typename T>([[maybe_unused]] T const& unnamed_param_1) -> std::void_t<decltype(T().f())>{}}; 
      static_assert(!(std::is_invocable_v<decltype(cpp2::move(g)),u>));
      using ns::f;
    }}; 
    static_cast<void>(cpp2::move(f));
  }
  {
    auto f {[]() -> void{
      using ns::f;
      static_cast<void>([]() -> void{
        static_assert(CPP2_UFCS(f)(t()) == 0);
        // static_assert(u().f() == 1);
      });
      {
        static_assert(CPP2_UFCS(f)(t()) == 0);
        // static_assert(u().f() == 1);
      }
    }}; 
    static_cast<void>(cpp2::move(f));
  }
  {
    auto f {[]() -> void{
      static_cast<void>([]() -> void{
        static_assert(t().f() == 0);
        auto g {[]<typename T>([[maybe_unused]] T const& unnamed_param_1) -> std::void_t<decltype(T().f())>{}}; 
        static_assert(!(std::is_invocable_v<decltype(cpp2::move(g)),u>));
      });
      {
        static_assert(t().f() == 0);
        auto g {[]<typename T>([[maybe_unused]] T const& unnamed_param_1) -> std::void_t<decltype(T().f())>{}}; 
        static_assert(!(std::is_invocable_v<decltype(cpp2::move(g)),u>));
      }
      using ns::f;
    }}; 
    static_cast<void>(cpp2::move(f));
  }
}

